"""
src/data.py
===========
Dataset classes, patient-level splits, and DataLoaders for:
  - CheXpert (small)  → internal train / val
  - NIH ChestX-ray14  → external test (zero-shot transfer)

Key design decisions (all documented in config.yaml):
  • Splits are by PATIENT ID — never by image — to avoid data leakage.
  • Uncertain labels (-1) in CheXpert use the U-Zeros policy by default
    (treat uncertain as negative), matching the original CheXpert paper.
  • Only the 4 canonical labels are kept; NIH's "Effusion" is renamed to
    "Pleural Effusion" to match CheXpert.
  • Training is capped at max_train_samples for Colab/Kaggle speed.

Usage
-----
From repo root:
    from src.data import get_chexpert_loaders, get_nih_loader
    train_loader, val_loader, class_weights = get_chexpert_loaders(cfg)
    nih_loader = get_nih_loader(cfg)
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Canonical label list (order matters — consistent across all files) ───────
LABEL_NAMES: List[str] = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Atelectasis",
    "Edema",
]

# ── CheXpert column → canonical name ─────────────────────────────────────────
CHEXPERT_COLS: Dict[str, str] = {
    "Cardiomegaly":    "Cardiomegaly",
    "Pleural Effusion": "Pleural Effusion",
    "Atelectasis":     "Atelectasis",
    "Edema":           "Edema",
}

# ── NIH column → canonical name ──────────────────────────────────────────────
# NIH uses "Effusion" instead of "Pleural Effusion"
NIH_COLS: Dict[str, str] = {
    "Cardiomegaly": "Cardiomegaly",
    "Effusion":     "Pleural Effusion",
    "Atelectasis":  "Atelectasis",
    "Edema":        "Edema",
}


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(cfg: dict, is_train: bool) -> transforms.Compose:
    """
    Return torchvision transforms for train or eval mode.

    Augmentation notes:
      - RandomHorizontalFlip: valid for chest X-rays (bilateral symmetry).
      - RandomRotation(10°): small rotations mimic patient positioning variance.
      - NO vertical flip — anatomical orientation must be preserved.
      - ColorJitter: handles scanner brightness/contrast variation.
      - Normalization uses ImageNet statistics (DenseNet/ResNet pretrained).
    """
    aug = cfg["augmentation"]
    img_size = cfg["dataset"]["image_size"]
    mean = aug["normalize_mean"]
    std  = aug["normalize_std"]

    if is_train:
        ops = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if aug.get("horizontal_flip") else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(aug.get("rotation_degrees", 10)),
            transforms.ColorJitter(
                brightness=aug["color_jitter"].get("brightness", 0.2),
                contrast=aug["color_jitter"].get("contrast", 0.2),
                saturation=aug["color_jitter"].get("saturation", 0.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        ops = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# CheXpert Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CheXpertDataset(Dataset):
    """
    CheXpert (small) dataset.

    Label encoding
    --------------
    CheXpert CSV has three values per finding:
        1  → positive
        0  → negative
       -1  → uncertain / unmentioned

    Uncertain policy (set via config.yaml → labels.uncertain_policy):
        "u-zeros" : -1 → 0  [default, conservative, CheXpert paper baseline]
        "u-ones"  : -1 → 1  [pessimistic, useful for ablation]

    Args
    ----
    csv_path   : Path to CheXpert train.csv or valid.csv
    img_root   : Root directory that contains the 'train/' and 'valid/' folders
    transform  : torchvision transform pipeline
    uncertain_policy : "u-zeros" | "u-ones"
    indices    : optional list of integer row indices (for patient-level split)
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        transform: transforms.Compose,
        uncertain_policy: str = "u-zeros",
        indices: List[int] = None,
    ):
        assert uncertain_policy in {"u-zeros", "u-ones"}, \
            f"Unknown uncertain_policy: {uncertain_policy}. Use 'u-zeros' or 'u-ones'."

        self.img_root = Path(img_root)
        self.transform = transform
        self.uncertain_policy = uncertain_policy

        df = pd.read_csv(csv_path)

        # ── Rename columns to canonical label names ───────────────────────
        df = df.rename(columns=CHEXPERT_COLS)

        # ── Keep only needed columns ──────────────────────────────────────
        df = df[["Path"] + LABEL_NAMES].copy()

        # ── Apply uncertain policy ────────────────────────────────────────
        fill_val = 0 if uncertain_policy == "u-zeros" else 1
        df[LABEL_NAMES] = df[LABEL_NAMES].fillna(0).replace(-1, fill_val)
        df[LABEL_NAMES] = df[LABEL_NAMES].astype(np.float32)

        # ── Patient-level subset if indices provided ──────────────────────
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df
        logger.info(
            f"CheXpertDataset loaded | rows={len(df)} | policy={uncertain_policy}"
        )
        self._log_prevalence()

    def _log_prevalence(self):
        for label in LABEL_NAMES:
            pos = self.df[label].sum()
            pct = 100.0 * pos / len(self.df)
            logger.info(f"  {label:20s}: {int(pos):5d} positive ({pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # CheXpert paths start with "CheXpert-v1.0-small/..."
        # Adjust prefix so it resolves from img_root
        img_path = self.img_root / row["Path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback: strip leading path components until it resolves
            parts = Path(row["Path"]).parts
            for start in range(1, len(parts)):
                candidate = self.img_root / Path(*parts[start:])
                if candidate.exists():
                    img = Image.open(candidate).convert("RGB")
                    break
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(row[LABEL_NAMES].values.astype(np.float32))
        return img, labels, str(row["Path"])


# ─────────────────────────────────────────────────────────────────────────────
# NIH ChestX-ray14 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NIHChestDataset(Dataset):
    """
    NIH ChestX-ray14 dataset (external test set — no labels used for training).

    Label format: NIH CSV stores findings as a pipe-separated string, e.g.
        "Atelectasis|Effusion|Cardiomegaly"
    We parse this into binary vectors for the 4 canonical labels.

    NIH → canonical name mapping (defined in NIH_COLS at top of file):
        "Effusion" → "Pleural Effusion"

    Args
    ----
    csv_path  : Path to Data_Entry_2017_v2020.csv
    img_root  : Directory containing all NIH images
    transform : torchvision transform pipeline
    max_samples : cap the number of images (set None for full dataset)
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        transform: transforms.Compose,
        max_samples: int = None,
    ):
        self.img_root = Path(img_root)
        self.transform = transform

        df = pd.read_csv(csv_path)

        # ── Parse pipe-separated findings into binary columns ─────────────
        # NIH canonical columns we care about (before renaming)
        nih_source_cols = list(NIH_COLS.keys())  # ["Cardiomegaly","Effusion","Atelectasis","Edema"]

        for nih_col in nih_source_cols:
            df[nih_col] = df["Finding Labels"].apply(
                lambda x: 1.0 if nih_col in str(x).split("|") else 0.0
            )

        # ── Rename to canonical ────────────────────────────────────────────
        df = df.rename(columns=NIH_COLS)
        df = df[["Image Index"] + LABEL_NAMES].copy()
        df[LABEL_NAMES] = df[LABEL_NAMES].astype(np.float32)

        # ── Optionally cap size ────────────────────────────────────────────
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.df = df
        logger.info(f"NIHChestDataset loaded | rows={len(df)}")
        self._log_prevalence()

    def _log_prevalence(self):
        for label in LABEL_NAMES:
            pos = self.df[label].sum()
            pct = 100.0 * pos / len(self.df)
            logger.info(f"  {label:20s}: {int(pos):5d} positive ({pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_root / row["Image Index"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[LABEL_NAMES].values.astype(np.float32))
        return img, labels, str(row["Image Index"])


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level Split (no leakage!)
# ─────────────────────────────────────────────────────────────────────────────

def patient_split_indices(
    csv_path: str,
    val_fraction: float = 0.15,
    max_train: int = 20_000,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Split CheXpert rows into train/val by PATIENT ID.

    CheXpert image paths encode patient ID, e.g.:
        CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
                                   ^^^^^^^^^^^^^

    We extract the patient ID from the path string, group all images by
    patient, shuffle patients, then split. This guarantees no image from
    the same patient appears in both train and val.

    Returns
    -------
    train_indices, val_indices : lists of integer row indices into the CSV.
    """
    rng = random.Random(seed)
    df = pd.read_csv(csv_path)

    # Extract patient ID from path
    df["patient_id"] = df["Path"].apply(
        lambda p: Path(p).parts[2]  # e.g. "patient00001"
    )

    patients = df["patient_id"].unique().tolist()
    rng.shuffle(patients)

    n_val_patients = int(len(patients) * val_fraction)
    val_patients  = set(patients[:n_val_patients])
    train_patients = set(patients[n_val_patients:])

    val_idx   = df.index[df["patient_id"].isin(val_patients)].tolist()
    train_idx = df.index[df["patient_id"].isin(train_patients)].tolist()

    # Cap training size (shuffle first so the cap is random but reproducible)
    rng.shuffle(train_idx)
    train_idx = train_idx[:max_train]

    logger.info(
        f"Patient split | "
        f"total_patients={len(patients)} | "
        f"train_patients={len(train_patients)} | "
        f"val_patients={len(val_patients)}"
    )
    logger.info(f"  train images (capped): {len(train_idx)}")
    logger.info(f"  val   images         : {len(val_idx)}")

    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights (for BCEWithLogitsLoss pos_weight)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weights(dataset: CheXpertDataset) -> torch.Tensor:
    """
    Compute per-label positive weights for BCEWithLogitsLoss.

        pos_weight[i] = (N - pos_i) / pos_i

    This upweights the positive class, which is typically rare in X-ray data.
    Weights are clamped to [1, 50] to avoid extreme instability.
    """
    labels = np.stack([dataset.df[l].values for l in LABEL_NAMES], axis=1)
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    weights = neg / np.maximum(pos, 1)
    weights = np.clip(weights, 1.0, 50.0)
    logger.info("Positive weights (pos_weight for BCE):")
    for name, w in zip(LABEL_NAMES, weights):
        logger.info(f"  {name:20s}: {w:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — get DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def get_chexpert_loaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build CheXpert train and val DataLoaders with patient-level splits.

    Returns
    -------
    train_loader, val_loader, pos_weights
    """
    ds_cfg   = cfg["dataset"]
    tr_cfg   = cfg["training"]
    lbl_cfg  = cfg["labels"]
    paths    = cfg["paths"]

    # ── Patient-level indices ──────────────────────────────────────────────
    train_idx, val_idx = patient_split_indices(
        csv_path=paths["chexpert_csv_train"],
        val_fraction=ds_cfg["val_split"],
        max_train=ds_cfg["max_train_samples"],
        seed=cfg["project"]["seed"],
    )

    train_tf = build_transforms(cfg, is_train=True)
    val_tf   = build_transforms(cfg, is_train=False)

    train_ds = CheXpertDataset(
        csv_path=paths["chexpert_csv_train"],
        img_root=paths["chexpert_img_root"],
        transform=train_tf,
        uncertain_policy=lbl_cfg["uncertain_policy"],
        indices=train_idx,
    )
    val_ds = CheXpertDataset(
        csv_path=paths["chexpert_csv_train"],
        img_root=paths["chexpert_img_root"],
        transform=val_tf,
        uncertain_policy=lbl_cfg["uncertain_policy"],
        indices=val_idx,
    )

    pos_weights = compute_pos_weights(train_ds)

    num_workers = cfg.get("dataset", {}).get("num_workers", 0)
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, pos_weights


def get_nih_loader(cfg: dict, max_samples: int = 5_000) -> DataLoader:
    """
    Build NIH ChestX-ray14 DataLoader for external evaluation.

    max_samples defaults to 5_000 for Colab/Kaggle speed.
    Set to None to use the full dataset (~112k images).
    """
    paths  = cfg["paths"]
    tr_cfg = cfg["training"]
    num_workers = cfg.get("dataset", {}).get("num_workers", 0)
    pin_mem = torch.cuda.is_available()

    tf = build_transforms(cfg, is_train=False)

    ds = NIHChestDataset(
        csv_path=paths["nih_csv"],
        img_root=paths["nih_img_root"],
        transform=tf,
        max_samples=max_samples,
    )

    return DataLoader(
        ds,
        batch_size=tr_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run: python -m src.data)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("SMOKE TEST: data.py")
    print("=" * 60)

    train_loader, val_loader, pos_weights = get_chexpert_loaders(cfg)

    imgs, labels, paths = next(iter(train_loader))
    print(f"\nTrain batch — images: {imgs.shape}  labels: {labels.shape}")
    print(f"Label names: {LABEL_NAMES}")
    print(f"Positive weights: {pos_weights.numpy().round(2)}")

    imgs_v, labels_v, _ = next(iter(val_loader))
    print(f"Val   batch — images: {imgs_v.shape}  labels: {labels_v.shape}")
    print("\n[OK] data.py smoke test passed.")
