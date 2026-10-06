"""
src/train.py
============
Training loop for the Chest X-ray multi-label classifier.

Features
--------
• DenseNet121 backbone (ResNet50 swappable via config.yaml → model.backbone)
• BCEWithLogitsLoss with per-label positive class weighting
• Automatic Mixed Precision (AMP) for ~2x GPU speedup
• Cosine annealing LR scheduler
• Early stopping on mean validation AUROC
• Checkpointing (saves best model to results/checkpoints/best_model.pth)
• Per-label AUROC and F1 logged every epoch
• All library versions printed at start for reproducibility

Run
---
  cd chest-xray-classifier
  python -m src.train

Expected output (each epoch):
  Epoch 01/30 | Loss: 0.4312 | Val mAUROC: 0.7841
    Cardiomegaly   AUROC: 0.831  F1: 0.612
    Pleural Effusion AUROC: 0.802 F1: 0.591
    ...
  ✓ New best model saved (val mAUROC: 0.7841)
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, f1_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ── Add repo root to path so `src` imports work ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import VERSIONS
from src.data import LABEL_NAMES, get_chexpert_loaders

# Type alias: (loss: float, metrics: dict, labels: np.ndarray, probs: np.ndarray)
Tuple_dict = tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def fix_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # deterministic > speed
    logger.info(f"Random seed fixed to {seed}")


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """
    Build a classification head on top of a pretrained CNN backbone.

    Supported backbones (set via config.yaml → model.backbone):
      "densenet121" — default, standard CheXpert baseline
      "resnet50"    — swap in for comparison; no other code changes needed

    The final FC/classifier layer is replaced with a linear layer outputting
    `num_classes` logits (raw — sigmoid applied at inference/loss time).
    """
    import torchvision.models as tvm

    backbone = cfg["model"]["backbone"]
    pretrained = cfg["model"]["pretrained"]

    # torchvision >= 0.13 uses weights= parameter
    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if backbone == "densenet121":
        model = tvm.densenet121(weights=weights_arg)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif backbone == "resnet50":
        model = tvm.resnet50(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown backbone: {backbone}. Choose 'densenet121' or 'resnet50'."
        )

    logger.info(
        f"Model: {backbone} | pretrained={pretrained} | "
        f"output_classes={num_classes}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,} | Trainable: {trainable:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute per-label AUROC and F1 from probability outputs.

    Returns dict:
      {
        "per_label": {"Cardiomegaly": {"auroc": float, "f1": float}, ...},
        "mean_auroc": float,
        "mean_f1": float,
      }

    AUROC is undefined if only one class is present — those labels are skipped.
    """
    per_label = {}
    aurocs, f1s = [], []

    for i, name in enumerate(LABEL_NAMES):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]
        y_pred = (y_prob >= threshold).astype(int)

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning(f"  {name}: skipped (single class in batch)")
            per_label[name] = {"auroc": float("nan"), "f1": float("nan")}
            continue

        auroc = roc_auc_score(y_true, y_prob)
        f1    = f1_score(y_true, y_pred, zero_division=0)
        aurocs.append(auroc)
        f1s.append(f1)
        per_label[name] = {"auroc": round(auroc, 4), "f1": round(f1, 4)}

    return {
        "per_label": per_label,
        "mean_auroc": round(float(np.mean(aurocs)), 4) if aurocs else float("nan"),
        "mean_f1":    round(float(np.mean(f1s)), 4)    if f1s    else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation Steps
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple_dict:
    """Run validation. Returns (mean_loss, metrics_dict, all_labels, all_probs)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels, _ in tqdm(loader, desc="  val  ", leave=False, dynamic_ncols=True):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(enabled=amp_enabled):
            logits = model(imgs)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)

    metrics = compute_metrics(all_labels, all_probs)
    return total_loss / len(loader), metrics, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation metric hasn't improved for `patience` epochs.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience: int, checkpoint_path: str, mode: str = "max"):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.best_score = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        improved = (
            score > self.best_score if self.mode == "max"
            else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info(
                f"  ✓ New best model saved "
                f"(val mAUROC: {score:.4f}) → {self.checkpoint_path}"
            )
            return True
        else:
            self.counter += 1
            logger.info(
                f"  No improvement ({self.counter}/{self.patience})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    # ── Print library versions for reproducibility log ────────────────────
    logger.info("=" * 60)
    logger.info("Library versions (for reproducibility):")
    for lib, ver in VERSIONS.items():
        logger.info(f"  {lib}: {ver}")
    logger.info("=" * 60)

    fix_seeds(cfg["project"]["seed"])

    device = torch.device(
        cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("\nLoading CheXpert data...")
    train_loader, val_loader, pos_weights = get_chexpert_loaders(cfg)
    pos_weights = pos_weights.to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(cfg, num_classes=cfg["labels"]["num_classes"])
    model = model.to(device)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )
    amp_enabled = cfg["training"]["amp_enabled"] and (device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    # ── Checkpointing ──────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best_model.pth"

    stopper = EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"],
        checkpoint_path=str(ckpt_path),
        mode="max",
    )

    # ── Training history ───────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    logger.info("\n" + "=" * 60)
    logger.info(f"Starting training for {cfg['training']['epochs']} epochs")
    logger.info("=" * 60)
    t_start = time.time()

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        logger.info(f"\nEpoch {epoch:02d}/{cfg['training']['epochs']:02d}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, amp_enabled
        )

        # Validate
        val_loss, metrics, _, _ = validate(
            model, val_loader, criterion, device, amp_enabled
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        logger.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        logger.info(f"  Val mAUROC: {metrics['mean_auroc']:.4f} | "
                    f"Val mF1: {metrics['mean_f1']:.4f}")
        for name, m in metrics["per_label"].items():
            logger.info(f"    {name:22s}: AUROC={m['auroc']:.4f}  F1={m['f1']:.4f}")

        # Early stopping
        stopper(metrics["mean_auroc"], model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(metrics)

        if stopper.should_stop:
            logger.info(f"\nEarly stopping at epoch {epoch}.")
            break

    elapsed = (time.time() - t_start) / 60
    logger.info(f"\nTraining complete in {elapsed:.1f} min")
    logger.info(f"Best val mAUROC: {stopper.best_score:.4f}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # ── Save training history ──────────────────────────────────────────────
    backbone = cfg["model"]["backbone"]
    import shutil
    shutil.copyfile(ckpt_path, ckpt_dir / f"best_model_{backbone}.pth")
    logger.info(f"Backbone-specific checkpoint saved → {ckpt_dir / f'best_model_{backbone}.pth'}")

    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(metrics_dir / f"train_history_{backbone}.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved → {metrics_dir / 'train_history.json'} and {metrics_dir / f'train_history_{backbone}.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
