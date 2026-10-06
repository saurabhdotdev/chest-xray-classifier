"""
src/gradcam.py
==============
Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap generation.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).
https://arxiv.org/abs/1610.02391

What Grad-CAM does
------------------
It computes the gradient of the class score w.r.t. the last conv feature map.
The gradient magnitudes (averaged over spatial dims) are used as weights to
create a weighted combination of feature maps → a coarse heatmap showing
"where the model looked" to make its prediction.

Why it matters for medical AI
------------------------------
Heatmaps allow radiologists to verify whether the model attends to anatomically
relevant regions (e.g., lung fields for effusion, cardiac silhouette for
cardiomegaly) vs. spurious shortcuts (e.g., image borders, text annotations).

Run
---
  cd chest-xray-classifier
  python -m src.gradcam

Outputs
-------
  results/figures/gradcam_correct_<label>.png   — correct predictions
  results/figures/gradcam_incorrect_<label>.png — false positives/negatives
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import LABEL_NAMES, get_chexpert_loaders
from src.train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Core
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation for DenseNet121 and ResNet50.

    Usage
    -----
    cam = GradCAM(model, backbone="densenet121")
    heatmap = cam(img_tensor, label_idx)
    """

    def __init__(self, model: nn.Module, backbone: str):
        self.model = model
        self.model.eval()
        self.backbone = backbone

        self._activations: torch.Tensor = None
        self._gradients:   torch.Tensor = None

        # Register hooks on the last conv layer
        target_layer = self._get_target_layer()
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _get_target_layer(self) -> nn.Module:
        """Return the last conv layer for the configured backbone."""
        if self.backbone == "densenet121":
            # Last dense block's normalisation layer (features.norm5 follows denseblock4)
            return self.model.features.denseblock4.denselayer16.conv2
        elif self.backbone == "resnet50":
            return self.model.layer4[-1].conv3
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        img_tensor: torch.Tensor,
        label_idx: int,
        device: torch.device,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.

        Args
        ----
        img_tensor : (1, C, H, W) — normalized image tensor
        label_idx  : which output class to explain (0–3)

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        img_tensor = img_tensor.to(device).requires_grad_(False)

        self.model.zero_grad()
        logits = self.model(img_tensor)              # forward pass
        score  = logits[0, label_idx]               # scalar for class of interest
        score.backward()                             # backward pass

        # Global average pooling of gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Image Utilities
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor (C, H, W) → uint8 RGB numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args
    ----
    img_rgb : (H, W, 3) uint8
    heatmap : (H', W') float32 in [0, 1]
    alpha   : heatmap transparency

    Returns
    -------
    overlaid : (H, W, 3) uint8
    """
    H, W = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Apply colormap (JET: blue=low, red=high)
    heatmap_uint8  = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlaid = (1 - alpha) * img_rgb + alpha * heatmap_colored
    return np.clip(overlaid, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Sample Collection
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_per_label: int = 8,
) -> Dict[int, Dict[str, list]]:
    """
    Collect correct and incorrect predictions for each label.

    Returns
    -------
    samples[label_idx]["correct"]   : list of (img_tensor, true_label, pred_prob)
    samples[label_idx]["incorrect"] : same for wrong predictions
    """
    model.eval()
    samples = {
        i: {"correct": [], "incorrect": []}
        for i in range(len(LABEL_NAMES))
    }
    target = max(n_per_label * 2, 16)  # collect extra, then subsample

    for imgs, labels, _ in tqdm(loader, desc="  collecting samples"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.sigmoid(logits).cpu()

        for b in range(imgs.size(0)):
            for i in range(len(LABEL_NAMES)):
                true = int(labels[b, i].item())
                pred = int((probs[b, i] > 0.5).item())
                bucket = "correct" if true == pred else "incorrect"
                if len(samples[i][bucket]) < target:
                    samples[i][bucket].append((
                        imgs[b].cpu(),
                        true,
                        float(probs[b, i].item()),
                    ))

        # Stop when we have enough for all labels
        if all(
            len(samples[i]["correct"]) >= target and
            len(samples[i]["incorrect"]) >= target
            for i in range(len(LABEL_NAMES))
        ):
            break

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

RESULT_COLORS = {"correct": "#4CAF50", "incorrect": "#F44336"}

def plot_gradcam_grid(
    samples: list,
    cam: GradCAM,
    label_idx: int,
    label_name: str,
    result_type: str,  # "correct" or "incorrect"
    n: int,
    alpha: float,
    save_path: Path,
    device: torch.device,
):
    """
    Plot a grid of (original | heatmap | overlay) for n samples.
    """
    n = min(n, len(samples))
    if n == 0:
        logger.warning(f"  No {result_type} samples for {label_name}")
        return

    fig_width = 12
    fig_height = 3.5 * n
    fig, axes = plt.subplots(n, 3, figsize=(fig_width, fig_height))
    if n == 1:
        axes = axes[np.newaxis, :]

    border_color = RESULT_COLORS[result_type]
    title_verb   = "✓ Correct" if result_type == "correct" else "✗ Incorrect"
    fig.suptitle(
        f"Grad-CAM — {label_name} — {title_verb} Predictions",
        fontsize=13, fontweight="bold",
        color=border_color, y=1.01,
    )

    for row, (img_t, true_lbl, pred_prob) in enumerate(samples[:n]):
        img_np = denormalize(img_t)

        # Need grad for Grad-CAM — use single-image batch
        img_batch = img_t.unsqueeze(0)
        heatmap = cam(img_batch, label_idx, device)
        overlay  = overlay_heatmap(img_np, heatmap, alpha=alpha)

        col_data = [
            ("Original", img_np),
            ("Grad-CAM Heatmap", plt.cm.jet(heatmap)[:, :, :3]),
            ("Overlay", overlay),
        ]
        for col, (col_title, img_data) in enumerate(col_data):
            ax = axes[row, col]
            ax.imshow(img_data, cmap="gray" if img_data.ndim == 2 else None)
            if row == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold")
            ax.axis("off")

            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

        # Add probability annotation
        axes[row, 0].set_ylabel(
            f"True: {true_lbl} | P={pred_prob:.3f}",
            fontsize=9, color=border_color, rotation=0,
            labelpad=70, va="center",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_gradcam(cfg: dict):
    device = torch.device(
        cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}. Run train.py first."

    model = build_model(cfg, num_classes=cfg["labels"]["num_classes"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ── Grad-CAM hook setup ─────────────────────────────────────────────────
    backbone = cfg["model"]["backbone"]
    cam = GradCAM(model, backbone=backbone)
    logger.info(f"Grad-CAM initialized for backbone: {backbone}")

    # ── Validation loader ──────────────────────────────────────────────────
    logger.info("\nLoading CheXpert validation set...")
    _, val_loader, _ = get_chexpert_loaders(cfg)

    # ── Collect predictions ────────────────────────────────────────────────
    n_per_label = cfg["gradcam"]["n_samples"]
    alpha       = cfg["gradcam"]["alpha"]

    logger.info(f"\nCollecting {n_per_label} correct + {n_per_label} incorrect per label...")
    samples = collect_predictions(model, val_loader, device, n_per_label=n_per_label)

    # ── Generate heatmaps ──────────────────────────────────────────────────
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating Grad-CAM visualizations...")
    for i, name in enumerate(LABEL_NAMES):
        safe_name = name.replace(" ", "_").lower()
        logger.info(f"\n  {name}:")

        for result_type in ["correct", "incorrect"]:
            plot_gradcam_grid(
                samples=samples[i][result_type],
                cam=cam,
                label_idx=i,
                label_name=name,
                result_type=result_type,
                n=n_per_label,
                alpha=alpha,
                save_path=fig_dir / f"gradcam_{result_type}_{safe_name}.png",
                device=device,
            )

    logger.info(f"\n✅ Grad-CAM complete. All figures in: {fig_dir}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_gradcam(cfg)
