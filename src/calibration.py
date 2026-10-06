"""
src/calibration.py
==================
Post-hoc calibration analysis for the chest X-ray classifier.

What is calibration?
--------------------
A well-calibrated model's predicted probability P(disease) actually reflects
the true frequency of disease. E.g., among patients where the model says
"70% chance of effusion", roughly 70% should actually have effusion.
Poor calibration is a known problem with modern deep neural networks (they
tend to be overconfident), and is clinically critical in medical AI.

This script
-----------
1. Loads the best checkpoint from train.py
2. Runs inference on the internal validation set
3. Plots reliability diagrams (before calibration)
4. Computes Expected Calibration Error (ECE) per label
5. Applies Temperature Scaling (single scalar T fitted on val set)
6. Re-plots reliability diagrams after scaling
7. Reports ECE before vs. after

Run
---
  cd chest-xray-classifier
  python -m src.calibration

Outputs
-------
  results/figures/reliability_before.png
  results/figures/reliability_after.png
  results/metrics/calibration.json
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.calibration import calibration_curve
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


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute ECE: weighted average |confidence - accuracy| over equal-width bins.

    ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|

    Args
    ----
    y_true : (N,) ground truth binary labels
    y_prob : (N,) predicted probabilities in [0, 1]
    n_bins : number of equal-width bins

    Returns
    -------
    ECE as a float in [0, 1]; lower is better.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling (Guo et al., 2017).

    Wraps the trained model and learns a single scalar T > 0 on the val set.
    Scaled logits = original_logits / T

    T > 1 : model was overconfident → probabilities are softened
    T < 1 : model was underconfident (rare for DNNs)

    Only T is optimized; all base model weights are frozen.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Initialize T = 1.0 (identity transform)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        temp = torch.clamp(self.temperature, min=0.05)
        return logits / temp

    def fit(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature on validation set by minimizing BCE loss.
        Returns the fitted temperature value.
        """
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        # Collect all logits & labels first (avoids recomputing every LBFGS step)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in tqdm(loader, desc="  collecting logits"):
                imgs = imgs.to(device)
                logits = self.model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def closure():
            optimizer.zero_grad()
            temp = torch.clamp(self.temperature, min=0.05)
            scaled = all_logits / temp
            loss = criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05)
        T = float(self.temperature.item())
        logger.info(f"  Fitted temperature T = {T:.4f}")
        return T


# ─────────────────────────────────────────────────────────────────────────────
# Inference (returns raw logits + labels)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels, _ in tqdm(loader, desc="  inference"):
        imgs = imgs.to(device)
        logits = model(imgs).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_logits,  axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Reliability Diagrams
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

def plot_reliability_diagrams(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    n_bins: int,
    save_path: Path,
    title: str = "Reliability Diagrams",
):
    """
    Plot reliability diagrams for all 4 labels in a 2x2 grid.

    A perfectly calibrated model lies on the diagonal y=x.
    Bars above diagonal → underconfident; below → overconfident.
    """
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    for i, (name, color) in enumerate(zip(LABEL_NAMES, LABEL_COLORS)):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]

        if y_true.sum() == 0:
            ax.set_title(f"{name}\n(no positives in split)")
            continue

        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Perfect")

        # Bar chart of calibration bins
        bin_width = 1.0 / n_bins
        bin_centers = np.linspace(bin_width / 2, 1 - bin_width / 2, n_bins)

        # Align mean_pred bins to centers for bar chart
        ax.bar(
            mean_pred,
            frac_pos,
            width=bin_width * 0.8,
            color=color,
            alpha=0.55,
            label="Model",
        )
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=5)

        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives",      fontsize=10)
        ax.set_title(f"{name}\nECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Reliability diagram saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(cfg: dict):
    device = torch.device(
        cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")
    n_bins = cfg["evaluation"]["calibration_bins"]

    # ── Load base model ────────────────────────────────────────────────────
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}. Run train.py first."

    base_model = build_model(cfg, num_classes=cfg["labels"]["num_classes"])
    base_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    base_model = base_model.to(device)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ── Validation loader ──────────────────────────────────────────────────
    logger.info("\nLoading CheXpert validation set...")
    _, val_loader, _ = get_chexpert_loaders(cfg)

    # ── Collect raw logits ─────────────────────────────────────────────────
    logger.info("\nCollecting logits (before scaling)...")
    logits_raw, all_labels = collect_logits(base_model, val_loader, device)
    probs_before = 1.0 / (1.0 + np.exp(-logits_raw))  # sigmoid

    # ── ECE before ────────────────────────────────────────────────────────
    ece_before = {}
    logger.info("\nECE BEFORE temperature scaling:")
    for i, name in enumerate(LABEL_NAMES):
        ece = expected_calibration_error(all_labels[:, i], probs_before[:, i], n_bins)
        ece_before[name] = round(ece, 5)
        logger.info(f"  {name:<22}: ECE = {ece:.5f}")

    # ── Reliability diagram before ─────────────────────────────────────────
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagrams(
        all_labels, probs_before, n_bins,
        save_path=fig_dir / "reliability_before.png",
        title="Reliability Diagrams — Before Temperature Scaling",
    )

    # ── Temperature Scaling ────────────────────────────────────────────────
    logger.info("\nFitting temperature scaling on validation set...")
    ts_model = TemperatureScaler(base_model).to(device)
    T = ts_model.fit(val_loader, device)

    # ── Collect logits after scaling ───────────────────────────────────────
    logits_scaled = logits_raw / T
    probs_after = 1.0 / (1.0 + np.exp(-logits_scaled))

    # ── ECE after ─────────────────────────────────────────────────────────
    ece_after = {}
    logger.info(f"\nECE AFTER temperature scaling (T={T:.4f}):")
    for i, name in enumerate(LABEL_NAMES):
        ece = expected_calibration_error(all_labels[:, i], probs_after[:, i], n_bins)
        ece_after[name] = round(ece, 5)
        delta = ece_after[name] - ece_before[name]
        logger.info(f"  {name:<22}: ECE = {ece:.5f}  (Δ {delta:+.5f})")

    # ── Reliability diagram after ──────────────────────────────────────────
    plot_reliability_diagrams(
        all_labels, probs_after, n_bins,
        save_path=fig_dir / "reliability_after.png",
        title=f"Reliability Diagrams — After Temperature Scaling (T={T:.3f})",
    )

    # ── Save calibration metrics ───────────────────────────────────────────
    results = {
        "temperature": round(T, 5),
        "ece_before": ece_before,
        "ece_after":  ece_after,
        "ece_improvement": {
            name: round(ece_before[name] - ece_after[name], 5)
            for name in LABEL_NAMES
        },
    }
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    calib_path = metrics_dir / "calibration.json"
    with open(calib_path, "w") as f:
        json.dump(results, f, indent=2)
    
    backbone = cfg["model"]["backbone"]
    with open(metrics_dir / f"calibration_{backbone}.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nCalibration results saved → {calib_path} and {metrics_dir / f'calibration_{backbone}.json'}")

    # ── Save temperature-scaled model ──────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ts_ckpt = ckpt_dir / "best_model_calibrated.pth"
    torch.save({
        "model_state": base_model.state_dict(),
        "temperature": T,
    }, ts_ckpt)
    torch.save({
        "model_state": base_model.state_dict(),
        "temperature": T,
    }, ckpt_dir / f"best_model_calibrated_{backbone}.pth")
    logger.info(f"Temperature-scaled model saved → {ts_ckpt} and {ckpt_dir / f'best_model_calibrated_{backbone}.pth'}")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'Label':<24} {'ECE Before':>12} {'ECE After':>12} {'Improvement':>14}")
    print("-" * 66)
    for name in LABEL_NAMES:
        imp = results["ece_improvement"][name]
        print(
            f"  {name:<22} {ece_before[name]:>12.5f} "
            f"{ece_after[name]:>12.5f}  {imp:>+12.5f}"
        )
    print("-" * 66)
    print(f"  Fitted Temperature: T = {T:.4f}")

    return results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_calibration(cfg)
