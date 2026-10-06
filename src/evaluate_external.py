"""
src/evaluate_external.py
========================
External validation of the trained model on NIH ChestX-ray14.

This script evaluates WITHOUT retraining — it loads the checkpoint saved by
train.py and runs inference on NIH images. This measures the real-world
performance drop under dataset shift (CheXpert → NIH).

Outputs
-------
• Per-label AUROC on NIH with 95% bootstrap confidence intervals (1000 reps)
• Performance drop table vs. internal val (loaded from train_history.json)
• ROC curves saved to results/figures/
• Metrics JSON saved to results/metrics/external_eval.json

Run
---
  cd chest-xray-classifier
  python -m src.evaluate_external

Expected output:
  ┌──────────────────────┬──────────────┬──────────────────────┬──────────┐
  │ Label                │ Internal Val │ NIH External         │   Drop   │
  ├──────────────────────┼──────────────┼──────────────────────┼──────────┤
  │ Cardiomegaly         │   0.XXXX     │ 0.XXXX [0.XX–0.XX]  │  -0.XXX  │
  │ ...                                                                    │
  └──────────────────────┴──────────────┴──────────────────────┴──────────┘
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import LABEL_NAMES, get_nih_loader
from src.train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute AUROC with bootstrap confidence interval.

    Returns
    -------
    (auroc, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        # Skip if only one class in bootstrap sample
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            aurocs.append(roc_auc_score(yt, yp))
        except Exception:
            continue

    aurocs = np.array(aurocs)
    alpha = 1.0 - ci
    lo = float(np.percentile(aurocs, 100 * alpha / 2))
    hi = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    center = float(np.mean(aurocs))
    return center, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Inference on NIH
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference on a DataLoader.

    Returns
    -------
    all_labels : (N, num_classes) ground truth
    all_probs  : (N, num_classes) sigmoid probabilities
    """
    model.eval()
    all_labels, all_probs = [], []

    for imgs, labels, _ in tqdm(loader, desc="NIH inference"):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_probs,  axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    save_dir: Path,
    prefix: str = "nih",
):
    """Plot per-label ROC curves and save to figures directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("ROC Curves — NIH External Evaluation", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    axes_flat = axes.flatten()

    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]

        if y_true.sum() == 0:
            axes_flat[i].set_title(f"{name}\n(no positives)")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        ax = axes_flat[i]
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / f"{prefix}_roc_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved → {out_path}")


def print_results_table(
    nih_results: dict,
    internal_val_aurocs: dict,
):
    """Pretty-print the internal vs. external AUROC comparison table."""
    header = f"\n{'Label':<24} {'Internal Val':>14} {'NIH External (95% CI)':>26} {'Drop':>8}"
    print(header)
    print("-" * 78)

    for name in LABEL_NAMES:
        int_auroc = internal_val_aurocs.get(name, float("nan"))
        nih_data  = nih_results.get(name, {})
        nih_auroc = nih_data.get("auroc", float("nan"))
        ci_lo     = nih_data.get("ci_lower", float("nan"))
        ci_hi     = nih_data.get("ci_upper", float("nan"))
        drop      = nih_auroc - int_auroc if not (np.isnan(int_auroc) or np.isnan(nih_auroc)) else float("nan")

        drop_str = f"{drop:+.4f}" if not np.isnan(drop) else "  N/A"
        print(
            f"  {name:<22} {int_auroc:>14.4f} "
            f"{nih_auroc:.4f} [{ci_lo:.4f}–{ci_hi:.4f}] "
            f"{drop_str:>8}"
        )
    print("-" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_external(cfg: dict):
    device = torch.device(
        cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Load model from checkpoint ─────────────────────────────────────────
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best_model.pth"
    assert ckpt_path.exists(), (
        f"Checkpoint not found: {ckpt_path}\n"
        "Run src/train.py first."
    )

    model = build_model(cfg, num_classes=cfg["labels"]["num_classes"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    logger.info(f"Loaded checkpoint from {ckpt_path}")

    # ── NIH DataLoader ─────────────────────────────────────────────────────
    logger.info("\nLoading NIH data...")
    nih_loader = get_nih_loader(cfg, max_samples=5_000)

    # ── Run inference ──────────────────────────────────────────────────────
    all_labels, all_probs = run_inference(model, nih_loader, device)

    # ── Compute per-label AUROC + bootstrap CIs ────────────────────────────
    n_bootstrap = cfg["evaluation"]["bootstrap_n"]
    nih_results = {}

    logger.info(f"\nBootstrap AUROC ({n_bootstrap} reps, 95% CI):")
    for i, name in enumerate(LABEL_NAMES):
        y_true = all_labels[:, i]
        y_prob = all_probs[:, i]

        if y_true.sum() == 0:
            logger.warning(f"  {name}: no positives — skipping")
            nih_results[name] = {"auroc": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
            continue

        auroc, ci_lo, ci_hi = bootstrap_auroc(
            y_true, y_prob,
            n_bootstrap=n_bootstrap,
            seed=cfg["project"]["seed"],
        )
        nih_results[name] = {
            "auroc": round(auroc, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        }
        logger.info(f"  {name:<22}: {auroc:.4f}  [{ci_lo:.4f} – {ci_hi:.4f}]")

    # ── Load internal val AUROCs (from last epoch of training history) ─────
    history_path = Path(cfg["paths"]["metrics_dir"]) / "train_history.json"
    internal_val_aurocs = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        # Find the epoch with best mean AUROC
        best_epoch_metrics = max(
            history["val_metrics"],
            key=lambda m: m["mean_auroc"] if not np.isnan(m["mean_auroc"]) else -1,
        )
        for name, m in best_epoch_metrics["per_label"].items():
            internal_val_aurocs[name] = m["auroc"]
        logger.info("\nInternal val AUROCs (best epoch) loaded from training history.")
    else:
        logger.warning("train_history.json not found — internal val AUROCs unavailable.")

    # ── Print comparison table ─────────────────────────────────────────────
    print_results_table(nih_results, internal_val_aurocs)

    # ── Save metrics ───────────────────────────────────────────────────────
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "nih_results": nih_results,
        "internal_val_aurocs": internal_val_aurocs,
        "drop": {
            name: round(
                nih_results.get(name, {}).get("auroc", float("nan")) -
                internal_val_aurocs.get(name, float("nan")), 4
            )
            for name in LABEL_NAMES
        },
    }
    metrics_path = metrics_dir / "external_eval.json"
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2)
    
    backbone = cfg["model"]["backbone"]
    with open(metrics_dir / f"external_eval_{backbone}.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nMetrics saved → {metrics_path} and {metrics_dir / f'external_eval_{backbone}.json'}")

    # ── Plot ROC curves ────────────────────────────────────────────────────
    plot_roc_curves(
        all_labels, all_probs,
        save_dir=Path(cfg["paths"]["figures_dir"]),
    )
    import shutil
    fig_dir = Path(cfg["paths"]["figures_dir"])
    if (fig_dir / "nih_roc_curves.png").exists():
        shutil.copyfile(fig_dir / "nih_roc_curves.png", fig_dir / f"nih_roc_curves_{backbone}.png")

    return all_labels, all_probs, nih_results


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    evaluate_external(cfg)
