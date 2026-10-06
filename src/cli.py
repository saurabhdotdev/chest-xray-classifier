"""
src/cli.py
==========
Command-Line Interface (CLI) for Headless Chest X-ray Classification,
Batch Screening, and Automated Clinical Report Generation.

Usage Examples:
---------------
  # Single image formatted text prediction
  python -m src.cli --image data/samples/sample_cardiomegaly.jpg

  # Single image JSON output
  python -m src.cli --image data/samples/sample_pulmonary_edema.jpg --format json

  # Generate official PDF radiology report
  python -m src.cli --image data/samples/sample_cardiomegaly.jpg --output-report results/sample_report.pdf

  # Batch directory screening to CSV
  python -m src.cli --batch data/samples --output results/batch_results.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.train import build_model
from src.gradcam import GradCAM, denormalize, overlay_heatmap
from src.image_utils import load_image_from_bytes
from src.clinical_engine import estimate_ctr, diagnose_syndrome
from src.report_generator import generate_pdf_report, generate_structured_report_text, compute_triage_tier

LABEL_NAMES = ["Cardiomegaly", "Pleural Effusion", "Atelectasis", "Edema"]
LABEL_ICONS = ["🫀", "💧", "🫁", "🌊"]


def load_model_weights(device: torch.device):
    ckpt_path = ROOT / "results" / "checkpoints" / "best_model.pth"
    cal_ckpt_path = ROOT / "results" / "checkpoints" / "best_model_calibrated.pth"

    cfg_path = ROOT / "config.yaml"
    backbone = "densenet121"
    if cfg_path.exists():
        with open(cfg_path) as f:
            backbone = yaml.safe_load(f).get("model", {}).get("backbone", "densenet121")

    fast_cfg = {"model": {"backbone": backbone, "pretrained": False}}
    model = build_model(fast_cfg, num_classes=4)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval().to(device)

    fitted_temp = 0.05
    if cal_ckpt_path.exists():
        try:
            cal_data = torch.load(str(cal_ckpt_path), map_location=device)
            if isinstance(cal_data, dict) and "temperature" in cal_data:
                fitted_temp = float(cal_data["temperature"])
        except Exception:
            pass

    return model, fitted_temp


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def analyze_single_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    tf: transforms.Compose,
    temperature: float,
    threshold: float = 0.5,
    use_calibration: bool = True,
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    img_t = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        raw_probs = torch.sigmoid(logits).cpu().numpy()[0]
        cal_probs = torch.sigmoid(logits / float(temperature)).cpu().numpy()[0]

    active_probs = cal_probs if use_calibration else raw_probs

    # CTR measurement
    ctr_res = estimate_ctr(img)

    # Syndrome
    syndrome = diagnose_syndrome(active_probs.tolist(), LABEL_NAMES, threshold)
    triage = compute_triage_tier(active_probs.tolist(), LABEL_NAMES, threshold)

    findings = {}
    for name, p, raw_p in zip(LABEL_NAMES, active_probs, raw_probs):
        findings[name] = {
            "probability": float(p),
            "raw_probability": float(raw_p),
            "status": "POSITIVE" if p >= threshold else "NEGATIVE",
        }

    return {
        "file": str(image_path),
        "timestamp": datetime.now().isoformat(),
        "triage_tier": triage["tier"],
        "triage_badge": triage["badge"],
        "syndrome": syndrome["syndrome"],
        "acuity": syndrome["acuity"],
        "icd10": syndrome["icd10"],
        "ctr": float(ctr_res["ctr"]),
        "ctr_grade": ctr_res["classification"],
        "findings": findings,
        "differentials": syndrome["differentials"],
        "recommendations": syndrome["recommendations"],
        "img_pil": img,
        "img_t": img_t,
        "active_probs": active_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="ChestAI CLI — Command-Line Radiology Intelligence")
    parser.add_argument("--image", type=str, help="Path to frontal chest radiograph (JPG/PNG)")
    parser.add_argument("--batch", type=str, help="Directory containing radiographs for batch screening")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--output", type=str, help="Path to save batch results CSV")
    parser.add_argument("--output-report", type=str, help="Path to save official RSNA PDF report")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification sensitivity threshold")
    parser.add_argument("--raw", action="store_true", help="Use raw uncalibrated probabilities instead of temperature scaling")
    parser.add_argument("--patient-id", type=str, default="CLI-PT-001", help="Patient MRN / ID for report")

    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.print_help()
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fitted_temp = load_model_weights(device)
    tf = get_transforms()
    use_cal = not args.raw

    # Single Image Mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: File not found at {args.image}", file=sys.stderr)
            sys.exit(1)

        res = analyze_single_image(
            img_path, model, device, tf, fitted_temp, threshold=args.threshold, use_calibration=use_cal
        )

        if args.format == "json":
            out_dict = {k: v for k, v in res.items() if k not in ("img_pil", "img_t", "active_probs")}
            print(json.dumps(out_dict, indent=2))
        else:
            print("\n" + "=" * 64)
            print("🫁 ChestAI Radiology Intelligence — Automated Case Analysis")
            print("=" * 64)
            print(f"📁 Image: {img_path.name}")
            print(f"🚨 Triage: Tier {res['triage_tier']} — {res['triage_badge']}")
            print(f"🏥 Clinical Syndrome: {res['syndrome']} ({res['acuity']})")
            print(f"📖 ICD-10: {res['icd10']}")
            print(f"📏 Cardiothoracic Ratio (CTR): {res['ctr']:.1%} — {res['ctr_grade']}")
            print("-" * 64)
            print("Pathology Findings:")
            for icon, name in zip(LABEL_ICONS, LABEL_NAMES):
                f = res["findings"][name]
                status_box = "[+ POS]" if f["status"] == "POSITIVE" else "[- NEG]"
                bar = "█" * int(f["probability"] * 20) + "░" * (20 - int(f["probability"] * 20))
                print(f"  {icon} {name:<18} {f['probability']:>6.1%}  |{bar}|  {status_box}")
            print("-" * 64)
            print("Top Differential Diagnoses:")
            for d in res["differentials"]:
                print(f"  • {d}")
            print("Clinical Recommendations:")
            for r in res["recommendations"]:
                print(f"  • {r}")
            print("=" * 64 + "\n")

        # Generate PDF Report if requested
        if args.output_report:
            cfg_path = ROOT / "config.yaml"
            bb = "densenet121"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    bb = yaml.safe_load(f).get("model", {}).get("backbone", "densenet121")
            cam_gen = GradCAM(model, backbone=bb)
            primary_idx = int(np.argmax(res["active_probs"]))
            hm = cam_gen(res["img_t"], primary_idx, device)
            img_np = denormalize(res["img_t"].squeeze(0))
            overlay = overlay_heatmap(img_np, hm, alpha=0.45)

            patient_info = {
                "patient_id": args.patient_id,
                "age": 65,
                "sex": "Unspecified",
                "view": "PA (Posteroanterior)",
                "indication": "CLI automated headless diagnostic evaluation.",
            }

            pdf_bytes = generate_pdf_report(
                patient_info,
                res["active_probs"].tolist(),
                args.threshold,
                use_cal,
                LABEL_NAMES,
                res["img_pil"],
                Image.fromarray(overlay),
            )
            out_p = Path(args.output_report)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wb") as f:
                f.write(pdf_bytes)
            print(f"✓ Official RSNA Radiology Report saved to: {out_p}")

    # Batch Directory Mode
    elif args.batch:
        batch_dir = Path(args.batch)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [p for p in batch_dir.iterdir() if p.suffix.lower() in valid_exts]

        if not files:
            print(f"No valid image files found in {batch_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"⚡ Processing batch of {len(files)} radiographs in {batch_dir}...")
        rows = []
        for p in files:
            res = analyze_single_image(
                p, model, device, tf, fitted_temp, threshold=args.threshold, use_calibration=use_cal
            )
            row = {
                "Filename": p.name,
                "Triage_Tier": res["triage_tier"],
                "Acuity": res["acuity"],
                "Syndrome": res["syndrome"],
                "CTR_Pct": f"{res['ctr']:.1%}",
            }
            for name in LABEL_NAMES:
                row[f"{name}_Prob"] = f"{res['findings'][name]['probability']:.3f}"
                row[f"{name}_Status"] = res["findings"][name]["status"]
            rows.append(row)

        # Sort: Tier 3 (Critical) first
        rows.sort(key=lambda r: r["Triage_Tier"], reverse=True)

        if args.output:
            import pandas as pd
            df = pd.DataFrame(rows)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"✓ Batch results saved to: {out_path}")
        else:
            print("\nBatch Triage Summary:")
            for r in rows:
                print(f"  [Tier {r['Triage_Tier']}] {r['Filename']:<28} | {r['Syndrome']:<36} | CTR: {r['CTR_Pct']}")


if __name__ == "__main__":
    main()
