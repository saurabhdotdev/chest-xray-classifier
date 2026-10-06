"""
src/api.py
==========
FastAPI Medical Inference Microservice for Chest Radiograph Classification,
Grad-CAM Explainability, and Automated RSNA Report Generation.

Run locally:
------------
  uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
  Interactive Swagger Docs: http://localhost:8000/docs
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from torchvision import transforms

from src.train import build_model
from src.gradcam import GradCAM, denormalize, overlay_heatmap
from src.image_utils import load_image_from_bytes
from src.clinical_engine import estimate_ctr, diagnose_syndrome
from src.report_generator import generate_pdf_report, compute_triage_tier

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="ChestAI — Radiology Intelligence API",
    description="Multi-label chest X-ray disease detection, temperature scaling calibration, Grad-CAM attention maps, and RSNA clinical reporting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_NAMES = ["Cardiomegaly", "Pleural Effusion", "Atelectasis", "Edema"]

# Global model state
_model = None
_device = None
_temperature = 0.05
_cam_gen = None
_tf = None


def get_inference_resources():
    global _model, _device, _temperature, _cam_gen, _tf
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = ROOT / "results" / "checkpoints" / "best_model.pth"
        cal_ckpt_path = ROOT / "results" / "checkpoints" / "best_model_calibrated.pth"

        cfg_path = ROOT / "config.yaml"
        backbone = "densenet121"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
                backbone = cfg.get("model", {}).get("backbone", "densenet121")

        fast_cfg = {"model": {"backbone": backbone, "pretrained": False}}
        _model = build_model(fast_cfg, num_classes=4)
        if ckpt_path.exists():
            _model.load_state_dict(torch.load(str(ckpt_path), map_location=_device))
        _model.eval().to(_device)

        if cal_ckpt_path.exists():
            try:
                cal_data = torch.load(str(cal_ckpt_path), map_location=_device)
                if isinstance(cal_data, dict) and "temperature" in cal_data:
                    _temperature = float(cal_data["temperature"])
            except Exception:
                pass

        _cam_gen = GradCAM(_model, backbone=backbone)
        _tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _model, _device, _temperature, _cam_gen, _tf


@app.get("/")
def root():
    return {
        "service": "ChestAI Radiology Intelligence API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": ["/health", "/predict", "/explain", "/report"],
    }


@app.get("/health")
def health():
    model, device, temp, _, _ = get_inference_resources()
    cfg_path = ROOT / "config.yaml"
    backbone = "densenet121"
    if cfg_path.exists():
        with open(cfg_path) as f:
            backbone = yaml.safe_load(f).get("model", {}).get("backbone", "densenet121")
    ckpt_path = ROOT / "results" / "checkpoints" / "best_model.pth"
    return {
        "status": "healthy",
        "device": str(device),
        "backbone": backbone,
        "num_classes": len(LABEL_NAMES),
        "classes": LABEL_NAMES,
        "calibrated_temperature": temp,
        "checkpoint_exists": ckpt_path.exists(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.05, le=0.95),
    calibrated: bool = Query(True),
):
    """Predict probabilities for the 4 target thoracic abnormalities with triage & syndrome synthesis."""
    try:
        contents = await file.read()
        img = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    model, device, temp, _, tf = get_inference_resources()
    img_t = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        raw_probs = torch.sigmoid(logits).cpu().numpy()[0]
        cal_probs = torch.sigmoid(logits / float(temp)).cpu().numpy()[0]

    active_probs = cal_probs if calibrated else raw_probs

    triage = compute_triage_tier(active_probs.tolist(), LABEL_NAMES, threshold)
    syndrome = diagnose_syndrome(active_probs.tolist(), LABEL_NAMES, threshold)
    ctr_res = estimate_ctr(img)

    findings = {}
    for name, p, raw_p in zip(LABEL_NAMES, active_probs, raw_probs):
        findings[name] = {
            "probability": float(p),
            "raw_probability": float(raw_p),
            "positive": bool(p >= threshold),
        }

    return {
        "filename": file.filename,
        "calibration_applied": calibrated,
        "temperature_used": temp if calibrated else 1.0,
        "threshold": threshold,
        "triage": triage,
        "findings": findings,
        "syndrome": {
            "name": syndrome["syndrome"],
            "acuity": syndrome["acuity"],
            "icd10": syndrome["icd10"],
            "summary": syndrome["summary"],
            "differentials": syndrome["differentials"],
            "recommendations": syndrome["recommendations"],
        },
        "cardiothoracic_ratio": {
            "ctr": float(ctr_res["ctr"]),
            "classification": ctr_res["classification"],
            "severity": ctr_res["severity"],
            "clinical_note": ctr_res["clinical_note"],
        },
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    alpha: float = Query(0.45, ge=0.1, le=0.9),
):
    """Generate Grad-CAM heatmaps and overlaid visualizations for all 4 pathologies as base64 images."""
    try:
        contents = await file.read()
        img = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    model, device, _, cam_gen, tf = get_inference_resources()
    img_t = tf(img).unsqueeze(0).to(device)
    img_np = denormalize(img_t.squeeze(0))

    heatmaps_b64 = {}
    for i, name in enumerate(LABEL_NAMES):
        hm = cam_gen(img_t, i, device)
        ov = overlay_heatmap(img_np, hm, alpha=alpha)

        # Encode to PNG base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        encoded = base64.b64encode(buffer).decode("utf-8")
        heatmaps_b64[name] = {
            "overlay_base64_png": f"data:image/png;base64,{encoded}",
        }

    return {
        "filename": file.filename,
        "alpha": alpha,
        "gradcam_overlays": heatmaps_b64,
    }


@app.post("/report")
async def report(
    file: UploadFile = File(...),
    patient_id: str = Query("PT-REST-001"),
    age: int = Query(65),
    sex: str = Query("Unspecified"),
    projection: str = Query("PA (Posteroanterior)"),
    indication: str = Query("API clinical diagnostic evaluation request"),
    threshold: float = Query(0.5),
    calibrated: bool = Query(True),
):
    """Generate and download an official RSNA Structured Radiology Report as a PDF."""
    try:
        contents = await file.read()
        img = load_image_from_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    model, device, temp, cam_gen, tf = get_inference_resources()
    img_t = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        cal_probs = torch.sigmoid(logits / float(temp)).cpu().numpy()[0]
        raw_probs = torch.sigmoid(logits).cpu().numpy()[0]

    active_probs = cal_probs if calibrated else raw_probs

    primary_idx = int(np.argmax(active_probs))
    hm = cam_gen(img_t, primary_idx, device)
    img_np = denormalize(img_t.squeeze(0))
    overlay = overlay_heatmap(img_np, hm, alpha=0.45)

    patient_info = {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "view": projection,
        "indication": indication,
    }

    pdf_bytes = generate_pdf_report(
        patient_info,
        active_probs.tolist(),
        threshold,
        calibrated,
        LABEL_NAMES,
        img,
        Image.fromarray(overlay),
    )

    filename = f"Radiology_Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
