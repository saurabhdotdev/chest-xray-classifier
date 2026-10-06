"""
src/clinical_engine.py
======================
Clinical Diagnostic Intelligence, Automated Cardiothoracic Ratio (CTR) Calipers,
and Multi-Pathology Diagnostic Syndrome Synthesis.
"""

from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def estimate_ctr(
    img_pil: Image.Image,
    manual_c_offset: int = 0,
    manual_t_offset: int = 0,
) -> Dict[str, Any]:
    """
    Automated and Caliper-assisted Cardiothoracic Ratio (CTR) Estimation.

    Clinical Definition:
    --------------------
    CTR = Maximum Transverse Cardiac Diameter (C) / Maximum Internal Thoracic Diameter (T).
    Normal value on PA chest radiograph is < 0.50 (50%).
    Values > 0.50 indicate cardiomegaly; > 0.65 indicates severe enlargement.

    Parameters
    ----------
    img_pil : PIL.Image.Image
        Input chest radiograph.
    manual_c_offset : int
        User fine-tuning adjustment for cardiac width (pixels).
    manual_t_offset : int
        User fine-tuning adjustment for thoracic width (pixels).

    Returns
    -------
    dict with 'ctr', 'classification', 'cardiac_px', 'thorax_px', 'annotated_pil', 'color'
    """
    arr = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. Smooth image to suppress high-frequency noise/text
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Thoracic diameter measurement level (lower thorax, ~68% height)
    y_thorax = int(h * 0.68)
    left_t = max(10, int(w * 0.12)) + manual_t_offset
    right_t = min(w - 10, int(w * 0.88)) - manual_t_offset
    t_width = max(20, right_t - left_t)

    # 3. Cardiac diameter measurement level (~62% height)
    y_cardiac = int(h * 0.62)
    row_c = blur[y_cardiac, :]
    mid_x = w // 2

    # Horizontal gradient to detect left and right cardiac borders
    grad_c = np.gradient(row_c.astype(float))

    # Right atrium border (left side of image, moving left from midline)
    left_search = grad_c[left_t:mid_x]
    rt_offset = np.argmax(left_search) if len(left_search) > 0 else int(w * 0.16)
    right_atrium_x = left_t + rt_offset - manual_c_offset

    # Left ventricle apex border (right side of image, moving right from midline)
    right_search = grad_c[mid_x:right_t]
    apex_offset = np.argmin(right_search) if len(right_search) > 0 else int(w * 0.22)
    left_apex_x = mid_x + apex_offset + manual_c_offset

    right_atrium_x = max(left_t, min(mid_x - 5, right_atrium_x))
    left_apex_x = max(mid_x + 5, min(right_t, left_apex_x))

    c_width = max(10, left_apex_x - right_atrium_x)
    ctr = max(0.20, min(0.95, c_width / float(t_width)))

    # 4. Clinical Grading
    if ctr < 0.50:
        grade = "Normal Cardiac Silhouette (CTR < 0.50)"
        color = "#00ff88"
        severity = "Normal"
        clinical_note = "Cardiac silhouette width is within standard limits (<50% of inner thoracic diameter)."
    elif ctr <= 0.55:
        grade = "Mild Cardiomegaly (CTR 0.50 – 0.55)"
        color = "#ffb300"
        severity = "Mild"
        clinical_note = "Borderline to mild cardiac enlargement. Correlate with patient habitus and AP vs PA projection."
    elif ctr <= 0.65:
        grade = "Moderate Cardiomegaly (CTR 0.55 – 0.65)"
        color = "#ff7a85"
        severity = "Moderate"
        clinical_note = "Significant cardiomegaly. Suggests left ventricular dilation, hypertrophy, or pericardial effusion."
    else:
        grade = "Marked / Severe Cardiomegaly (CTR > 0.65)"
        color = "#ff4757"
        severity = "Severe"
        clinical_note = "Marked global enlargement ('water-bottle' heart). Evaluate urgently for congestive failure or large pericardial effusion."

    # 5. Draw Digital Radiologist Calipers
    overlay = arr.copy()
    
    # Cyan line for Thoracic Diameter (T)
    cv2.line(overlay, (left_t, y_thorax), (right_t, y_thorax), (0, 212, 255), 2)
    cv2.line(overlay, (left_t, y_thorax - 6), (left_t, y_thorax + 6), (0, 212, 255), 2)
    cv2.line(overlay, (right_t, y_thorax - 6), (right_t, y_thorax + 6), (0, 212, 255), 2)

    # Magenta/Coral line for Cardiac Diameter (C)
    cv2.line(overlay, (right_atrium_x, y_cardiac), (left_apex_x, y_cardiac), (255, 71, 87), 2)
    cv2.line(overlay, (right_atrium_x, y_cardiac - 6), (right_atrium_x, y_cardiac + 6), (255, 71, 87), 2)
    cv2.line(overlay, (left_apex_x, y_cardiac - 6), (left_apex_x, y_cardiac + 6), (255, 71, 87), 2)

    # Midline dashed guide
    for y_dash in range(int(h * 0.3), int(h * 0.85), 8):
        cv2.line(overlay, (mid_x, y_dash), (mid_x, y_dash + 4), (180, 180, 180), 1)

    annotated_pil = Image.fromarray(overlay)

    return {
        "ctr": ctr,
        "classification": grade,
        "severity": severity,
        "color": color,
        "cardiac_px": c_width,
        "thorax_px": t_width,
        "left_t": left_t,
        "right_t": right_t,
        "right_atrium_x": right_atrium_x,
        "left_apex_x": left_apex_x,
        "y_cardiac": y_cardiac,
        "y_thorax": y_thorax,
        "clinical_note": clinical_note,
        "annotated_pil": annotated_pil,
    }


def diagnose_syndrome(
    probs: List[float],
    label_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Synthesize multi-label findings into overarching clinical syndromes
    with RSNA diagnostic management recommendations and ICD-10 suggestions.

    Findings: [Cardiomegaly, Pleural Effusion, Atelectasis, Edema]
    """
    f_map = dict(zip(label_names, probs))
    c_pos = f_map.get("Cardiomegaly", 0.0) >= threshold
    e_pos = f_map.get("Pleural Effusion", 0.0) >= threshold
    a_pos = f_map.get("Atelectasis", 0.0) >= threshold
    ed_pos = f_map.get("Edema", 0.0) >= threshold

    c_prob = f_map.get("Cardiomegaly", 0.0)
    e_prob = f_map.get("Pleural Effusion", 0.0)
    a_prob = f_map.get("Atelectasis", 0.0)
    ed_prob = f_map.get("Edema", 0.0)

    # 1. Congestive Heart Failure Triad: Cardiomegaly + Edema + Pleural Effusion
    if c_pos and ed_pos and e_pos:
        return {
            "syndrome": "Acute Decompensated Heart Failure (ADHF) Triad",
            "icd10": "I50.9 (Heart failure, unspecified) · J81.0 (Acute pulmonary edema)",
            "acuity": "CRITICAL EMERGENCY",
            "acuity_tier": 3,
            "color": "#ff4757",
            "icon": "🚨",
            "summary": (
                "Classic radiographic triad of cardiogenic decompensation: marked cardiomegaly, "
                "interstitial/alveolar pulmonary edema, and bilateral pleural fluid accumulation."
            ),
            "differentials": [
                "Congestive Heart Failure Exacerbation (Left ventricular systolic/diastolic failure)",
                "Acute Coronary Syndrome with flash pulmonary edema",
                "Severe Fluid Overload secondary to end-stage renal disease",
            ],
            "recommendations": [
                "Urgent cardiology consultation & stat bedside echocardiogram (TTE).",
                "Prompt initiation of IV loop diuretics (e.g. Furosemide) and preload/afterload optimization.",
                "Continuous pulse oximetry, supplemental O2, and monitor for non-invasive positive pressure ventilation (BiPAP/CPAP).",
                "Check serum NT-proBNP/BNP, high-sensitivity troponins, electrolytes, and arterial blood gas.",
            ],
        }

    # 2. Cardiogenic Pulmonary Congestion: Cardiomegaly + Edema
    if c_pos and ed_pos:
        return {
            "syndrome": "Cardiogenic Pulmonary Congestion",
            "icd10": "I50.1 (Left ventricular failure) · J81.0 (Pulmonary edema)",
            "acuity": "URGENT ACTIONABLE",
            "acuity_tier": 2,
            "color": "#ff7a85",
            "icon": "🌊",
            "summary": (
                "Combined cardiac silhouette enlargement and diffuse perihilar alveolar/vascular congestion, "
                "characteristic of elevated pulmonary capillary wedge pressure."
            ),
            "differentials": [
                "Decompensating congestive heart failure",
                "Hypertensive emergency with pulmonary vascular congestion",
                "Acute mitral regurgitation or ischemic cardiomyopathy",
            ],
            "recommendations": [
                "Titrate diuretic therapy; restrict sodium and fluid intake.",
                "Echocardiogram to assess left ventricular ejection fraction and regional wall motion abnormalities.",
                "Follow-up chest radiograph in 24–48 hours post-diuresis to monitor clearance.",
            ],
        }

    # 3. Basilar Volume Loss with Fluid: Atelectasis + Pleural Effusion
    if a_pos and e_pos:
        return {
            "syndrome": "Dependent Basilar Lung Collapse with Pleural Effusion",
            "icd10": "J90 (Pleural effusion) · J98.11 (Atelectasis)",
            "acuity": "URGENT ACTIONABLE",
            "acuity_tier": 2,
            "color": "#ffb300",
            "icon": "🫁",
            "summary": (
                "Concurrent pleural fluid accumulation and adjacent compressive/discoid atelectatic volume loss, "
                "frequent in post-operative, recumbent, or hypoventilating patients."
            ),
            "differentials": [
                "Compressive atelectasis secondary to moderate/large pleural effusion",
                "Post-thoracotomy / post-laparotomy dependent hypoventilation",
                "Parapneumonic effusion with subsegmental consolidation",
            ],
            "recommendations": [
                "Consider diagnostic/therapeutic thoracentesis if effusion is hemodynamically or respiratory significant.",
                "Point-of-care thoracic ultrasound (POCUS) to evaluate fluid depth and guide safe drainage.",
                "Aggressive pulmonary toilet, incentive spirometry, and patient mobilization.",
            ],
        }

    # 4. Isolated Pulmonary Edema
    if ed_pos:
        return {
            "syndrome": "Acute Pulmonary Edema / Alveolar Infiltration",
            "icd10": "J81.0 (Acute pulmonary edema) · J18.9 (Pneumonia, unspecified)",
            "acuity": "URGENT ACTIONABLE",
            "acuity_tier": 2,
            "color": "#ff7a85",
            "icon": "🌊",
            "summary": (
                "Prominent perihilar bat-wing opacities and alveolar haziness without significant cardiomegaly. "
                "May represent non-cardiogenic pulmonary edema (ARDS) or acute early congestive edema."
            ),
            "differentials": [
                "Non-cardiogenic pulmonary edema / ARDS (permeability edema)",
                "Acute heart failure (early, prior to remodeling)",
                "Bilateral infectious pneumonia / pneumonitis",
            ],
            "recommendations": [
                "Differentiate cardiogenic vs non-cardiogenic via echocardiogram and clinical volume status.",
                "Assess PaO2/FiO2 ratio and inflammatory biomarkers (CRP, Procalcitonin).",
            ],
        }

    # 5. Isolated Cardiomegaly
    if c_pos:
        return {
            "syndrome": "Isolated Cardiomegaly / Ventricular Remodeling",
            "icd10": "I51.7 (Cardiomegaly) · I11.9 (Hypertensive heart disease)",
            "acuity": "MODERATE MONITORING",
            "acuity_tier": 2,
            "color": "#ffb300",
            "icon": "🫀",
            "summary": (
                "Cardiothoracic ratio elevated (> 0.50) without overt alveolar edema or significant effusion. "
                "Indicates chronic remodeling, cardiomyopathy, or pericardial fluid."
            ),
            "differentials": [
                "Hypertensive heart disease with left ventricular hypertrophy (LVH)",
                "Dilated or ischemic cardiomyopathy",
                "Pericardial effusion without tamponade",
            ],
            "recommendations": [
                "Transthoracic echocardiogram (TTE) to measure chambers, wall thickness, and ejection fraction.",
                "Optimization of guideline-directed medical therapy (GDMT: ACEi/ARB/ARNI, beta-blocker, SGLT2i).",
            ],
        }

    # 6. Isolated Pleural Effusion
    if e_pos:
        return {
            "syndrome": "Isolated Pleural Effusion / Fluid Accumulation",
            "icd10": "J90 (Pleural effusion, not elsewhere classified)",
            "acuity": "MODERATE MONITORING",
            "acuity_tier": 2,
            "color": "#00d4ff",
            "icon": "💧",
            "summary": (
                "Blunting of costophrenic angles with lateral meniscus formation, indicating free fluid in pleural space."
            ),
            "differentials": [
                "Transudative effusion (heart failure, cirrhosis, nephrosis)",
                "Exudative parapneumonic effusion",
                "Malignancy or pulmonary embolism",
            ],
            "recommendations": [
                "Thoracic ultrasound to assess volume, loculation, and mark for thoracentesis.",
                "Light's criteria analysis (fluid protein/LDH) if diagnostic tap is performed.",
            ],
        }

    # 7. Isolated Atelectasis
    if a_pos:
        return {
            "syndrome": "Subsegmental / Linear Discoid Atelectasis",
            "icd10": "J98.11 (Atelectasis)",
            "acuity": "ROUTINE CLINICAL CARE",
            "acuity_tier": 1,
            "color": "#a8c8e0",
            "icon": "🫁",
            "summary": (
                "Linear horizontal or lobar volume loss opacities, frequently reflecting shallow breathing or secretions."
            ),
            "differentials": [
                "Plate-like hypoventilation atelectasis",
                "Mucous plugging of small bronchus",
                "Early resolving focal consolidation",
            ],
            "recommendations": [
                "Incentive spirometry, chest physiotherapy, and encourage deep breathing exercises.",
                "Repeat radiograph if hypoxemia persists.",
            ],
        }

    # 8. Unremarkable Study
    return {
        "syndrome": "Unremarkable Cardiopulmonary Examination",
        "icd10": "Z01.818 (Encounter for other preprocedural examination)",
        "acuity": "ROUTINE NORMAL",
        "acuity_tier": 1,
        "color": "#00ff88",
        "icon": "✅",
        "summary": (
            "All four target abnormalities are below clinical threshold. Clear lung fields, sharp costophrenic sulci, "
            "normal cardiac silhouette diameter, and absence of vascular congestion."
        ),
        "differentials": [
            "Normal frontal chest radiograph",
            "Non-radiopaque cardiopulmonary etiologies if clinically symptomatic",
        ],
        "recommendations": [
            "No acute radiographic intervention required.",
            "Clinical correlation with symptoms; consider cross-sectional CT if high clinical suspicion for non-radiopaque pathology.",
        ],
    }
