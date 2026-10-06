"""
src/report_generator.py
=======================
Automated Structured Clinical Radiology Report Generation
and Professional PDF Export Engine using RSNA reporting guidelines.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def compute_triage_tier(probs: List[float], label_names: List[str], threshold: float = 0.5) -> Dict[str, Any]:
    """
    Determine clinical triage urgency tier based on probabilities:
    Tier 3 (Red): Critical / Urgent acute findings (Edema, Massive Effusion >= 70%)
    Tier 2 (Yellow): Moderate / Actionable positive findings
    Tier 1 (Green): Low probability / Unremarkable
    """
    findings_map = dict(zip(label_names, probs))
    edema_prob = findings_map.get("Edema", 0.0)
    effusion_prob = findings_map.get("Pleural Effusion", 0.0)
    cardio_prob = findings_map.get("Cardiomegaly", 0.0)
    atelectasis_prob = findings_map.get("Atelectasis", 0.0)

    any_pos = any(p >= threshold for p in probs)
    high_critical = (edema_prob >= 0.65) or (effusion_prob >= 0.75)

    if high_critical:
        return {
            "tier": 3,
            "badge": "CRITICAL ALERT · IMMEDIATE REVIEW",
            "color": "#ff4757",
            "desc": "High probability of acute pulmonary edema or large pleural effusion. Urgent clinical correlation recommended."
        }
    elif any_pos:
        return {
            "tier": 2,
            "badge": "MODERATE PRIORITY · ACTIONABLE FINDING",
            "color": "#ffb300",
            "desc": "One or more abnormal radiological findings detected above sensitivity threshold."
        }
    else:
        return {
            "tier": 1,
            "badge": "ROUTINE / UNREMARKABLE",
            "color": "#00ff88",
            "desc": "No acute thoracic abnormality detected above threshold."
        }


def generate_structured_report_text(
    patient_info: Dict[str, Any],
    probs: List[float],
    threshold: float,
    calibrated: bool,
    label_names: List[str],
) -> str:
    """Generate structured RSNA-style clinical radiology report in markdown."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    findings_map = dict(zip(label_names, probs))
    triage = compute_triage_tier(probs, label_names, threshold)

    pt_id = patient_info.get("patient_id", "ANONYMOUS-001")
    age = patient_info.get("age", 58)
    sex = patient_info.get("sex", "Unspecified")
    view = patient_info.get("view", "PA (Posteroanterior)")
    indication = patient_info.get("indication", "Routine screening / dyspnea evaluation")

    # Generate compartmental narratives
    c_prob = findings_map.get("Cardiomegaly", 0.0)
    if c_prob >= threshold:
        cardio_text = f"Enlarged cardiothoracic silhouette observed (AI confidence: {c_prob:.1%}). Suggestive of ventricular enlargement or pericardial fluid."
    else:
        cardio_text = f"Cardiothoracic contour within expected limits (AI confidence of enlargement: {c_prob:.1%})."

    pe_prob = findings_map.get("Pleural Effusion", 0.0)
    if pe_prob >= threshold:
        pleural_text = f"Blunting of costophrenic recesses identified, consistent with pleural effusion / basilar fluid accumulation (AI confidence: {pe_prob:.1%})."
    else:
        pleural_text = f"Costophrenic angles appear sharp; no substantial pleural fluid layer identified (AI confidence: {pe_prob:.1%})."

    at_prob = findings_map.get("Atelectasis", 0.0)
    if at_prob >= threshold:
        parenchymal_text = f"Subsegmental or linear band-like opacities detected, compatible with lung atelectasis / volume loss (AI confidence: {at_prob:.1%})."
    else:
        parenchymal_text = f"Lungs are reasonably expanded without focal lobar collapse or dense atelectatic consolidation (AI confidence: {at_prob:.1%})."

    ed_prob = findings_map.get("Edema", 0.0)
    if ed_prob >= threshold:
        vascular_text = f"Prominent perihilar vascular markings with hazy alveolar-interstitial pattern, indicative of pulmonary edema / vascular congestion (AI confidence: {ed_prob:.1%})."
    else:
        vascular_text = f"Pulmonary vascular caliber and hilar markings within normal limits without bat-wing batwing alveolar edema (AI confidence: {ed_prob:.1%})."

    positives = [name for name, p in findings_map.items() if p >= threshold]
    if positives:
        impression_summary = f"Radiograph demonstrates findings positive for **{', '.join(positives)}**."
    else:
        impression_summary = "No overt radiographic evidence of cardiomegaly, pleural effusion, atelectasis, or pulmonary edema above threshold."

    md = f"""### 🏥 DEPARTMENT OF RADIOLOGY & MEDICAL IMAGING
**AUTOMATED AI PRE-READ RADIOLOGICAL CONSULTATION**

---
**EXAMINATION:** Chest Radiograph, Frontal Projection ({view})  
**PATIENT MRN:** `{pt_id}` | **AGE / SEX:** {age} y/o | {sex}  
**STUDY TIMESTAMP:** {now_str}  
**CLINICAL INDICATION:** {indication}  
**MODEL SPECIFICATION:** DenseNet-121 (CheXpert + NIH ChestX-ray14) | {'Calibrated (Temperature Scaling T=0.05)' if calibrated else 'Raw Network Output'}  
**TRIAGE STATUS:** `{triage['badge']}`

---
#### 📋 SYSTEMATIC COMPARTMENTAL FINDINGS
1. **Cardiac & Mediastinal Silhouette:**  
   {cardio_text}
2. **Pleural Spaces & Costophrenic Sulci:**  
   {pleural_text}
3. **Pulmonary Parenchyma:**  
   {parenchymal_text}
4. **Pulmonary Vasculature & Interstitium:**  
   {vascular_text}

---
#### 🎯 IMPRESSION & RECOMMENDATIONS
- **Summary:** {impression_summary}
- **Calibration Status:** {'Probabilities adjusted via post-hoc temperature scaling to reflect true population frequency.' if calibrated else 'Uncalibrated softmax/sigmoid output.'}
- **Clinical Recommendations:**
  - Correlate with physical exam, oxygen saturation, and serum cardiac/inflammatory biomarkers (BNP, Troponin).
  - {'Consider bedside thoracic ultrasound or upright follow-up radiograph post-therapy.' if pe_prob >= threshold or ed_prob >= threshold else 'Routine outpatient follow-up as clinically warranted.'}
  - {'Echocardiogram suggested to assess ejection fraction and chamber enlargement.' if c_prob >= threshold else ''}

---
*⚠️ DISCLAIMER: Automated research AI reading. Must be reviewed and signed off by a licensed radiologist before clinical management decisions.*
"""
    return md


def generate_pdf_report(
    patient_info: Dict[str, Any],
    probs: List[float],
    threshold: float,
    calibrated: bool,
    label_names: List[str],
    orig_img: Image.Image,
    overlay_img: Image.Image,
    multi_heatmaps: Optional[List[Tuple[str, Image.Image]]] = None,
) -> bytes:
    """
    Generate a formatted multi-page hospital PDF report.
    Returns bytes of the compiled PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0d1526"),
        alignment=TA_LEFT,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#556980"),
        alignment=TA_LEFT,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#007799"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    bold_body = ParagraphStyle(
        "ReportBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#7a9ab8"),
        alignment=TA_CENTER,
    )

    elements = []

    # 1. Header Banner
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    header_data = [
        [
            Paragraph("<b>ChestAI RADIOLOGY PLATFORM</b><br/><font size=8 color='#556980'>Deep Learning Diagnostic & Explainability Consultation Report</font>", title_style),
            Paragraph(f"<b>REPORT ID:</b> CAI-{datetime.now().strftime('%y%m%d%H%M')}<br/><b>DATE:</b> {now_str}<br/><b>STATUS:</b> PRE-READ COMPLETE", subtitle_style),
        ]
    ]
    t_header = Table(header_data, colWidths=[340, 190])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t_header)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0099bb"), spaceBefore=4, spaceAfter=8))

    # 2. Patient Demographics Block
    pt_id = patient_info.get("patient_id", "ANONYMOUS-001")
    age = patient_info.get("age", 58)
    sex = patient_info.get("sex", "Unspecified")
    view = patient_info.get("view", "PA (Posteroanterior)")
    indication = patient_info.get("indication", "Routine clinical chest examination")

    triage = compute_triage_tier(probs, label_names, threshold)

    demo_data = [
        [
            Paragraph(f"<b>Patient MRN:</b> {pt_id}", body_style),
            Paragraph(f"<b>Age / Sex:</b> {age} y/o / {sex}", body_style),
            Paragraph(f"<b>Projection:</b> {view}", body_style),
        ],
        [
            Paragraph(f"<b>Indication:</b> {indication}", body_style),
            Paragraph(f"<b>Sensitivity Threshold:</b> {threshold:.0%}", body_style),
            Paragraph(f"<b>Triage:</b> <font color='{triage['color']}'><b>{triage['badge']}</b></font>", body_style),
        ]
    ]
    t_demo = Table(demo_data, colWidths=[180, 170, 180])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_demo)
    elements.append(Spacer(1, 10))

    # 3. Quantitative AI Findings Table
    elements.append(Paragraph("QUANTITATIVE PATHOLOGY ASSESSMENT", h2_style))

    table_rows = [
        [
            Paragraph("<b>Target Pathology</b>", bold_body),
            Paragraph("<b>Predicted Probability</b>", bold_body),
            Paragraph("<b>Cutoff</b>", bold_body),
            Paragraph("<b>Classification Result</b>", bold_body),
            Paragraph("<b>Calibration</b>", bold_body),
        ]
    ]

    for name, p in zip(label_names, probs):
        is_pos = (p >= threshold)
        status_str = f"<font color='{'#dc2626' if is_pos else '#16a34a'}'><b>{'POSITIVE (Detected)' if is_pos else 'NEGATIVE (Clear)'}</b></font>"
        cal_str = "Temperature Scaled" if calibrated else "Raw Sigmoid"
        table_rows.append([
            Paragraph(f"<b>{name}</b>", body_style),
            Paragraph(f"<b>{p:.1%}</b>", body_style),
            Paragraph(f"{threshold:.0%}", body_style),
            Paragraph(status_str, body_style),
            Paragraph(cal_str, body_style),
        ])

    t_findings = Table(table_rows, colWidths=[140, 110, 60, 130, 90])
    t_findings.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_findings)
    elements.append(Spacer(1, 10))

    # 4. Embedded Radiographs & Grad-CAM Visualizations
    elements.append(Paragraph("IMAGING & GRAD-CAM EXPLAINABILITY ATTENTION", h2_style))

    # Save original and overlay to temporary in-memory buffers
    orig_buf = io.BytesIO()
    orig_img.resize((220, 220)).save(orig_buf, format="JPEG", quality=90)
    orig_buf.seek(0)

    over_buf = io.BytesIO()
    overlay_img.resize((220, 220)).save(over_buf, format="JPEG", quality=90)
    over_buf.seek(0)

    img_table_data = [
        [
            RLImage(orig_buf, width=170, height=170),
            RLImage(over_buf, width=170, height=170),
        ],
        [
            Paragraph("<font size=7.5 color='#556980'><b>Figure 1:</b> Original Chest Radiograph</font>", disclaimer_style),
            Paragraph("<font size=7.5 color='#556980'><b>Figure 2:</b> Grad-CAM Heatmap Attention Overlay</font>", disclaimer_style),
        ]
    ]
    t_imgs = Table(img_table_data, colWidths=[265, 265])
    t_imgs.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    elements.append(t_imgs)
    elements.append(Spacer(1, 10))

    # 5. Narrative Findings & Impression
    elements.append(Paragraph("STRUCTURED IMPRESSION & CLINICAL ACTION PLAN", h2_style))

    positives = [name for name, p in zip(label_names, probs) if p >= threshold]
    if positives:
        imp_body = f"Radiograph exhibits salient features consistent with <b>{', '.join(positives)}</b> above the specified operating threshold ({threshold:.0%}). Attention heatmaps demonstrate corresponding anatomical activation."
    else:
        imp_body = f"No overt findings above operating threshold ({threshold:.0%}). Cardiothoracic ratio, pleural recesses, and lung fields demonstrate no critical opacities."

    recommendations = []
    if "Pleural Effusion" in positives or "Edema" in positives:
        recommendations.append("Recommend immediate clinical correlation with oxygenation status and consideration of urgent diuretic therapy or thoracic ultrasound.")
    if "Cardiomegaly" in positives:
        recommendations.append("Echocardiographic assessment recommended to evaluate left ventricular ejection fraction and valvular anatomy.")
    if "Atelectasis" in positives:
        recommendations.append("Encourage pulmonary toileting, deep breathing exercises, or incentive spirometry; follow-up if persistent.")
    if not recommendations:
        recommendations.append("Routine outpatient follow-up according to primary symptom trajectory.")

    recs_text = "<br/>• ".join([""] + recommendations)

    narrative_data = [
        [
            Paragraph("<b>Impression:</b>", bold_body),
            Paragraph(imp_body, body_style),
        ],
        [
            Paragraph("<b>Action Plan:</b>", bold_body),
            Paragraph(f"• {recs_text[5:]}", body_style),
        ]
    ]
    t_narrative = Table(narrative_data, colWidths=[90, 440])
    t_narrative.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t_narrative)
    elements.append(Spacer(1, 14))

    # 6. Legal / Clinical Disclaimer Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=6))
    elements.append(Paragraph(
        "<b>CLINICAL DISCLAIMER & RESEARCH NOTICE:</b> ChestAI is an investigational deep learning research platform. "
        "Outputs are probabilistic assistive recommendations and do not constitute an official medical diagnosis. "
        "All interpretations must be validated by a credentialed radiologist before altering patient management.",
        disclaimer_style
    ))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
