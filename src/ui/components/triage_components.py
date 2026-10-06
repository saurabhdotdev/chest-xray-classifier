"""
src/ui/components/triage_components.py
======================================
Batch decompression and triage worklist serialization utilities.
"""

from datetime import datetime
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any
import zipfile
from PIL import Image


def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, Image.Image]]:
    """Extracts valid radiograph images from an uploaded ZIP archive."""
    cases = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for fname in sorted(z.namelist()):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")) and not fname.startswith("__MACOSX"):
                    try:
                        data = z.read(fname)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        clean_name = Path(fname).name
                        cases.append((clean_name, img))
                    except Exception:
                        pass
    except Exception:
        pass
    return cases


def build_triage_csv(results: List[Dict[str, Any]]) -> str:
    """Serializes triage acuity ranked results into standard clinical CSV format."""
    lines = ["Priority_Rank,Patient_ID,Triage_Tier,Urgency_Badge,Primary_Finding,Cardiomegaly_Pct,Effusion_Pct,Atelectasis_Pct,Edema_Pct,Timestamp"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, r in enumerate(results, 1):
        probs = r["probs"]
        lines.append(
            f"{idx},{r['patient'].replace(',', ' ')},Tier {r['tier']},{r['badge'].replace(',', ' ')},"
            f"{r['primary_finding'].replace(',', ' ')},{probs[0]:.1%},{probs[1]:.1%},{probs[2]:.1%},{probs[3]:.1%},{now_str}"
        )
    return "\n".join(lines)
