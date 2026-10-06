"""
src/image_utils.py
==================
Radiology imaging utilities, DICOM/web image acquisition,
and PACS window/level preprocessing for chest radiographs.
"""

import io
from typing import Dict, Any, Tuple
import requests
from PIL import Image, ImageOps
import numpy as np
import cv2


SAMPLE_WEB_XRAYS = {
    "🌐 NIH Public Radiograph - Normal": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Chest_Xray_PA_3-8-2010.png/640px-Chest_Xray_PA_3-8-2010.png",
    "🌐 Severe Bilateral Pneumonia / Infiltrates": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Pneumonia_chest_x-ray.jpg/640px-Pneumonia_chest_x-ray.jpg",
    "🌐 Marked Cardiomegaly & Heart Failure": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Cardiomegaly.jpg/640px-Cardiomegaly.jpg",
}


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load an image from raw file bytes and convert to RGB."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # respect smartphone/camera orientation
    return img.convert("RGB")


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Fetch an image from a public URL with timeout and header masking."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type and not any(url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
        # Still attempt to parse in case Content-Type is application/octet-stream
        pass

    return load_image_from_bytes(response.content)


def apply_windowing(
    img: Image.Image,
    brightness: int = 0,
    contrast: float = 1.0,
    invert: bool = False,
    use_clahe: bool = False,
    sharpen: bool = False,
) -> Image.Image:
    """
    Apply standard radiologist PACS window/level and contrast enhancements.

    Parameters
    ----------
    img : PIL.Image (RGB)
    brightness : int in [-100, 100], shifts intensity baseline (Window Level)
    contrast : float in [0.2, 3.0], scales intensity spread (Window Width)
    invert : bool, photonegative mode (bone black vs bone white)
    use_clahe : bool, Contrast-Limited Adaptive Histogram Equalization
    sharpen : bool, edge enhancement for lung markings and borders
    """
    arr = np.array(img.convert("RGB"))

    # 1. Grayscale Invert (bones black / background white vs standard)
    if invert:
        arr = 255 - arr

    # 2. CLAHE (Adaptive Histogram Equalization on luminance channel)
    if use_clahe:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 3. Brightness & Contrast (Window / Level)
    if contrast != 1.0 or brightness != 0:
        arr_f = contrast * arr.astype(np.float32) + float(brightness)
        arr = np.clip(arr_f, 0, 255).astype(np.uint8)

    # 4. Sharpening filter (Unsharp mask approximation)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        arr = cv2.filter2D(arr, -1, kernel)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def compute_image_stats(image: Image.Image) -> Dict[str, Any]:
    """Compute physical and radiometric properties of the radiograph."""
    w, h = image.size
    arr = np.array(image.convert("L"))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = int(np.min(arr))
    max_val = int(np.max(arr))
    dynamic_range = max_val - min_val

    # Estimate aspect ratio description
    aspect = w / h
    if 0.85 <= aspect <= 1.15:
        aspect_desc = "Square / Standard PA (1:1)"
    elif aspect < 0.85:
        aspect_desc = "Portrait / Tall Thoracic"
    else:
        aspect_desc = "Landscape / Wide Thoracic"

    return {
        "width": w,
        "height": h,
        "aspect_ratio": f"{aspect:.2f}:1 ({aspect_desc})",
        "mean_intensity": f"{mean_val:.1f} / 255",
        "std_intensity": f"{std_val:.1f}",
        "dynamic_range": f"{dynamic_range} levels ({min_val}–{max_val})",
        "format": getattr(image, "format", "RGB"),
    }
