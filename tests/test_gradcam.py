"""
tests/test_gradcam.py
=====================
Unit tests for Grad-CAM activation mapping and overlay blending.
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.train import build_model
from src.gradcam import GradCAM, denormalize, overlay_heatmap

ROOT = Path(__file__).resolve().parent.parent


class TestGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cfg_path = ROOT / "config.yaml"
        backbone = "densenet121"
        if cfg_path.exists():
            import yaml
            with open(cfg_path) as f:
                backbone = yaml.safe_load(f).get("model", {}).get("backbone", "densenet121")
        cfg = {"model": {"backbone": backbone, "pretrained": False}}
        cls.model = build_model(cfg, num_classes=4)
        cls.model.eval().to(cls.device)
        cls.cam = GradCAM(cls.model, backbone=backbone)

    def test_heatmap_shape_and_range(self):
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        for class_idx in range(4):
            heatmap = self.cam(dummy_input, class_idx, self.device)
            self.assertEqual(len(heatmap.shape), 2)
            self.assertTrue((heatmap >= 0.0).all())
            self.assertTrue((heatmap <= 1.0).all())

    def test_overlay_heatmap_output(self):
        img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap = np.ones((100, 100), dtype=np.float32) * 0.5
        overlay = overlay_heatmap(img_rgb, heatmap, alpha=0.4)
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
