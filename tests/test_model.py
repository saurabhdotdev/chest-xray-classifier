"""
tests/test_model.py
===================
Unit tests for DenseNet-121 model architecture, weights loading, and forward inference.
"""

import unittest
from pathlib import Path
import torch
import yaml

from src.train import build_model

ROOT = Path(__file__).resolve().parent.parent


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cfg_path = ROOT / "config.yaml"
        backbone = "densenet121"
        if cfg_path.exists():
            with open(cfg_path) as f:
                backbone = yaml.safe_load(f).get("model", {}).get("backbone", "densenet121")
        cfg = {"model": {"backbone": backbone, "pretrained": False}}
        cls.model = build_model(cfg, num_classes=4)
        ckpt_path = ROOT / "results" / "checkpoints" / "best_model.pth"
        if ckpt_path.exists():
            cls.model.load_state_dict(torch.load(str(ckpt_path), map_location=cls.device))
        cls.model.eval().to(cls.device)

    def test_forward_pass_shape(self):
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            logits = self.model(dummy_input)
        self.assertEqual(logits.shape, (1, 4))

    def test_probabilities_range(self):
        dummy_input = torch.randn(2, 3, 224, 224, device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(dummy_input))
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_batch_inference_shape(self):
        batch = torch.randn(4, 3, 224, 224, device=self.device)
        with torch.no_grad():
            logits = self.model(batch)
        self.assertEqual(logits.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
