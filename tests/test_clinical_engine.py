"""
tests/test_clinical_engine.py
=============================
Unit tests for Cardiothoracic Ratio (CTR) and clinical diagnostic syndrome synthesis.
"""

import unittest
from PIL import Image
from src.clinical_engine import estimate_ctr, diagnose_syndrome

LABEL_NAMES = ["Cardiomegaly", "Pleural Effusion", "Atelectasis", "Edema"]


class TestClinicalEngine(unittest.TestCase):
    def test_ctr_estimation_bounds(self):
        # Create a test synthetic image
        img = Image.new("RGB", (224, 224), color=(80, 80, 80))
        res = estimate_ctr(img)
        self.assertIn("ctr", res)
        self.assertIn("classification", res)
        self.assertGreaterEqual(res["ctr"], 0.20)
        self.assertLessEqual(res["ctr"], 0.95)
        self.assertIsInstance(res["annotated_pil"], Image.Image)

    def test_adhf_triad_syndrome(self):
        probs = [0.90, 0.85, 0.10, 0.92]
        res = diagnose_syndrome(probs, LABEL_NAMES, threshold=0.5)
        self.assertIn("Decompensated Heart Failure", res["syndrome"])
        self.assertEqual(res["acuity_tier"], 3)
        self.assertGreater(len(res["differentials"]), 0)
        self.assertGreater(len(res["recommendations"]), 0)

    def test_normal_syndrome(self):
        probs = [0.05, 0.02, 0.08, 0.04]
        res = diagnose_syndrome(probs, LABEL_NAMES, threshold=0.5)
        self.assertEqual(res["acuity_tier"], 1)
        self.assertIn("Unremarkable", res["syndrome"])

    def test_isolated_cardiomegaly(self):
        probs = [0.88, 0.10, 0.05, 0.12]
        res = diagnose_syndrome(probs, LABEL_NAMES, threshold=0.5)
        self.assertIn("Cardiomegaly", res["syndrome"])
        self.assertEqual(res["acuity_tier"], 2)


if __name__ == "__main__":
    unittest.main()
