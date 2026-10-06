"""
tests/test_report.py
====================
Unit tests for structured radiology reports and PDF document compilation.
"""

import unittest
from PIL import Image
from src.report_generator import generate_pdf_report, generate_structured_report_text, compute_triage_tier

LABEL_NAMES = ["Cardiomegaly", "Pleural Effusion", "Atelectasis", "Edema"]


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.patient_info = {
            "patient_id": "TEST-MRN-999",
            "age": 62,
            "sex": "Female",
            "view": "PA (Posteroanterior)",
            "indication": "Shortness of breath and fever",
        }
        self.probs = [0.85, 0.40, 0.15, 0.75]
        self.img = Image.new("RGB", (224, 224), color=(60, 60, 60))
        self.overlay = Image.new("RGB", (224, 224), color=(100, 30, 30))

    def test_structured_text_report(self):
        txt = generate_structured_report_text(
            self.patient_info, self.probs, 0.5, True, LABEL_NAMES
        )
        self.assertIn("TEST-MRN-999", txt)
        self.assertIn("Cardiomegaly", txt)
        self.assertIn("Edema", txt)

    def test_pdf_generation_bytes(self):
        pdf_bytes = generate_pdf_report(
            self.patient_info,
            self.probs,
            0.5,
            True,
            LABEL_NAMES,
            self.img,
            self.overlay,
        )
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_bytes), 2000)

    def test_triage_tier_logic(self):
        triage = compute_triage_tier([0.1, 0.2, 0.1, 0.8], LABEL_NAMES, threshold=0.5)
        self.assertEqual(triage["tier"], 3)  # High edema is tier 3 critical


if __name__ == "__main__":
    unittest.main()
