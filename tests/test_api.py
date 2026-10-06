"""
tests/test_api.py
=================
Integration tests for FastAPI inference endpoints.
"""

import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("documentation", data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["num_classes"], 4)

    def test_predict_endpoint(self):
        # Create a test synthetic PNG in memory
        img = Image.new("RGB", (224, 224), color=(70, 70, 70))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("test_xray.png", buf, "image/png")},
            params={"threshold": 0.5, "calibrated": True},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("findings", data)
        self.assertIn("triage", data)
        self.assertIn("syndrome", data)
        self.assertIn("cardiothoracic_ratio", data)


if __name__ == "__main__":
    unittest.main()
