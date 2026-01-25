import json
import os
import shutil

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase

from core.models import Dataset, Experiment


class ChemMLApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.token = "test-token"
        os.environ["API_SECRET_TOKEN"] = self.token
        # cleanup uploads
        if os.path.exists("uploads_test"):
            shutil.rmtree("uploads_test")

    def tearDown(self):
        if "API_SECRET_TOKEN" in os.environ:
            del os.environ["API_SECRET_TOKEN"]

    def test_upload_dataset(self):
        content = b"SMILES,target\nC,1.0\nCC,2.0\n"
        file = SimpleUploadedFile("test.csv", content, content_type="text/csv")
        payload = {
            "name": "Test Dataset", 
            "smiles_col": "SMILES", 
            "target_col": "target",
            "file": file
        }
        
        # Note: /api/ is the prefix we set in urls.py for NinjaAPI
        response = self.client.post(
            "/api/datasets", 
            data=payload,
            HTTP_AUTHORIZATION=f"Bearer {self.token}"
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Test Dataset")
        self.assertTrue(Dataset.objects.filter(name="Test Dataset").exists())

    def test_create_experiment(self):
        # Create dataset first
        ds = Dataset.objects.create(name="ExpDS", file_path="dummy.csv")
        
        payload = {
            "dataset_id": ds.id,
            "name": "Test Exp",
            "features": ["rdkit"],
            "model_type": "lgbm"
        }
        
        response = self.client.post(
            "/api/experiments", 
            data=json.dumps(payload), 
            content_type="application/json",
            HTTP_AUTHORIZATION=f"Bearer {self.token}"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Test Exp")
        self.assertEqual(data["status"], "PENDING")
