from django.test import TestCase, Client
from core.models import Dataset, Experiment
from core.services.features.uma_eng import UMAFeatureExtractor
from core.services.ml.tracking import MLTracker
import pandas as pd
import numpy as np
import os
import tempfile
import mlflow
import shutil
from unittest.mock import patch, MagicMock

class Phase4Tests(TestCase):
    def setUp(self):
        self.client = Client()
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = Dataset.objects.create(
            name="TestDS",
            file_path=os.path.join(self.temp_dir, "data.csv"),
            smiles_col="SMILES",
            target_col="target"
        )
        # Create dummy csv
        df = pd.DataFrame({
            "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC"] * 10, # 50 samples
            "target": np.random.rand(50)
        })
        df.to_csv(self.dataset.file_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_uma_persistence(self):
        # 1. Fit and Save
        eng = UMAFeatureExtractor()
        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"] * 10
        y = np.random.rand(50)
        eng.fit(smiles, y)
        
        save_path = os.path.join(self.temp_dir, "uma.joblib")
        eng.save(save_path)
        
        self.assertTrue(os.path.exists(save_path))
        
        # 2. Load
        eng2 = UMAFeatureExtractor()
        eng2.load(save_path)
        
        # 3. Verify Transform
        res1 = eng.transform(smiles)
        res2 = eng2.transform(smiles)
        pd.testing.assert_frame_equal(res1, res2)
        
    @patch("core.services.ml.tracking.MLTracker") # Fixed patch target
    def test_pipeline_with_uma_artifact(self, mock_tracker_cls):
        # Verify that run_training_task saves the artifact
        from core.tasks import run_training_task
        
        exp = Experiment.objects.create(
            dataset=self.dataset,
            name="UMA Exp",
            status="PENDING",
            config={"features": ["uma"], "model_type": "lgbm"}
        )
        
        # Mock tracker instance
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        
        # Execute task synchronously
        # Huey tasks have a .call_local method to bypass the queue
        run_training_task.call_local(exp.id) 
        
        # Reload exp to check status updates (which happen in the task)
        exp.refresh_from_db()
        self.assertEqual(exp.status, "COMPLETED")
        
        # Check if log_artifact was called for uma_reducer
        calls = mock_tracker.log_artifact.call_args_list
        uma_call = any("uma_reducer.joblib" in str(c[0][0]) for c in calls)
        self.assertTrue(uma_call, "UMA reducer should be logged as artifact")

    @patch("mlflow.search_runs")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.artifacts.download_artifacts")
    @patch("core.services.ml.tracking.MLTracker.load_latest_model")
    def test_predict_endpoint_with_uma(self, mock_load_model, mock_download, mock_get_exp, mock_search):
        # Setup mocks
        exp = Experiment.objects.create(
            dataset=self.dataset,
            name="UMA Exp Predict",
            status="COMPLETED",
            config={"features": ["uma"], "model_type": "lgbm"}
        )
        
        # Mock Model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_load_model.return_value = mock_model
        
        # Mock MLflow artifact download (UMA)
        # We need a real UMA file to load
        eng = UMAFeatureExtractor()
        eng.fit(["C"]*20, [0]*20)
        uma_path = os.path.join(self.temp_dir, "uma_reducer.joblib")
        eng.save(uma_path)
        mock_download.return_value = uma_path
        
        # Mock MLflow Run search
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_get_exp.return_value = mock_exp
        
        mock_run = MagicMock()
        mock_run.run_id = "run123"
        mock_search.return_value.empty = False
        mock_search.return_value.iloc = [mock_run]
        
        # Call API
        payload = {"smiles": "C"}
        response = self.client.post(f"/api/experiments/{exp.id}/predict", 
                                   data=payload, content_type="application/json")
        
        self.assertEqual(response.status_code, 200, response.json())
        data = response.json()
        self.assertEqual(data['prediction'], 0.5)
