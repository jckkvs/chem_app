from django.test import SimpleTestCase
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from core.services.ml.pipeline import MLPipeline
from core.services.ml.tracking import MLTracker

class MLPipelineTests(SimpleTestCase):
    def test_train_regression(self):
        # Synthetic Data
        X = pd.DataFrame(np.random.rand(20, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.rand(20))
        
        config = {
            'model_type': 'random_forest',
            'task_type': 'regression',
            'cv_folds': 2
        }
        
        # Mock Tracker to avoid MLFlow warnings or disk write
        tracker = MagicMock(spec=MLTracker)
        tracker.start_run.return_value.__enter__.return_value = None
        
        pipeline = MLPipeline(config, tracker=tracker)
        metrics = pipeline.train(X, y)
        
        self.assertIn('cv_mean_score', metrics)
        self.assertIn('train_r2', metrics)
        self.assertIsNotNone(pipeline.model)
        
        # Predict
        preds = pipeline.predict(X)
        self.assertEqual(len(preds), 20)
