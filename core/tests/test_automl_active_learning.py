"""
Tests for AutoML and Active Learning Engines

Implements: T-AUTOML-001, T-AL-001
Targets:
 - core/services/ml/automl.py
 - core/services/ml/active_learning.py
"""

import math
import unittest
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.services.ml.automl import AutoMLEngine
from core.services.ml.active_learning import ActiveLearner

# ==============================================================================
# AutoML Tests
# ==============================================================================

# ==============================================================================
# AutoML Tests
# ==============================================================================

class TestAutoMLEngine(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.rand(20, 5), columns=[f'f{i}' for i in range(5)])
        self.y = pd.Series(np.random.rand(20))
        
    def test_optimize_lightgbm_regression(self):
        """Test optimization loop for LightGBM/Regression"""
        # Mock optuna module and availability BEFORE creating engine
        mock_optuna = MagicMock()
        mock_study = MagicMock()
        mock_optuna.create_study.return_value = mock_study
        
        # Setup study mock
        mock_study.best_params = {'n_estimators': 100, 'learning_rate': 0.1}
        mock_study.best_value = 0.85
        
        with patch.dict('sys.modules', {'optuna': mock_optuna, 'optuna.samplers': MagicMock()}):
            with patch('core.services.ml.automl.OPTUNA_AVAILABLE', True):
                # Now we can import/reload or just run if classes are lazy
                # But since OPTUNA_AVAILABLE is checked at init, this patch works
                
                with patch('core.services.ml.automl.cross_val_score') as mock_cv:
                    mock_cv.return_value = np.array([0.8, 0.9, 0.85]) 
                    
                    # Instantiate
                    automl = AutoMLEngine(model_type='lightgbm', task_type='regression', n_trials=1)
                    
                    # Ensure optuna was grabbed from sys.modules
                    # Re-inject our mock_optuna into the instance if needed, 
                    # but since we patched sys.modules, automl.py global import might need attention if it was already imported as None
                    # Actually, automl.py does `import optuna` inside try/except. 
                    # If it failed initially, `optuna` name might not be bound.
                    # We need to ensure AutoMLEngine can access it.
                    
                    # In `automl.py`:
                    # try: import optuna ... except: OPTUNA_AVAILABLE = False
                    
                    # We need to Patch `optuna` in `core.services.ml.automl` namespace if it exists, or global
                    with patch('core.services.ml.automl.optuna', mock_optuna, create=True), \
                         patch('core.services.ml.automl.TPESampler', MagicMock(), create=True):
                         best_params, best_score = automl.optimize(self.X, self.y)
        
        # Assertions
        assert mock_study.optimize.called
        assert best_score == 0.85

    def test_suggest_params_structure(self):
        """Test parameter suggestion structure"""
        # We need mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_int.return_value = 100
        mock_trial.suggest_float.return_value = 0.1
        mock_trial.suggest_categorical.return_value = 'sqrt'
        
        with patch('core.services.ml.automl.OPTUNA_AVAILABLE', True):
             # LightGBM
             automl = AutoMLEngine(model_type='lightgbm')
             params = automl._suggest_params(mock_trial)
             assert 'n_estimators' in params
             
             # Random Forest
             automl = AutoMLEngine(model_type='random_forest')
             params = automl._suggest_params(mock_trial)
             assert 'max_features' in params


# ==============================================================================
# Active Learning Tests
# ==============================================================================

class TestActiveLearner(unittest.TestCase):
    def setUp(self):
        self.X_pool = pd.DataFrame(np.random.rand(20, 5))
        self.n_samples = 5
        
    def test_uncertainty_sampling_no_model(self):
        """Should return random samples if no model provided"""
        al = ActiveLearner(strategy='uncertainty')
        indices = al.suggest_next(self.X_pool, n_samples=5)
        
        assert len(indices) == 5
        assert len(set(indices)) == 5 # unique
        
    def test_uncertainty_sampling_with_model(self):
        """Should select samples with highest uncertainty"""
        mock_uq_model = MagicMock()
        # Mock uncertainty: increasing values (0..19)
        # Higher index = Higher uncertainty
        mock_uq_model.get_uncertainty.return_value = np.arange(20)
        
        al = ActiveLearner(uncertainty_model=mock_uq_model, strategy='uncertainty')
        indices = al.suggest_next(self.X_pool, n_samples=5)
        
        # Should pick indices 19, 18, 17, 16, 15
        assert set(indices) == {19, 18, 17, 16, 15}
        
    def test_diversity_sampling(self):
        """Test diversity sampling (KMeans)"""
        # Use simple 1D data to make clustering predictable
        X_simple = pd.DataFrame({'f1': [1, 1.1, 10, 10.1, 20, 20.1]})
        
        al = ActiveLearner(strategy='diversity')
        # Asking for 3 samples -> should pick roughly one from each cluster (1, 10, 20 range)
        indices = al.suggest_next(X_simple, n_samples=3)
        
        assert len(indices) == 3
        # We can't guarantee exact indices due to random init, but logic should run no error
        
    def test_hybrid_sampling(self):
        """Test hybrid strategy"""
        mock_uq_model = MagicMock()
        mock_uq_model.get_uncertainty.return_value = np.random.rand(20)
        
        al = ActiveLearner(uncertainty_model=mock_uq_model, strategy='hybrid')
        indices = al.suggest_next(self.X_pool, n_samples=4)
        
        assert len(indices) == 4
        # Should be mix of uncertainty and diversity selected
        # (Hard to assert exact mix without inspecting internals, but ensuring it runs is key)

    def test_expected_improvement(self):
        """Test EI calculation"""
        mock_uq_model = MagicMock()
        # mean=10, lower=8, upper=12 -> std approx 1.0
        mock_uq_model.predict_with_interval.return_value = (
            np.array([10.0]), 
            np.array([8.0]), 
            np.array([12.0])
        )
        
        al = ActiveLearner(uncertainty_model=mock_uq_model)
        
        # Best so far = 9.0. Mean is 10.0. improvement likely.
        ei = al.expected_improvement(pd.DataFrame([1]), y_best=9.0)
        
        assert len(ei) == 1
        assert ei[0] > 0
