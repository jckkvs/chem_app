"""
mlxtend Ensemblesのテスト

Implements: TEST-MLXTEND-ENSEMBLE-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


# mlxtendの可用性チェック
try:
    import mlxtend
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


@unittest.skipUnless(MLXTEND_AVAILABLE, "mlxtend not installed")
class MLXtendStackingTests(unittest.TestCase):
    """Stackingのテスト"""
    
    def setUp(self):
        # 分類データ
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
        
        # 回帰データ
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=[f'f{i}' for i in range(10)])
        self.y_reg = pd.Series(y_reg)
    
    def test_stacking_classification(self):
        """Stacking分類のテスト"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from core.services.ml.mlxtend_ensembles import MLXtendEnsembles
        
        base_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=500),
        ]
        meta_model = LogisticRegression(random_state=42, max_iter=500)
        
        ensemble = MLXtendEnsembles(
            method='stacking',
            base_models=base_models,
            meta_model=meta_model,
            task_type='classification',
            use_probas=True,
        )
        
        ensemble.fit(self.X_clf, self.y_clf)
        predictions = ensemble.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
        self.assertEqual(predictions.shape[0], self.X_clf.shape[0])
    
    def test_stacking_regression(self):
        """Stacking回帰のテスト"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from core.services.ml.mlxtend_ensembles import MLXtendEnsembles
        
        base_models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            Ridge(random_state=42),
        ]
        meta_model = Ridge(random_state=42)
        
        ensemble = MLXtendEnsembles(
            method='stacking',
            base_models=base_models,
            meta_model=meta_model,
            task_type='regression',
        )
        
        ensemble.fit(self.X_reg, self.y_reg)
        predictions = ensemble.predict(self.X_reg)
        
        self.assertEqual(len(predictions), len(self.y_reg))
    
    def test_stacking_predict_proba(self):
        """Stacking確率予測のテスト"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from core.services.ml.mlxtend_ensembles import MLXtendEnsembles
        
        base_models = [RandomForestClassifier(n_estimators=10, random_state=42)]
        meta_model = LogisticRegression(random_state=42)
        
        ensemble = MLXtendEnsembles(
            method='stacking',
            base_models=base_models,
            meta_model=meta_model,
            task_type='classification',
        )
        
        ensemble.fit(self.X_clf, self.y_clf)
        probas = ensemble.predict_proba(self.X_clf)
        
        # 確率の合計は1
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))


@unittest.skipUnless(MLXTEND_AVAILABLE, "mlxtend not installed")
class MLXtendVotingTests(unittest.TestCase):
    """Votingのテスト"""
    
    def setUp(self):
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
    
    def test_voting_soft(self):
        """Soft Votingのテスト"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from core.services.ml.mlxtend_ensembles import MLXtendEnsembles
        
        base_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=500),
        ]
        
        ensemble = MLXtendEnsembles(
            method='voting',
            base_models=base_models,
            task_type='classification',
            voting='soft',
        )
        
        ensemble.fit(self.X_clf, self.y_clf)
        predictions = ensemble.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
    
    def test_voting_with_weights(self):
        """重み付きVotingのテスト"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from core.services.ml.mlxtend_ensembles import MLXtendEnsembles
        
        base_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=500),
        ]
        
        ensemble = MLXtendEnsembles(
            method='voting',
            base_models=base_models,
            task_type='classification',
            voting='soft',
            weights=[2.0, 1.0],  # RandomForestに2倍の重み
        )
        
        ensemble.fit(self.X_clf, self.y_clf)
        predictions = ensemble.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))


@unittest.skipUnless(MLXTEND_AVAILABLE, "mlxtend not installed")
class MLXtendHelperTests(unittest.TestCase):
    """ヘルパー関数のテスト"""
    
    def setUp(self):
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
    
    def test_create_stacking_ensemble(self):
        """Stacking簡易作成のテスト"""
        from sklearn.ensemble import RandomForestClassifier
        from core.services.ml.mlxtend_ensembles import create_stacking_ensemble
        
        base_models = [RandomForestClassifier(n_estimators=10, random_state=42)]
        
        ensemble = create_stacking_ensemble(
            base_models=base_models,
            task_type='classification',
        )
        
        self.assertIsNotNone(ensemble)
        self.assertEqual(ensemble.method, 'stacking')
   
    def test_auto_ensemble(self):
        """自動アンサンブルのテスト"""
        from core.services.ml.mlxtend_ensembles import auto_ensemble
        
        ensemble, info = auto_ensemble(
            self.X_clf,
            self.y_clf,
            task_type='classification',
            method='stacking',
            n_base_models=2,
        )
        
        self.assertIsNotNone(ensemble)
        self.assertIsNotNone(info)
        self.assertEqual(info['n_base_models'], 2)
        self.assertEqual(info['method'], 'stacking')
