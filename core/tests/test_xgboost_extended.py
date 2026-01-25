"""
XGBoostWrapperのテスト
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# XGBoost可用性チェック
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
class XGBoostWrapperTests(unittest.TestCase):
    """XGBoostWrapperのテスト"""
    
    def setUp(self):
        # 分類データ
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
        
        # 回帰データ
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=[f'f{i}' for i in range(10)])
        self.y_reg = pd.Series(y_reg)
    
    def test_xgb_classifier(self):
        """XGBClassifierのテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        xgb_wrapper = XGBoostWrapper(
            model_type='classifier',
            n_estimators=10,
            max_depth=3
        )
        xgb_wrapper.fit(self.X_clf, self.y_clf)
        predictions = xgb_wrapper.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
    
    def test_xgb_regressor(self):
        """XGBRegressorのテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        xgb_wrapper = XGBoostWrapper(
            model_type='regressor',
            n_estimators=10
        )
        xgb_wrapper.fit(self.X_reg, self.y_reg)
        predictions = xgb_wrapper.predict(self.X_reg)
        
        self.assertEqual(len(predictions), len(self.y_reg))
    
    def test_xgbrf_classifier(self):
        """XGBRFClassifierのテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        xgb_wrapper = XGBoostWrapper(
            model_type='rf_classifier',
            n_estimators=10
        )
        xgb_wrapper.fit(self.X_clf, self.y_clf)
        predictions = xgb_wrapper.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
    
    def test_xgbrf_regressor(self):
        """XGBRFRegressorのテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        xgb_wrapper = XGBoostWrapper(
            model_type='rf_regressor',
            n_estimators=10
        )
        xgb_wrapper.fit(self.X_reg, self.y_reg)
        predictions = xgb_wrapper.predict(self.X_reg)
        
        self.assertEqual(len(predictions), len(self.y_reg))
    
    def test_predict_proba(self):
        """確率予測のテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        xgb_wrapper = XGBoostWrapper(model_type='classifier', n_estimators=10)
        xgb_wrapper.fit(self.X_clf, self.y_clf)
        probas = xgb_wrapper.predict_proba(self.X_clf)
        
        # 確率の合計は1
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))
    
    def test_feature_importance(self):
        """特徴重要度のテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        xgb_wrapper = XGBoostWrapper(model_type='classifier', n_estimators=10)
        xgb_wrapper.fit(self.X_clf, self.y_clf)
        importance = xgb_wrapper.get_feature_importance()
        
        self.assertIsInstance(importance, pd.Series)
        self.assertGreater(len(importance), 0)
    
    def test_custom_params(self):
        """カスタムパラメータのテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import XGBoostWrapper
        
        # 任意のXGBoostパラメータを渡せる
        xgb_wrapper = XGBoostWrapper(
            model_type='regressor',
            n_estimators=5,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=3
        )
        xgb_wrapper.fit(self.X_reg, self.y_reg)
        
        self.assertIsNotNone(xgb_wrapper.model_)
    
    def test_create_xgboost_model_helper(self):
        """create_xgboost_modelヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules.xgboost_extended import create_xgboost_model
        
        model = create_xgboost_model(
            task_type='classification',
            n_estimators=10
        )
        model.fit(self.X_clf, self.y_clf)
        predictions = model.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
