"""
FeatureSelectorのテスト

Implements: TEST-FEATURE-SELECT-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureSelectorFilterMethodsTests(unittest.TestCase):
    """Filter Methodsのテスト"""
    
    def setUp(self):
        # 回帰データ
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=20, n_informative=10, random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(20)])
        self.y_reg = pd.Series(y_reg)
        
        # 分類データ
        X_clf, y_clf = make_classification(
            n_samples=100, n_features=20, n_informative=10, random_state=42
        )
        self.X_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(20)])
        self.y_clf = pd.Series(y_clf)
    
    def test_variance_threshold(self):
        """Variance Thresholdのテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        selector = FeatureSelector(method='variance', threshold=0.8)
        X_selected = selector.fit_transform(self.X_reg, self.y_reg)
        
        # threshold=0.8で分散が低い特徴が削除されることを確認
        self.assertLessEqual(X_selected.shape[1], self.X_reg.shape[1])
        self.assertEqual(len(selector.get_selected_features()), X_selected.shape[1])
    
    def test_k_best_regression(self):
        """SelectKBest（回帰）のテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        selector = FeatureSelector(method='k_best', task_type='regression', k=10)
        X_selected = selector.fit_transform(self.X_reg, self.y_reg)
        
        self.assertEqual(X_selected.shape[1], 10)
        self.assertEqual(len(selector.get_selected_features()), 10)
    
    def test_k_best_classification(self):
        """SelectKBest（分類）のテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        selector = FeatureSelector(method='k_best', task_type='classification', k=10)
        X_selected = selector.fit_transform(self.X_clf, self.y_clf)
        
        self.assertEqual(X_selected.shape[1], 10)
    
    def test_percentile(self):
        """SelectPercentileのテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        selector = FeatureSelector(method='percentile', percentile=50)
        X_selected = selector.fit_transform(self.X_reg, self.y_reg)
        
        # 50%の特徴を選択
        self.assertEqual(X_selected.shape[1], 10)


class FeatureSelectorWrapperMethodsTests(unittest.TestCase):
    """Wrapper Methodsのテスト"""
    
    def setUp(self):
        # 小さいデータセット（Wrapper Methodsは計算コストが高いため）
        X_reg, y_reg = make_regression(
            n_samples=50, n_features=10, n_informative=5, random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
        self.y_reg = pd.Series(y_reg)
        
        X_clf, y_clf = make_classification(
            n_samples=50, n_features=10, n_informative=5, random_state=42
        )
        self.X_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
    
    def test_rfe(self):
        """RFE（Recursive Feature Elimination）のテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        selector = FeatureSelector(
            method='rfe',
            estimator=estimator,
            n_features_to_select=5,
        )
        X_selected = selector.fit_transform(self.X_reg, self.y_reg)
        
        self.assertEqual(X_selected.shape[1], 5)
    
    def test_rfecv(self):
        """RFECV（RFE with Cross-Validation）のテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        selector = FeatureSelector(
            method='rfecv',
            estimator=estimator,
            cv=3,
        )
        X_selected = selector.fit_transform(self.X_reg, self.y_reg)
        
        # 最適な特徴数が自動選択される
        self.assertGreater(X_selected.shape[1], 0)
        self.assertLessEqual(X_selected.shape[1], self.X_reg.shape[1])
    
    def test_sequential_forward(self):
        """Sequential Feature Selector（Forward）のテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = FeatureSelector(
            method='sequential',
            estimator=estimator,
            n_features_to_select=5,
            direction='forward',
            cv=3,
        )
        X_selected = selector.fit_transform(self.X_clf, self.y_clf)
        
        self.assertEqual(X_selected.shape[1], 5)


class FeatureSelectorEmbeddedMethodsTests(unittest.TestCase):
    """Embedded Methodsのテスト"""
    
    def setUp(self):
        X_reg, y_reg = make_regression(
            n_samples=100, n_features=20, n_informative=10, random_state=42
        )
        self.X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(20)])
        self.y_reg = pd.Series(y_reg)
    
    def test_select_from_model(self):
        """SelectFromModel（L1/Tree-based）のテスト"""
        from core.services.ml.feature_selector import FeatureSelector
        
        # Random Forestでfitしてから使用
        estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        estimator.fit(self.X_reg, self.y_reg)
        
        selector = FeatureSelector(
            method='from_model',
            estimator=estimator,
            prefit=True,
            threshold='median',
        )
        X_selected = selector.fit_transform(self.X_reg, self.y_reg)
        
        # 中央値以上の重要度を持つ特徴が選択される
        self.assertLess(X_selected.shape[1], self.X_reg.shape[1])


class FeatureSelectorHelperTests(unittest.TestCase):
    """ヘルパー関数のテスト"""
    
    def test_auto_select_features(self):
        """auto_select_features関数のテスト"""
        from core.services.ml.feature_selector import auto_select_features
        
        X, y = make_regression(n_samples=100, n_features=50, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(50)])
        y_series = pd.Series(y)
        
        X_selected, selected_features = auto_select_features(
            X_df, y_series, target_n_features=20
        )
        
        self.assertEqual(X_selected.shape[1], 20)
        self.assertEqual(len(selected_features), 20)
        self.assertIsInstance(X_selected, pd.DataFrame)
        self.assertIsInstance(selected_features, list)
