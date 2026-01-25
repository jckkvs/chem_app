"""
ImputeWrapperのテスト
"""

import unittest

import numpy as np
import pandas as pd


class ImputeWrapperTests(unittest.TestCase):
    """ImputeWrapperのテスト"""
    
    def setUp(self):
        # 欠損値を含むデータ
        np.random.seed(42)
        X = np.random.randn(100, 5)
        # ランダムに欠損値を作成
        mask = np.random.rand(100, 5) < 0.1
        X[mask] = np.nan
        
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
    
    def test_simple_imputer_mean(self):
        """SimpleImputer（平均値）のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        imputer = ImputeWrapper(method='simple', strategy='mean')
        X_imputed = imputer.fit_transform(self.X)
        
        # 欠損値が補完される
        self.assertFalse(X_imputed.isnull().any().any())
    
    def test_simple_imputer_median(self):
        """SimpleImputer（中央値）のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        imputer = ImputeWrapper(method='simple', strategy='median')
        X_imputed = imputer.fit_transform(self.X)
        
        self.assertFalse(X_imputed.isnull().any().any())
    
    def test_knn_imputer(self):
        """KNNImputerのテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        imputer = ImputeWrapper(method='knn', n_neighbors=3)
        X_imputed = imputer.fit_transform(self.X)
        
        self.assertFalse(X_imputed.isnull().any().any())
    
    def test_iterative_imputer(self):
        """IterativeImputer（MICE）のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        imputer = ImputeWrapper(method='iterative', max_iter=5, random_state=42)
        X_imputed = imputer.fit_transform(self.X)
        
        self.assertFalse(X_imputed.isnull().any().any())
    
    def test_auto_select(self):
        """自動選択のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        imputer = ImputeWrapper(method='auto')
        X_imputed = imputer.fit_transform(self.X)
        
        self.assertFalse(X_imputed.isnull().any().any())
        self.assertIsNotNone(imputer.selected_method_)
    
    def test_missing_indicator(self):
        """MissingIndicatorのテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        imputer = ImputeWrapper(method='missing_indicator')
        X_indicator = imputer.fit_transform(self.X)
        
        # 欠損値指標（0 or 1）
        unique_vals = np.unique(X_indicator.values)
        self.assertTrue(set(unique_vals).issubset({0.0, 1.0, True, False}))
    
    def test_get_available_methods(self):
        """利用可能手法一覧のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import ImputeWrapper
        
        methods = ImputeWrapper.get_available_methods()
        
        self.assertEqual(len(methods), 4)
        self.assertIn('simple', methods)
        self.assertIn('iterative', methods)
    
    def test_auto_impute_helper(self):
        """auto_imputeヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import auto_impute
        
        X_imputed = auto_impute(self.X, method='simple', strategy='mean')
        
        self.assertFalse(X_imputed.isnull().any().any())
    
    def test_get_missing_summary(self):
        """get_missing_summaryヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules.impute_extended import get_missing_summary
        
        summary = get_missing_summary(self.X)
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('missing_count', summary.columns)
        self.assertIn('missing_rate', summary.columns)
