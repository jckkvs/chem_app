"""
PreprocessingWrapperのテスト
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class PreprocessingWrapperTests(unittest.TestCase):
    """PreprocessingWrapperのテスト"""
    
    def setUp(self):
        # テストデータ
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        self.y = pd.Series(y)
    
    def test_standard_scaler(self):
        """StandardScalerのテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        prep = PreprocessingWrapper(method='standard')
        X_scaled = prep.fit_transform(self.X)
        
        # 平均≈0、標準偏差≈1
        self.assertTrue(np.allclose(X_scaled.mean(), 0, atol=1e-10))
        self.assertTrue(np.allclose(X_scaled.std(), 1, atol=0.1))
    
    def test_minmax_scaler(self):
        """MinMaxScalerのテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        prep = PreprocessingWrapper(method='minmax')
        X_scaled = prep.fit_transform(self.X)
        
        # 最小値≈0、最大値≈1
        self.assertTrue(np.allclose(X_scaled.min(), 0, atol=1e-10))
        self.assertTrue(np.allclose(X_scaled.max(), 1, atol=1e-10))
    
    def test_normalizer(self):
        """Normalizerのテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        prep = PreprocessingWrapper(method='normalizer', norm='l2')
        X_normalized = prep.fit_transform(self.X)
        
        # 各行のL2ノルム≈1
        row_norms = np.sqrt((X_normalized ** 2).sum(axis=1))
        self.assertTrue(np.allclose(row_norms, 1.0))
    
    def test_polynomial_features(self):
        """PolynomialFeaturesのテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        prep = PreprocessingWrapper(method='polynomial', degree=2, include_bias=False)
        X_poly = prep.fit_transform(self.X)
        
        # 特徴数が増加
        self.assertGreater(X_poly.shape[1], self.X.shape[1])
    
    def test_binarizer(self):
        """Binarizerのテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        prep = PreprocessingWrapper(method='binarizer', threshold=0.0)
        X_binary = prep.fit_transform(self.X)
        
        # 0または1のみ
        unique_vals = np.unique(X_binary.values)
        self.assertTrue(set(unique_vals).issubset({0.0, 1.0}))
    
    def test_power_transformer(self):
        """PowerTransformerのテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        X_positive = self.X + np.abs(self.X.min().min()) + 1  # 正の値に
        
        prep = PreprocessingWrapper(method='power', power_method='yeo-johnson')
        X_transformed = prep.fit_transform(X_positive)
        
        self.assertEqual(X_transformed.shape, X_positive.shape)
    
    def test_inverse_transform(self):
        """逆変換のテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        prep = PreprocessingWrapper(method='standard')
        X_scaled = prep.fit_transform(self.X)
        X_original = prep.inverse_transform(X_scaled)
        
        # 元に戻る
        self.assertTrue(np.allclose(X_original.values, self.X.values))
    
    def test_get_available_methods(self):
        """利用可能手法一覧のテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import PreprocessingWrapper
        
        methods = PreprocessingWrapper.get_available_methods()
        
        self.assertEqual(len(methods), 10)
        self.assertIn('standard', methods)
        self.assertIn('polynomial', methods)
    
    def test_auto_scale_helper(self):
        """auto_scaleヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules.preprocessing_extended import auto_scale
        
        X_scaled = auto_scale(self.X, method='auto')
        
        self.assertEqual(X_scaled.shape, self.X.shape)
