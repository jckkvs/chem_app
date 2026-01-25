"""
DimensionalityReducerのテスト

Implements: TEST-DIM-REDUCTION-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


class DimensionalityReducerLinearMethodsTests(unittest.TestCase):
    """線形次元削減手法のテスト"""
    
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
        self.y = pd.Series(y)
    
    def test_pca(self):
        """PCAのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(method='pca', n_components=5)
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 5)
        self.assertIsNotNone(reducer.explained_variance_ratio_)
    
    def test_incremental_pca(self):
        """IncrementalPCAのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='incremental_pca',
            n_components=5,
            batch_size=20
        )
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 5)
    
    def test_kernel_pca(self):
        """KernelPCAのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='kernel_pca',
            n_components=5,
            kernel='rbf'
        )
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 5)
    
    def test_truncated_svd(self):
        """TruncatedSVDのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(method='truncated_svd', n_components=5)
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 5)


class DimensionalityReducerManifoldMethodsTests(unittest.TestCase):
    """多様体学習手法のテスト"""
    
    def setUp(self):
        # 小規模データ（Manifold学習は計算コストが高い）
        X, y = make_classification(
            n_samples=50, n_features=10, random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        self.y = pd.Series(y)
    
    def test_tsne(self):
        """TSNEのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='tsne',
            n_components=2,
            perplexity=10,
            random_state=42
        )
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 2)
    
    def test_isomap(self):
        """Isomapのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='isomap',
            n_components=2,
            n_neighbors=5
        )
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 2)
    
    def test_mds(self):
        """MDSのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='mds',
            n_components=2,
            metric=True,
            random_state=42
        )
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 2)


class DimensionalityReducerMatrixFactorizationTests(unittest.TestCase):
    """行列分解手法のテスト"""
    
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=15, random_state=42)
        # NMFは非負値が必要
        self.X_nonneg = pd.DataFrame(
            np.abs(X), columns=[f'f{i}' for i in range(15)]
        )
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(15)])
        self.y = pd.Series(y)
    
    def test_nmf(self):
        """NMFのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='nmf',
            n_components=5,
            init='nndsvda',
            random_state=42
        )
        X_reduced = reducer.fit_transform(self.X_nonneg)
        
        self.assertEqual(X_reduced.shape[1], 5)
    
    def test_factor_analysis(self):
        """Factor Analysisのテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(
            method='factor_analysis',
            n_components=5,
            random_state=42
        )
        X_reduced = reducer.fit_transform(self.X)
        
        self.assertEqual(X_reduced.shape[1], 5)


class DimensionalityReducerHelperTests(unittest.TestCase):
    """ヘルパー関数のテスト"""
    
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
        self.y = pd.Series(y)
    
    def test_auto_reduce_dimensions(self):
        """auto_reduce_dimensionsのテスト"""
        from core.services.ml.dimensionality_reduction import auto_reduce_dimensions
        
        X_reduced, info = auto_reduce_dimensions(
            self.X,
            method='pca',
            target_variance=0.95,
        )
        
        self.assertLessEqual(X_reduced.shape[1], self.X.shape[1])
        self.assertGreaterEqual(info['explained_variance'], 0.95)
        self.assertEqual(info['target_variance'], 0.95)
    
    def test_visualize_2d(self):
        """visualize_2dのテスト"""
        from core.services.ml.dimensionality_reduction import visualize_2d
        
        X_2d = visualize_2d(self.X, method='pca')
        
        self.assertEqual(X_2d.shape[1], 2)
        self.assertIn('x', X_2d.columns)
        self.assertIn('y', X_2d.columns)
    
    def test_inverse_transform(self):
        """逆変換のテスト"""
        from core.services.ml.dimensionality_reduction import DimensionalityReducer
        
        reducer = DimensionalityReducer(method='pca', n_components=5)
        X_reduced = reducer.fit_transform(self.X)
        X_reconstructed = reducer.inverse_transform(X_reduced)
        
        # 元の次元に戻ることを確認
        self.assertEqual(X_reconstructed.shape[1], self.X.shape[1])
