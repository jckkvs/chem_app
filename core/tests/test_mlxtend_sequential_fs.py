"""
mlxtend Sequential FSのテスト

Implements: TEST-MLXTEND-SFS-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# mlxtendの可用性チェック
try:
    import mlxtend
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


@unittest.skipUnless(MLXTEND_AVAILABLE, "mlxtend not installed")
class MLXtendSequentialFSTests(unittest.TestCase):
    """Sequential Feature Selectionのテスト"""
    
    def setUp(self):
        # 小規模データセット（Sequential FSは計算コストが高い）
        X, y = make_classification(
            n_samples=50,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        self.y = pd.Series(y)
    
    def test_sfs(self):
        """SFS（Sequential Forward Selection）のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import MLXtendSequentialFS
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        selector = MLXtendSequentialFS(
            method='sfs',
            estimator=estimator,
            k_features=5,
            cv=3,
            scoring='accuracy'
        )
        
        X_selected = selector.fit_transform(self.X, self.y)
        
        # 5つの特徴が選択される
        self.assertEqual(X_selected.shape[1], 5)
        self.assertEqual(len(selector.get_selected_features()), 5)
    
    def test_sbs(self):
        """SBS（Sequential Backward Selection）のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import MLXtendSequentialFS
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        selector = MLXtendSequentialFS(
            method='sbs',
            estimator=estimator,
            k_features=5,
            cv=3
        )
        
        X_selected = selector.fit_transform(self.X, self.y)
        
        self.assertEqual(X_selected.shape[1], 5)
    
    def test_sffs(self):
        """SFFS（Sequential Floating Forward Selection）のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import MLXtendSequentialFS
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        selector = MLXtendSequentialFS(
            method='sffs',
            estimator=estimator,
            k_features=5,
            cv=3
        )
        
        X_selected = selector.fit_transform(self.X, self.y)
        
        self.assertEqual(X_selected.shape[1], 5)
    
    def test_sbfs(self):
        """SBFS（Sequential Floating Backward Selection）のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import MLXtendSequentialFS
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        selector = MLXtendSequentialFS(
            method='sbfs',
            estimator=estimator,
            k_features=5,
            cv=3
        )
        
        X_selected = selector.fit_transform(self.X, self.y)
        
        self.assertEqual(X_selected.shape[1], 5)
    
    def test_best_score(self):
        """ベストスコア取得のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import MLXtendSequentialFS
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        selector = MLXtendSequentialFS(
            method='sfs',
            estimator=estimator,
            k_features=5,
            cv=3
        )
        
        selector.fit(self.X, self.y)
        best_score = selector.get_best_score()
        
        # スコアは0〜1の範囲
        self.assertGreaterEqual(best_score, 0.0)
        self.assertLessEqual(best_score, 1.0)


@unittest.skipUnless(MLXTEND_AVAILABLE, "mlxtend not installed")
class MLXtendSequentialFSHelperTests(unittest.TestCase):
    """ヘルパー関数のテスト"""
    
    def setUp(self):
        X, y = make_classification(
            n_samples=50, n_features=10, n_informative=5, random_state=42
        )
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        self.y = pd.Series(y)
    
    def test_sequential_feature_selection(self):
        """sequential_feature_selection関数のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import sequential_feature_selection
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        X_selected, selected_features, best_score = sequential_feature_selection(
            self.X, self.y, estimator=estimator, k_features=5, cv=3
        )
        
        self.assertEqual(X_selected.shape[1], 5)
        self.assertEqual(len(selected_features), 5)
        self.assertIsInstance(best_score, float)
    
    def test_compare_sequential_methods(self):
        """compare_sequential_methods関数のテスト"""
        from sklearn.neighbors import KNeighborsClassifier
        from core.services.ml.mlxtend_sequential_fs import compare_sequential_methods
        
        estimator = KNeighborsClassifier(n_neighbors=3)
        results = compare_sequential_methods(
            self.X, self.y, estimator=estimator, k_features=5
        )
        
        # 4種類の手法が比較される
        self.assertIn('sfs', results)
        self.assertIn('sbs', results)
        self.assertIn('sffs', results)
        self.assertIn('sbfs', results)
        
        # 各手法の結果を確認
        for method, result in results.items():
            if 'error' not in result:
                self.assertIn('selected_features', result)
                self.assertIn('best_score', result)
