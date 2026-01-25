"""
MetricsCalculatorのテスト

Implements: TEST-METRICS-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.cluster import KMeans


class MetricsCalculatorRegressionTests(unittest.TestCase):
    """回帰メトリクスのテスト"""
    
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        self.y_true = y
        # 予測値を真値に少しノイズを追加して作成
        self.y_pred = y + np.random.RandomState(42).normal(0, 1, size=y.shape)
    
    def test_regression_metrics(self):
        """回帰メトリクス計算のテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='regression')
        metrics = calc.calculate_regression_metrics(self.y_true, self.y_pred)
        
        # 必須メトリクスの存在確認
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # メトリクス値の妥当性確認
        self.assertGreater(metrics['r2'], 0.5)  # 相関が高いはず
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['mse'], 0)
    
    def test_calculate_all_regression(self):
        """calculate_allメソッド（回帰）のテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='regression')
        metrics = calc.calculate_all(self.y_true, self.y_pred)
        
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 5)


class MetricsCalculatorClassificationTests(unittest.TestCase):
    """分類メトリクスのテスト"""
    
    def setUp(self):
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        self.y_true = y
        # 高精度の予測を作成
        self.y_pred = y.copy()
        # 少しだけ誤分類を混ぜる
        flip_idx = np.random.RandomState(42).choice(len(y), size=10, replace=False)
        self.y_pred[flip_idx] = 1 - self.y_pred[flip_idx]
        
        # 予測確率（ダミー）
        self.y_pred_proba = np.column_stack([1 - self.y_pred, self.y_pred]) + \
                             np.random.RandomState(42).uniform(-0.1, 0.1, size=(len(y), 2))
        self.y_pred_proba = np.clip(self.y_pred_proba, 0, 1)
        self.y_pred_proba /= self.y_pred_proba.sum(axis=1, keepdims=True)
    
    def test_classification_metrics(self):
        """分類メトリクス計算のテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='classification')
        metrics = calc.calculate_classification_metrics(
            self.y_true, self.y_pred, average='binary'
        )
        
        # 必須メトリクスの存在確認
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # メトリクス値の妥当性確認
        self.assertGreater(metrics['accuracy'], 0.7)  # 70%以上の精度
        self.assertEqual(metrics['confusion_matrix'].shape, (2, 2))
    
    def test_classification_metrics_with_proba(self):
        """確率付き分類メトリクスのテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='classification')
        metrics = calc.calculate_classification_metrics(
            self.y_true, self.y_pred, self.y_pred_proba, average='binary'
        )
        
        # 確率ベースメトリクス
        self.assertIn('roc_auc', metrics)
        self.assertIn('log_loss', metrics)
        self.assertIn('average_precision', metrics)
        
        # ROC-AUCは0.5以上
        self.assertGreater(metrics['roc_auc'], 0.5)
    
    def test_calculate_all_classification(self):
        """calculate_allメソッド（分類）のテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='classification')
        metrics = calc.calculate_all(self.y_true, self.y_pred, self.y_pred_proba)
        
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 8)


class MetricsCalculatorClusteringTests(unittest.TestCase):
    """クラスタリングメトリクスのテスト"""
    
    def setUp(self):
        X, y_true = make_classification(
            n_samples=100, n_features=10, n_classes=3,
            n_informative=8, n_redundant=0, random_state=42
        )
        self.X = X
        self.y_true = y_true
        
        # クラスタリング
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(X)
    
    def test_clustering_metrics_internal(self):
        """クラスタリングメトリクス（内部指標）のテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='clustering')
        metrics = calc.calculate_clustering_metrics(self.X, self.labels)
        
        # 内部指標の存在確認
        self.assertIn('silhouette', metrics)
        self.assertIn('calinski_harabasz', metrics)
        self.assertIn('davies_bouldin', metrics)
        
        # Silhouetteスコアは-1〜1
        self.assertGreaterEqual(metrics['silhouette'], -1)
        self.assertLessEqual(metrics['silhouette'], 1)
    
    def test_clustering_metrics_external(self):
        """クラスタリングメトリクス（外部指標）のテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='clustering')
        metrics = calc.calculate_clustering_metrics(
            self.X, self.labels, self.y_true
        )
        
        # 外部指標の存在確認
        self.assertIn('adjusted_rand_score', metrics)
        self.assertIn('adjusted_mutual_info', metrics)
        self.assertIn('homogeneity', metrics)
        self.assertIn('completeness', metrics)
        self.assertIn('v_measure', metrics)


class MetricsCalculatorHelperTests(unittest.TestCase):
    """ヘルパー関数のテスト"""
    
    def test_evaluate_model_regression(self):
        """evaluate_model（回帰）のテスト"""
        from core.services.ml.metrics_calculator import evaluate_model
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = evaluate_model(y_true, y_pred, task_type='regression')
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('r2', metrics)
        self.assertIn('mae', metrics)
    
    def test_evaluate_model_classification(self):
        """evaluate_model（分類）のテスト"""
        from core.services.ml.metrics_calculator import evaluate_model
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        metrics = evaluate_model(y_true, y_pred, task_type='classification')
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
    
    def test_get_best_metrics(self):
        """get_best_metricsのテスト"""
        from core.services.ml.metrics_calculator import MetricsCalculator
        
        calc = MetricsCalculator(task_type='regression')
        metrics = {'r2': 0.85, 'mae': 1.2, 'rmse': 1.5}
        
        best = calc.get_best_metrics(metrics)
        
        self.assertEqual(best['best_metric'], 'r2')
        self.assertEqual(best['best_value'], 0.85)
