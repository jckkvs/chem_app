"""
ClusteringWrapperのテスト
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


class ClusteringWrapperTests(unittest.TestCase):
    """ClusteringWrapperのテスト"""
    
    def setUp(self):
        # クラスタリング用データ
        X, y = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
        self.X = pd.DataFrame(X, columns=['f0', 'f1'])
    
    def test_kmeans(self):
        """KMeansのテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import ClusteringWrapper
        
        cluster = ClusteringWrapper(method='kmeans', n_clusters=3)
        labels = cluster.fit_predict(self.X)
        
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(np.unique(labels)), 3)
    
    def test_dbscan(self):
        """DBSCANのテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import ClusteringWrapper
        
        cluster = ClusteringWrapper(method='dbscan', eps=1.0)
        labels = cluster.fit_predict(self.X)
        
        self.assertEqual(len(labels), len(self.X))
        # DBSCANはノイズポイント（-1）を含む可能性がある
        self.assertGreater(len(np.unique(labels)), 0)
    
    def test_agglomerative(self):
        """AgglomerativeClusteringのテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import ClusteringWrapper
        
        cluster = ClusteringWrapper(method='agglomerative', n_clusters=3)
        labels = cluster.fit_predict(self.X)
        
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(np.unique(labels)), 3)
    
    def test_auto_select(self):
        """自動選択のテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import ClusteringWrapper
        
        cluster = ClusteringWrapper(method='auto')
        labels = cluster.fit_predict(self.X)
        
        self.assertEqual(len(labels), len(self.X))
        self.assertIsNotNone(cluster.selected_method_)
    
    def test_get_cluster_info(self):
        """クラスタ情報取得のテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import ClusteringWrapper
        
        cluster = ClusteringWrapper(method='kmeans', n_clusters=3)
        cluster.fit(self.X)
        info = cluster.get_cluster_info()
        
        self.assertEqual(info['n_clusters'], 3)
        self.assertEqual(info['method'], 'kmeans')
        self.assertIn('cluster_sizes', info)
    
    def test_get_available_methods(self):
        """利用可能手法一覧のテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import ClusteringWrapper
        
        methods = ClusteringWrapper.get_available_methods()
        
        self.assertEqual(len(methods), 7)
        self.assertIn('kmeans', methods)
        self.assertIn('dbscan', methods)
    
    def test_auto_cluster_helper(self):
        """auto_clusterヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules.cluster_extended import auto_cluster
        
        labels, info = auto_cluster(self.X, method='kmeans', n_clusters=3)
        
        self.assertEqual(len(labels), len(self.X))
        self.assertIn('n_clusters', info)
        self.assertEqual(info['n_clusters'], 3)
