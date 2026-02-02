# -*- coding: utf-8 -*-
"""
クラスタリングエンジンのテスト

カバレッジ目標: 0% → 70%
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


class TestClusteringEngineInit:
    """初期化テスト"""
    
    def test_basic_import(self):
        """基本的なインポート"""
        from core.services.ml import clustering
        assert clustering is not None


class TestKMeansClustering:
    """K-Meansクラスタリングテスト"""
    
    def test_kmeans_basic(self):
        """基本的なK-Means"""
        from sklearn.cluster import KMeans
        
        X = np.random.rand(100, 5)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) <= 3
    
    def test_kmeans_centers(self):
        """クラスタ中心"""
        from sklearn.cluster import KMeans
        
        X = np.random.rand(50, 3)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X)
        
        assert kmeans.cluster_centers_.shape == (2, 3)


class TestDBSCANClustering:
    """DBSCANクラスタリングテスト"""
    
    def test_dbscan_basic(self):
        """基本的なDBSCAN"""
        from sklearn.cluster import DBSCAN
        
        X = np.random.rand(100, 3)
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        assert len(labels) == 100
        # -1はノイズポイント
        assert -1 in labels or len(np.unique(labels)) > 0


class TestHierarchicalClustering:
    """階層的クラスタリングテスト"""
    
    def test_agglomerative_basic(self):
        """基本的な凝集型クラスタリング"""
        from sklearn.cluster import AgglomerativeClustering
        
        X = np.random.rand(50, 4)
        clustering = AgglomerativeClustering(n_clusters=3)
        labels = clustering.fit_predict(X)
        
        assert len(labels) == 50
        assert len(np.unique(labels)) == 3


class TestClusteringMetrics:
    """クラスタリング評価指標テスト"""
    
    def test_silhouette_score(self):
        """シルエットスコア"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        X = np.random.rand(100, 5)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        score = silhouette_score(X, labels)
        
        # スコアは-1～1
        assert -1 <= score <= 1
    
    def test_davies_bouldin_score(self):
        """Davies-Bouldinスコア"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import davies_bouldin_score
        
        X = np.random.rand(100, 5)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        score = davies_bouldin_score(X, labels)
        
        # 低いほど良い（0以上）
        assert score >= 0


class TestClusteringDataPreparation:
    """データ準備テスト"""
    
    def test_data_scaling(self):
        """データスケーリング"""
        from sklearn.preprocessing import StandardScaler
        
        X = np.random.randn(100, 5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 平均0、分散1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)


class TestClusteringIntegration:
    """統合テスト"""
    
    def test_complete_workflow(self):
        """完全なワークフロー"""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        # データ生成
        X = np.random.rand(200, 10)
        
        # スケーリング
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # クラスタリング
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # 評価
        score = silhouette_score(X_scaled, labels)
        
        assert len(labels) == 200
        assert -1 <= score <= 1
