"""
sklearn.cluster拡張ラッパー

既存KMeansに加え、6種類のクラスタリング手法を追加。
メンテナブル設計: 300行程度、単一責任原則適用。

Implements: sklearn.cluster完全対応
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ClusteringWrapper:
    """
    sklearn.clusterの拡張ラッパー
    
    全7種類のクラスタリング手法をサポート:
    - KMeans: 最も基本的（既存も利用可能）
    - DBSCAN: 密度ベース、クラスタ数不要
    - AgglomerativeClustering: 階層的
    - SpectralClustering: グラフベース
    - MeanShift: 密度ピーク検出
    - OPTICS: DBSCAN拡張
    - Birch: 大規模データ向け
    
    Example:
        >>> # 自動選択
        >>> cluster = ClusteringWrapper(method='auto')
        >>> labels = cluster.fit_predict(X)
        >>> 
        >>> # 手動選択
        >>> cluster = ClusteringWrapper(method='dbscan', eps=0.5)
        >>> labels = cluster.fit_predict(X)
    """
    
    def __init__(
        self,
        method: Literal['auto', 'kmeans', 'dbscan', 'agglomerative', 'spectral', 'meanshift', 'optics', 'birch'] = 'auto',
        n_clusters: Optional[int] = None,
        **params
    ):
        """
        Args:
            method: クラスタリング手法
            n_clusters: クラスタ数（手法により必須/不要）
            **params: 手法固有のパラメータ
        """
        self.method = method
        self.n_clusters = n_clusters
        self.params = params
        self.model_: Optional[BaseEstimator] = None
        self.selected_method_: Optional[str] = None
        self.labels_: Optional[np.ndarray] = None
    
    def fit(self, X: pd.DataFrame) -> 'ClusteringWrapper':
        """
        クラスタリング実行
        
        Args:
            X: 特徴量DataFrame
        
        Returns:
            self
        """
        # 手法自動選択
        if self.method == 'auto':
            self.selected_method_ = self._auto_select_method(X)
        else:
            self.selected_method_ = self.method
        
        # モデル作成
        self.model_ = self._create_model(self.selected_method_)
        
        logger.info(f"Clustering開始: method={self.selected_method_}")
        self.labels_ = self.model_.fit_predict(X.values)
        logger.info(f"Clustering完了: n_clusters={len(np.unique(self.labels_))}")
        
        return self
    
    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """クラスタリング＆ラベル返却"""
        self.fit(X)
        return self.labels_
    
    def _auto_select_method(self, X: pd.DataFrame) -> str:
        """データに基づいて手法自動選択"""
        
        n_samples = len(X)
        
        # 大規模データ → Birch
        if n_samples > 10000:
            return 'birch'
        
        # 中規模 → KMeans（高速）
        if n_samples > 1000:
            return 'kmeans'
        
        # 小規模 → DBSCAN（クラスタ数不要）
        return 'dbscan'
    
    def _create_model(self, method: str) -> BaseEstimator:
        """手法に応じてモデル作成"""
        
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            n_clusters = self.n_clusters or 3
            return KMeans(n_clusters=n_clusters, **self.params)
        
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            eps = self.params.pop('eps', 0.5)
            min_samples = self.params.pop('min_samples', 5)
            return DBSCAN(eps=eps, min_samples=min_samples, **self.params)
        
        elif method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = self.n_clusters or 3
            linkage = self.params.pop('linkage', 'ward')
            return AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage, **self.params
            )
        
        elif method == 'spectral':
            from sklearn.cluster import SpectralClustering
            n_clusters = self.n_clusters or 3
            return SpectralClustering(n_clusters=n_clusters, **self.params)
        
        elif method == 'meanshift':
            from sklearn.cluster import MeanShift
            return MeanShift(**self.params)
        
        elif method == 'optics':
            from sklearn.cluster import OPTICS
            min_samples = self.params.pop('min_samples', 5)
            return OPTICS(min_samples=min_samples, **self.params)
        
        elif method == 'birch':
            from sklearn.cluster import Birch
            n_clusters = self.n_clusters or 3
            return Birch(n_clusters=n_clusters, **self.params)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """利用可能な手法一覧"""
        return ['kmeans', 'dbscan', 'agglomerative', 'spectral', 'meanshift', 'optics', 'birch']
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """クラスタ情報取得"""
        if self.labels_ is None:
            return {'status': 'not fitted'}
        
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels)
        n_noise = np.sum(self.labels_ == -1) if -1 in unique_labels else 0
        
        return {
            'method': self.selected_method_,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_sizes': {
                int(label): int(np.sum(self.labels_ == label))
                for label in unique_labels
            }
        }


# =============================================================================
# ヘルパー関数
# =============================================================================

def auto_cluster(
    X: pd.DataFrame,
    method: str = 'auto',
    n_clusters: Optional[int] = None,
    **params
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    自動クラスタリング
    
    Args:
        X: 特徴量
        method: 手法
        n_clusters: クラスタ数
        **params: パラメータ
    
    Returns:
        Tuple[ラベル, クラスタ情報]
    
    Example:
        >>> labels, info = auto_cluster(X, method='auto')
        >>> print(f"Found {info['n_clusters']} clusters")
    """
    clustering = ClusteringWrapper(method=method, n_clusters=n_clusters, **params)
    labels = clustering.fit_predict(X)
    info = clustering.get_cluster_info()
    
    logger.info(f"Auto clustering: {info}")
    
    return labels, info


def compare_clustering_methods(
    X: pd.DataFrame,
    methods: Optional[List[str]] = None,
    n_clusters: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    複数のクラスタリング手法を比較
    
    Args:
        X: 特徴量
        methods: 比較する手法リスト
        n_clusters: クラスタ数
    
    Returns:
        Dict: 各手法の結果
    """
    if methods is None:
        methods = ['kmeans', 'dbscan', 'agglomerative']
    
    results = {}
    
    for method in methods:
        try:
            clustering = ClusteringWrapper(method=method, n_clusters=n_clusters)
            clustering.fit(X)
            results[method] = clustering.get_cluster_info()
        except Exception as e:
            logger.warning(f"{method} failed: {e}")
            results[method] = {'error': str(e)}
    
    return results
