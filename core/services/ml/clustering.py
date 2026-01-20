"""
分子クラスタリングエンジン（DataWarrior inspired）

Implements: F-CLUSTER-001
設計思想:
- 化学空間でのクラスタリング
- 代表分子選択
- 多様性分析
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """クラスタリング結果"""
    labels: np.ndarray
    n_clusters: int
    cluster_centers: Optional[np.ndarray] = None
    representative_indices: Optional[List[int]] = None
    silhouette_score: Optional[float] = None


class MolecularClusterer:
    """
    分子クラスタリングエンジン（DataWarrior/ChEMBL inspired）
    
    Features:
    - K-Means, Hierarchical, DBSCAN
    - 代表分子選択
    - クラスタ品質評価
    
    Example:
        >>> clusterer = MolecularClusterer(method='kmeans', n_clusters=5)
        >>> result = clusterer.fit(X)
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: int = 10,
        random_state: int = 42,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_: Optional[Any] = None
    
    def fit(self, X: pd.DataFrame) -> ClusterResult:
        """クラスタリング実行"""
        X_arr = np.array(X)
        
        if self.method == 'kmeans':
            self.model_ = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = self.model_.fit_predict(X_arr)
            centers = self.model_.cluster_centers_
            
        elif self.method == 'hierarchical':
            self.model_ = AgglomerativeClustering(n_clusters=self.n_clusters)
            labels = self.model_.fit_predict(X_arr)
            centers = None
            
        elif self.method == 'dbscan':
            self.model_ = DBSCAN(eps=0.5, min_samples=5)
            labels = self.model_.fit_predict(X_arr)
            centers = None
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 代表分子選択
        representatives = self._select_representatives(X_arr, labels, centers)
        
        # Silhouetteスコア
        silhouette = self._calculate_silhouette(X_arr, labels)
        
        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=centers,
            representative_indices=representatives,
            silhouette_score=silhouette,
        )
    
    def _select_representatives(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: Optional[np.ndarray],
    ) -> List[int]:
        """各クラスタから代表を選択"""
        representatives = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:  # ノイズ
                continue
            
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if centers is not None and cluster_id < len(centers):
                center = centers[cluster_id]
                distances = np.linalg.norm(X[cluster_indices] - center, axis=1)
                best_idx = cluster_indices[np.argmin(distances)]
            else:
                best_idx = cluster_indices[0]
            
            representatives.append(int(best_idx))
        
        return representatives
    
    def _calculate_silhouette(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> Optional[float]:
        """Silhouetteスコア計算"""
        try:
            from sklearn.metrics import silhouette_score
            
            if len(set(labels)) < 2:
                return None
            
            return float(silhouette_score(X, labels))
        except Exception:
            return None
    
    def get_cluster_stats(
        self,
        X: pd.DataFrame,
        result: ClusterResult,
    ) -> Dict[int, Dict[str, Any]]:
        """クラスタ統計"""
        stats = {}
        
        for cluster_id in sorted(set(result.labels)):
            cluster_mask = result.labels == cluster_id
            cluster_data = X.iloc[cluster_mask]
            
            stats[cluster_id] = {
                'size': int(cluster_mask.sum()),
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict(),
            }
        
        return stats
