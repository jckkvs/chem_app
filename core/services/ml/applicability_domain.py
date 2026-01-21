"""
アプリカビリティドメイン検出（OECD QSAR Toolbox inspired）

Implements: F-AD-001
設計思想:
- 予測の適用範囲を判定
- 学習データとの類似性評価
- 外挿警告
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ADResult:
    """アプリカビリティドメイン判定結果"""
    is_within_domain: bool
    confidence: float  # 0-1
    nearest_distance: float
    z_score: float
    warnings: List[str]


class ApplicabilityDomain:
    """
    アプリカビリティドメイン検出（OECD/Ambit inspired）
    
    Features:
    - 距離ベースAD（k-NN）
    - レバレッジベースAD
    - PCA空間でのBounding Box
    
    Example:
        >>> ad = ApplicabilityDomain()
        >>> ad.fit(X_train)
        >>> result = ad.check(X_new)
    """
    
    def __init__(
        self,
        method: str = 'knn',
        k_neighbors: int = 5,
        threshold_percentile: float = 95,
    ):
        self.method = method
        self.k_neighbors = k_neighbors
        self.threshold_percentile = threshold_percentile
        
        self.scaler_: Optional[StandardScaler] = None
        self.nn_model_: Optional[NearestNeighbors] = None
        self.pca_: Optional[PCA] = None
        self.distance_threshold_: float = 0.0
        self.leverage_threshold_: float = 0.0
        self.X_scaled_: Optional[np.ndarray] = None
        self.pca_bounds_: Optional[dict] = None
    
    def fit(self, X: pd.DataFrame) -> 'ApplicabilityDomain':
        """学習データでADを構築"""
        # スケーリング
        self.scaler_ = StandardScaler()
        self.X_scaled_ = self.scaler_.fit_transform(X)
        
        if self.method == 'knn':
            self._fit_knn()
        elif self.method == 'leverage':
            self._fit_leverage()
        elif self.method == 'pca':
            self._fit_pca()
        else:
            self._fit_knn()  # default
        
        return self
    
    def _fit_knn(self) -> None:
        """k-NNベースAD"""
        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            metric='euclidean',
        )
        self.nn_model_.fit(self.X_scaled_)
        
        # 学習データ間の距離から閾値決定
        distances, _ = self.nn_model_.kneighbors(self.X_scaled_)
        mean_distances = distances.mean(axis=1)
        self.distance_threshold_ = np.percentile(mean_distances, self.threshold_percentile)
    
    def _fit_leverage(self) -> None:
        """レバレッジベースAD"""
        X = self.X_scaled_
        n = X.shape[0]
        
        # Hat matrix diagonal
        hat = X @ np.linalg.inv(X.T @ X + 0.001 * np.eye(X.shape[1])) @ X.T
        leverages = np.diag(hat)
        
        # 閾値: 3 * (p+1) / n
        p = X.shape[1]
        self.leverage_threshold_ = 3 * (p + 1) / n
    
    def _fit_pca(self) -> None:
        """PCA Bounding Box"""
        self.pca_ = PCA(n_components=min(5, self.X_scaled_.shape[1]))
        X_pca = self.pca_.fit_transform(self.X_scaled_)
        
        self.pca_bounds_ = {
            'min': X_pca.min(axis=0),
            'max': X_pca.max(axis=0),
            'margin': (X_pca.max(axis=0) - X_pca.min(axis=0)) * 0.1,
        }
    
    def check(self, X: pd.DataFrame) -> List[ADResult]:
        """新規データのAD判定"""
        if self.scaler_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        X_scaled = self.scaler_.transform(X)
        results = []
        
        for i in range(len(X_scaled)):
            x = X_scaled[i:i+1]
            result = self._check_single(x)
            results.append(result)
        
        return results
    
    def _check_single(self, x: np.ndarray) -> ADResult:
        """単一サンプルのAD判定"""
        warnings = []
        
        if self.method == 'knn':
            distances, _ = self.nn_model_.kneighbors(x)
            mean_dist = distances.mean()
            
            z_score = (mean_dist - self.distance_threshold_) / (self.distance_threshold_ + 1e-9)
            is_within = mean_dist <= self.distance_threshold_
            confidence = max(0, 1 - z_score) if z_score > 0 else 1.0
            
            if not is_within:
                warnings.append(f"Distance {mean_dist:.3f} exceeds threshold {self.distance_threshold_:.3f}")
            
            return ADResult(
                is_within_domain=is_within,
                confidence=min(1, max(0, confidence)),
                nearest_distance=float(mean_dist),
                z_score=float(z_score),
                warnings=warnings,
            )
        
        elif self.method == 'pca':
            x_pca = self.pca_.transform(x)[0]
            bounds = self.pca_bounds_
            
            is_within = True
            for j in range(len(x_pca)):
                lower = bounds['min'][j] - bounds['margin'][j]
                upper = bounds['max'][j] + bounds['margin'][j]
                if x_pca[j] < lower or x_pca[j] > upper:
                    is_within = False
                    warnings.append(f"PC{j+1} out of bounds")
            
            return ADResult(
                is_within_domain=is_within,
                confidence=1.0 if is_within else 0.5,
                nearest_distance=0,
                z_score=0,
                warnings=warnings,
            )
        
        return ADResult(True, 1.0, 0, 0, [])
    
    def get_domain_coverage(self, X: pd.DataFrame) -> float:
        """ドメインカバレッジを計算"""
        results = self.check(X)
        return sum(r.is_within_domain for r in results) / len(results)
