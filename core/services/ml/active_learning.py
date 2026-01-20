"""
アクティブラーニングエンジン

Implements: F-AL-001
設計思想:
- 次に測定すべき分子を提案
- 不確実性サンプリング
- 多様性サンプリング
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    アクティブラーニングエンジン
    
    Features:
    - 不確実性ベース選択
    - 多様性ベース選択
    - ハイブリッド選択
    
    Example:
        >>> al = ActiveLearner(model, uncertainty_model)
        >>> next_indices = al.suggest_next(X_pool, n_samples=10)
    """
    
    def __init__(
        self,
        model=None,
        uncertainty_model=None,
        strategy: Literal['uncertainty', 'diversity', 'hybrid'] = 'hybrid',
    ):
        """
        Args:
            model: 予測モデル（オプション）
            uncertainty_model: 不確実性モデル（オプション）
            strategy: 選択戦略
        """
        self.model = model
        self.uncertainty_model = uncertainty_model
        self.strategy = strategy
    
    def suggest_next(
        self,
        X_pool: pd.DataFrame,
        n_samples: int = 10,
        X_train: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """
        次に測定すべきサンプルを提案
        
        Args:
            X_pool: 未ラベルデータプール
            n_samples: 選択するサンプル数
            X_train: 既存の学習データ（多様性計算用）
            
        Returns:
            選択されたインデックスリスト
        """
        n_samples = min(n_samples, len(X_pool))
        
        if self.strategy == 'uncertainty':
            return self._uncertainty_sampling(X_pool, n_samples)
        elif self.strategy == 'diversity':
            return self._diversity_sampling(X_pool, n_samples, X_train)
        else:  # hybrid
            return self._hybrid_sampling(X_pool, n_samples, X_train)
    
    def _uncertainty_sampling(
        self,
        X_pool: pd.DataFrame,
        n_samples: int,
    ) -> List[int]:
        """不確実性ベースサンプリング"""
        if self.uncertainty_model is None:
            # モデルがない場合はランダム
            return list(np.random.choice(len(X_pool), n_samples, replace=False))
        
        # 不確実性スコアを取得
        uncertainties = self.uncertainty_model.get_uncertainty(X_pool)
        
        # 不確実性が高い順にソート
        sorted_indices = np.argsort(uncertainties)[::-1]
        return list(sorted_indices[:n_samples])
    
    def _diversity_sampling(
        self,
        X_pool: pd.DataFrame,
        n_samples: int,
        X_train: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """多様性ベースサンプリング（k-meansクラスタリング）"""
        X_arr = np.array(X_pool)
        
        # クラスタリング
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        kmeans.fit(X_arr)
        
        # 各クラスタから中心に最も近いサンプルを選択
        selected = []
        for i in range(n_samples):
            cluster_mask = kmeans.labels_ == i
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # クラスタ中心に最も近いサンプル
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_arr[cluster_indices] - center, axis=1)
            closest = cluster_indices[np.argmin(distances)]
            selected.append(int(closest))
        
        # 足りない場合はランダム追加
        while len(selected) < n_samples:
            remaining = list(set(range(len(X_pool))) - set(selected))
            if not remaining:
                break
            selected.append(np.random.choice(remaining))
        
        return selected
    
    def _hybrid_sampling(
        self,
        X_pool: pd.DataFrame,
        n_samples: int,
        X_train: Optional[pd.DataFrame] = None,
    ) -> List[int]:
        """ハイブリッドサンプリング（不確実性 + 多様性）"""
        half = n_samples // 2
        
        # 半分を不確実性で選択
        uncertainty_selected = self._uncertainty_sampling(X_pool, half)
        
        # 残りを多様性で選択（既選択を除外）
        remaining_pool = X_pool.drop(index=uncertainty_selected, errors='ignore')
        if len(remaining_pool) > 0:
            diversity_selected = self._diversity_sampling(
                remaining_pool, n_samples - half, X_train
            )
            # インデックスを元のプールに合わせて変換
            original_indices = list(remaining_pool.index)
            diversity_selected = [original_indices[i] for i in diversity_selected if i < len(original_indices)]
        else:
            diversity_selected = []
        
        return uncertainty_selected + diversity_selected[:n_samples - len(uncertainty_selected)]
    
    def expected_improvement(
        self,
        X_pool: pd.DataFrame,
        y_best: float,
    ) -> np.ndarray:
        """
        期待改善量（Expected Improvement）を計算
        
        Args:
            X_pool: 候補データ
            y_best: 現在の最良値
            
        Returns:
            各サンプルのEI値
        """
        if self.uncertainty_model is None:
            return np.zeros(len(X_pool))
        
        mean, lower, upper = self.uncertainty_model.predict_with_interval(X_pool)
        std = (upper - lower) / 4  # 近似標準偏差
        
        # EI計算
        from scipy.stats import norm
        z = (mean - y_best) / (std + 1e-9)
        ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
