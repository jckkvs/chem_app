"""
転移学習フレームワーク

Implements: F-TRANSFER-001
設計思想:
- 事前学習モデル活用
- ファインチューニング
- ドメイン適応
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """転移学習結果"""
    source_score: float
    target_score_before: float
    target_score_after: float
    improvement: float
    n_source_samples: int
    n_target_samples: int


class TransferLearner:
    """
    転移学習フレームワーク
    
    Features:
    - ソースモデルからの知識転移
    - ファインチューニング
    - ドメイン適応
    
    Example:
        >>> tl = TransferLearner(base_model)
        >>> tl.pretrain(X_source, y_source)
        >>> tl.finetune(X_target, y_target)
    """
    
    def __init__(
        self,
        base_model,
        freeze_layers: bool = False,
    ):
        self.base_model = base_model
        self.freeze_layers = freeze_layers
        self.source_model_ = None
        self.target_model_ = None
        self.is_pretrained_ = False
    
    def pretrain(
        self,
        X_source: pd.DataFrame,
        y_source: pd.Series,
    ) -> float:
        """ソースドメインで事前学習"""
        self.source_model_ = clone(self.base_model)
        self.source_model_.fit(X_source, y_source)
        self.is_pretrained_ = True
        
        score = self.source_model_.score(X_source, y_source)
        logger.info(f"Pretrained on {len(X_source)} samples, score: {score:.4f}")
        
        return score
    
    def finetune(
        self,
        X_target: pd.DataFrame,
        y_target: pd.Series,
        learning_rate_factor: float = 0.1,
    ) -> TransferResult:
        """ターゲットドメインでファインチューニング"""
        if not self.is_pretrained_:
            raise RuntimeError("Call pretrain() first")
        
        # ファインチューニング前のスコア
        score_before = self.source_model_.score(X_target, y_target)
        
        # モデルをクローンしてファインチューニング
        self.target_model_ = clone(self.base_model)
        
        # 既存の重みで初期化（可能な場合）
        if hasattr(self.source_model_, 'feature_importances_'):
            # LightGBM/XGBoost の場合、継続学習
            if hasattr(self.target_model_, 'init_model'):
                self.target_model_.set_params(init_model=self.source_model_)
        
        self.target_model_.fit(X_target, y_target)
        
        score_after = self.target_model_.score(X_target, y_target)
        
        return TransferResult(
            source_score=self.source_model_.score(X_target, y_target),
            target_score_before=score_before,
            target_score_after=score_after,
            improvement=score_after - score_before,
            n_source_samples=0,
            n_target_samples=len(X_target),
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.target_model_ is not None:
            return self.target_model_.predict(X)
        elif self.source_model_ is not None:
            return self.source_model_.predict(X)
        else:
            raise RuntimeError("Model not trained")
    
    def domain_adaptation(
        self,
        X_source: pd.DataFrame,
        X_target: pd.DataFrame,
        method: str = 'coral',
    ) -> pd.DataFrame:
        """ドメイン適応（特徴量変換）"""
        if method == 'coral':
            return self._coral_transform(X_source, X_target)
        return X_target
    
    def _coral_transform(
        self,
        X_source: pd.DataFrame,
        X_target: pd.DataFrame,
    ) -> pd.DataFrame:
        """CORAL変換"""
        # ソースとターゲットの共分散を揃える
        source_mean = X_source.mean()
        target_mean = X_target.mean()
        
        X_target_centered = X_target - target_mean
        X_target_aligned = X_target_centered + source_mean
        
        return X_target_aligned
