"""
モデル圧縮（知識蒸留/量子化）

Implements: F-COMPRESS-001
設計思想:
- 知識蒸留
- 特徴選択による軽量化
- 推論高速化
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
class CompressionResult:
    """圧縮結果"""
    original_features: int
    compressed_features: int
    compression_ratio: float
    original_score: float
    compressed_score: float
    speedup: float


class ModelCompressor:
    """
    モデル圧縮
    
    Features:
    - 知識蒸留
    - 特徴選択による軽量化
    - モデルプルーニング
    
    Example:
        >>> compressor = ModelCompressor()
        >>> student = compressor.distill(teacher_model, X, target_features=50)
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.5,
    ):
        self.compression_ratio = compression_ratio
    
    def distill(
        self,
        teacher_model,
        X: pd.DataFrame,
        y: pd.Series,
        student_model=None,
        target_features: Optional[int] = None,
    ) -> CompressionResult:
        """知識蒸留"""
        import time
        
        original_features = X.shape[1]
        
        # 教師モデルの予測時間
        start = time.time()
        teacher_preds = teacher_model.predict(X)
        teacher_time = time.time() - start
        
        teacher_score = teacher_model.score(X, y)
        
        # 特徴選択で圧縮
        if target_features is None:
            target_features = int(original_features * self.compression_ratio)
        
        X_compressed = self._select_features(X, teacher_model, target_features)
        
        # 生徒モデル
        if student_model is None:
            from sklearn.ensemble import RandomForestRegressor
            student_model = RandomForestRegressor(n_estimators=50, max_depth=6)
        
        # 教師の予測で学習（ソフトラベル）
        student_model = clone(student_model)
        student_model.fit(X_compressed, teacher_preds)
        
        # 生徒モデルの予測時間
        start = time.time()
        student_preds = student_model.predict(X_compressed)
        student_time = time.time() - start
        
        student_score = student_model.score(X_compressed, y)
        
        return CompressionResult(
            original_features=original_features,
            compressed_features=X_compressed.shape[1],
            compression_ratio=X_compressed.shape[1] / original_features,
            original_score=teacher_score,
            compressed_score=student_score,
            speedup=teacher_time / (student_time + 1e-9),
        )
    
    def _select_features(
        self,
        X: pd.DataFrame,
        model,
        n_features: int,
    ) -> pd.DataFrame:
        """特徴選択"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            selected_cols = [X.columns[i] for i in indices]
            return X[selected_cols]
        else:
            # 分散が大きい特徴を選択
            variances = X.var()
            selected_cols = variances.nlargest(n_features).index.tolist()
            return X[selected_cols]
    
    def prune_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        prune_ratio: float = 0.3,
    ) -> Any:
        """モデルプルーニング（木モデル用）"""
        pruned_model = clone(model)
        
        # 木の数を削減
        if hasattr(pruned_model, 'n_estimators'):
            original_n = pruned_model.n_estimators
            pruned_model.set_params(n_estimators=int(original_n * (1 - prune_ratio)))
        
        # 深さを制限
        if hasattr(pruned_model, 'max_depth'):
            current_depth = pruned_model.max_depth or 10
            pruned_model.set_params(max_depth=max(3, int(current_depth * (1 - prune_ratio))))
        
        pruned_model.fit(X, y)
        
        return pruned_model
