"""
特徴選択エンジン

Implements: F-FEATSEL-001
設計思想:
- フィルタ法
- ラッパー法
- 埋め込み法
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression,
    RFE,
    VarianceThreshold,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """特徴選択結果"""
    selected_features: List[str]
    n_selected: int
    importance_scores: Optional[dict] = None


class FeatureSelector:
    """
    特徴選択エンジン
    
    Features:
    - 分散閾値
    - 相互情報量
    - RFE（再帰的特徴削除）
    - モデルベース選択
    
    Example:
        >>> selector = FeatureSelector(method='mutual_info', n_features=50)
        >>> result = selector.fit_select(X, y)
    """
    
    def __init__(
        self,
        method: str = 'variance',
        n_features: Optional[int] = None,
        threshold: float = 0.01,
    ):
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.selector_: Optional[Any] = None
        self.selected_features_: List[str] = []
    
    def fit_select(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> SelectionResult:
        """特徴選択を実行"""
        self.selected_features_ = list(X.columns)
        
        if self.method == 'variance':
            return self._variance_selection(X)
        
        elif self.method == 'mutual_info':
            if y is None:
                raise ValueError("y required for mutual_info")
            return self._mutual_info_selection(X, y)
        
        elif self.method == 'rfe':
            if y is None:
                raise ValueError("y required for RFE")
            return self._rfe_selection(X, y)
        
        elif self.method == 'correlation':
            return self._correlation_selection(X)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _variance_selection(self, X: pd.DataFrame) -> SelectionResult:
        """分散閾値"""
        self.selector_ = VarianceThreshold(threshold=self.threshold)
        self.selector_.fit(X)
        
        mask = self.selector_.get_support()
        selected = [col for col, m in zip(X.columns, mask) if m]
        
        return SelectionResult(
            selected_features=selected,
            n_selected=len(selected),
        )
    
    def _mutual_info_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SelectionResult:
        """相互情報量"""
        k = self.n_features or max(10, len(X.columns) // 2)
        
        self.selector_ = SelectKBest(
            score_func=mutual_info_regression,
            k=min(k, len(X.columns)),
        )
        self.selector_.fit(X, y)
        
        mask = self.selector_.get_support()
        selected = [col for col, m in zip(X.columns, mask) if m]
        
        scores = dict(zip(X.columns, self.selector_.scores_))
        
        return SelectionResult(
            selected_features=selected,
            n_selected=len(selected),
            importance_scores=scores,
        )
    
    def _rfe_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SelectionResult:
        """RFE"""
        from sklearn.ensemble import RandomForestRegressor
        
        n_features = self.n_features or max(10, len(X.columns) // 2)
        
        base_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.selector_ = RFE(base_model, n_features_to_select=n_features, step=0.1)
        self.selector_.fit(X, y)
        
        mask = self.selector_.get_support()
        selected = [col for col, m in zip(X.columns, mask) if m]
        
        return SelectionResult(
            selected_features=selected,
            n_selected=len(selected),
        )
    
    def _correlation_selection(self, X: pd.DataFrame) -> SelectionResult:
        """高相関特徴削除"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        selected = [col for col in X.columns if col not in to_drop]
        
        return SelectionResult(
            selected_features=selected,
            n_selected=len(selected),
        )
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """選択した特徴のみ抽出"""
        if self.selector_ is not None:
            mask = self.selector_.get_support()
            return X.loc[:, mask]
        return X[self.selected_features_]
