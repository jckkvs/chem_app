"""
sklearn.impute拡張ラッパー

既存に加え、4種類の欠損値補完手法を追加。
メンテナブル設計: 250行程度、全引数カスタマイズ可能。

Implements: sklearn.impute完全対応
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ImputeWrapper:
    """
    sklearn.imputeの拡張ラッパー
    
    全4種類の欠損値補完手法をサポート:
    - SimpleImputer: 平均値/中央値/最頻値/定数で補完
    - KNNImputer: K近傍法で補完
    - IterativeImputer: MICE（多重代入、反復的補完）
    - MissingIndicator: 欠損値指標（補完ではなく指標作成）
    
    全引数カスタマイズ可能（**kwargs透過）
    
    Example:
        >>> # SimpleImputer（平均値）
        >>> imputer = ImputeWrapper(method='simple', strategy='mean')
        >>> X_imputed = imputer.fit_transform(X)
        >>> 
        >>> # KNNImputer
        >>> imputer = ImputeWrapper(method='knn', n_neighbors=5)
        >>> X_imputed = imputer.fit_transform(X)
        >>> 
        >>> # IterativeImputer（MICE）
        >>> imputer = ImputeWrapper(
        ...     method='iterative',
        ...     max_iter=10,
        ...     random_state=42
        ... )
        >>> X_imputed = imputer.fit_transform(X)
    """
    
    def __init__(
        self,
        method: Literal['auto', 'simple', 'knn', 'iterative', 'missing_indicator'] = 'auto',
        **params
    ):
        """
        Args:
            method: 補完手法
            **params: 手法固有の全パラメータ
        """
        self.method = method
        self.params = params
        self.imputer_: Optional[BaseEstimator] = None
        self.selected_method_: Optional[str] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'ImputeWrapper':
        """
        フィット
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット（未使用）
        
        Returns:
            self
        """
        # 手法自動選択
        if self.method == 'auto':
            self.selected_method_ = self._auto_select_method(X)
        else:
            self.selected_method_ = self.method
        
        # Imputer作成
        self.imputer_ = self._create_imputer()
        
        logger.info(f"Impute fit開始: method={self.selected_method_}")
        self.imputer_.fit(X.values)
        logger.info("Impute fit完了")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        if self.imputer_ is None:
            raise ValueError("Imputer not fitted. Call fit() first.")
        
        X_imputed = self.imputer_.transform(X.values)
        
        # MissingIndicatorは列数が変わる可能性がある
        if self.selected_method_ == 'missing_indicator':
            n_features = X_imputed.shape[1]
            columns = [f'missing_{i}' for i in range(n_features)]
        else:
            columns = X.columns.tolist()
        
        return pd.DataFrame(
            X_imputed,
            columns=columns,
            index=X.index
        )
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """フィット＆変換"""
        return self.fit(X, y).transform(X)
    
    def _auto_select_method(self, X: pd.DataFrame) -> str:
        """データに基づいて手法自動選択"""
        
        # 欠損率計算
        missing_rate = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        
        # 欠損値なし → そのまま
        if missing_rate == 0:
            logger.info("No missing values detected")
            return 'simple'  # ダミー
        
        # 欠損率が高い（>30%） → Iterative
        if missing_rate > 0.3:
            return 'iterative'
        
        # 欠損率が中程度（10-30%） → KNN
        if missing_rate > 0.1:
            return 'knn'
        
        # 欠損率が低い（<10%） → Simple（平均値）
        return 'simple'
    
    def _create_imputer(self) -> BaseEstimator:
        """手法に応じてImputer作成"""
        
        if self.selected_method_ == 'simple':
            from sklearn.impute import SimpleImputer
            strategy = self.params.pop('strategy', 'mean')
            return SimpleImputer(strategy=strategy, **self.params)
        
        elif self.selected_method_ == 'knn':
            from sklearn.impute import KNNImputer
            n_neighbors = self.params.pop('n_neighbors', 5)
            return KNNImputer(n_neighbors=n_neighbors, **self.params)
        
        elif self.selected_method_ == 'iterative':
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            max_iter = self.params.pop('max_iter', 10)
            return IterativeImputer(max_iter=max_iter, **self.params)
        
        elif self.selected_method_ == 'missing_indicator':
            from sklearn.impute import MissingIndicator
            features = self.params.pop('features', 'missing-only')
            return MissingIndicator(features=features, **self.params)
        
        else:
            raise ValueError(f"Unknown method: {self.selected_method_}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """利用可能な手法一覧"""
        return ['simple', 'knn', 'iterative', 'missing_indicator']
    
    def get_impute_info(self) -> Dict[str, Any]:
        """補完情報取得"""
        info = {
            'method': self.selected_method_ or 'not fitted',
            'params': self.params,
        }
        
        if self.imputer_ is not None and hasattr(self.imputer_, 'statistics_'):
            info['statistics'] = self.imputer_.statistics_
        
        return info


# =============================================================================
# ヘルパー関数
# =============================================================================

def auto_impute(
    X: pd.DataFrame,
    method: str = 'auto',
    **params
) -> pd.DataFrame:
    """
    自動欠損値補完
    
    Args:
        X: 特徴量
        method: 補完手法（'auto'で自動選択）
        **params: パラメータ
    
    Returns:
        補完後DataFrame
    
    Example:
        >>> X_imputed = auto_impute(X, method='auto')
    """
    imputer = ImputeWrapper(method=method, **params)
    X_imputed = imputer.fit_transform(X)
    
    logger.info(f"Auto impute: method={imputer.selected_method_}")
    
    return X_imputed


def get_missing_summary(X: pd.DataFrame) -> pd.DataFrame:
    """
    欠損値サマリー取得
    
    Args:
        X: DataFrame
    
    Returns:
        欠損値サマリーDataFrame
    """
    summary = pd.DataFrame({
        'missing_count': X.isnull().sum(),
        'missing_rate': X.isnull().sum() / len(X),
        'dtype': X.dtypes
    })
    
    summary = summary[summary['missing_count'] > 0].sort_values(
        'missing_rate', ascending=False
    )
    
    return summary
