"""
sklearn.preprocessing拡張ラッパー

既存に加え、6種類の前処理手法を追加。
メンテナブル設計: 250行程度、全引数カスタマイズ可能。

Implements: sklearn.preprocessing完全対応
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class PreprocessingWrapper:
    """
    sklearn.preprocessingの拡張ラッパー
    
    全10種類の前処理手法をサポート:
    - StandardScaler: 平均0、分散1に標準化（既存も利用可能）
    - MinMaxScaler: 最小値0、最大値1にスケーリング
    - RobustScaler: 外れ値に頑健なスケーリング
    - Normalizer: サンプル単位でL1/L2正規化
    - Binarizer: 閾値で二値化
    - PolynomialFeatures: 多項式特徴量生成
    - KBinsDiscretizer: ビニング（離散化）
    - TargetEncoder: ターゲットエンコーディング（sklearn 1.3+）
    - PowerTransformer: Box-Cox/Yeo-Johnson変換
    - QuantileTransformer: 分位点変換
    
    全引数カスタマイズ可能（**kwargs透過）
    
    Example:
        >>> # 標準化
        >>> prep = PreprocessingWrapper(method='standard')
        >>> X_scaled = prep.fit_transform(X)
        >>> 
        >>> # 多項式特徴量
        >>> prep = PreprocessingWrapper(
        ...     method='polynomial',
        ...     degree=2,
        ...     include_bias=False
        ... )
        >>> X_poly = prep.fit_transform(X)
    """
    
    def __init__(
        self,
        method: Literal['standard', 'minmax', 'robust', 'normalizer', 'binarizer', 'polynomial', 'kbins', 'target_encoder', 'power', 'quantile'] = 'standard',
        **params
    ):
        """
        Args:
            method: 前処理手法
            **params: 手法固有の全パラメータ
        """
        self.method = method
        self.params = params
        self.transformer_: Optional[BaseEstimator] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'PreprocessingWrapper':
        """
        フィット
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット（TargetEncoder等で使用）
        
        Returns:
            self
        """
        self.transformer_ = self._create_transformer()
        
        logger.info(f"Preprocessing fit開始: method={self.method}")
        
        if y is not None:
            self.transformer_.fit(X.values, y.values)
        else:
            self.transformer_.fit(X.values)
        
        logger.info("Preprocessing fit完了")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        if self.transformer_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        X_transformed = self.transformer_.transform(X.values)
        
        # カラム名生成
        if self.method == 'polynomial':
            # PolynomialFeaturesは特徴数が変わる
            n_features = X_transformed.shape[1]
            columns = [f'poly_{i}' for i in range(n_features)]
        else:
            columns = X.columns.tolist()
        
        return pd.DataFrame(
            X_transformed,
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
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆変換（可能な場合）"""
        if self.transformer_ is None:
            raise ValueError("Transformer not fitted.")
        
        if not hasattr(self.transformer_, 'inverse_transform'):
            raise ValueError(f"{self.method} does not support inverse_transform")
        
        X_original = self.transformer_.inverse_transform(X.values)
        
        return pd.DataFrame(X_original, index=X.index)
    
    def _create_transformer(self) -> BaseEstimator:
        """手法に応じてTransformer作成"""
        
        if self.method == 'standard':
            from sklearn.preprocessing import StandardScaler
            return StandardScaler(**self.params)
        
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler(**self.params)
        
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler(**self.params)
        
        elif self.method == 'normalizer':
            from sklearn.preprocessing import Normalizer
            norm = self.params.pop('norm', 'l2')  # l1, l2, max
            return Normalizer(norm=norm, **self.params)
        
        elif self.method == 'binarizer':
            from sklearn.preprocessing import Binarizer
            threshold = self.params.pop('threshold', 0.0)
            return Binarizer(threshold=threshold, **self.params)
        
        elif self.method == 'polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            degree = self.params.pop('degree', 2)
            return PolynomialFeatures(degree=degree, **self.params)
        
        elif self.method == 'kbins':
            from sklearn.preprocessing import KBinsDiscretizer
            n_bins = self.params.pop('n_bins', 5)
            encode = self.params.pop('encode', 'onehot')
            strategy = self.params.pop('strategy', 'quantile')
            return KBinsDiscretizer(
                n_bins=n_bins, encode=encode, strategy=strategy, **self.params
            )
        
        elif self.method == 'target_encoder':
            try:
                from sklearn.preprocessing import TargetEncoder
                return TargetEncoder(**self.params)
            except ImportError:
                raise ImportError(
                    "TargetEncoder requires sklearn>=1.3. "
                    "Upgrade: pip install scikit-learn>=1.3"
                )
        
        elif self.method == 'power':
            from sklearn.preprocessing import PowerTransformer
            method = self.params.pop('power_method', 'yeo-johnson')
            return PowerTransformer(method=method, **self.params)
        
        elif self.method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(**self.params)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """利用可能な手法一覧"""
        return [
            'standard', 'minmax', 'robust', 'normalizer', 'binarizer',
            'polynomial', 'kbins', 'target_encoder', 'power', 'quantile'
        ]


# =============================================================================
# ヘルパー関数
# =============================================================================

def auto_scale(
    X: pd.DataFrame,
    method: str = 'auto',
    **params
) -> pd.DataFrame:
    """
    自動スケーリング
    
    Args:
        X: 特徴量
        method: スケーリング手法（'auto'で自動選択）
        **params: パラメータ
    
    Returns:
        スケーリング後DataFrame
    
    Example:
        >>> X_scaled = auto_scale(X, method='auto')
    """
    if method == 'auto':
        # 外れ値の有無で自動判定
        has_outliers = _has_outliers(X)
        method = 'robust' if has_outliers else 'standard'
        logger.info(f"Auto-selected scaling method: {method}")
    
    prep = PreprocessingWrapper(method=method, **params)
    return prep.fit_transform(X)


def _has_outliers(X: pd.DataFrame, threshold: float = 3.0) -> bool:
    """外れ値の有無を判定（簡易版）"""
    for col in X.columns:
        z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
        if (z_scores > threshold).any():
            return True
    return False
