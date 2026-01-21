"""
スマート前処理パイプライン - ColumnTransformerベースの変数タイプ別前処理

Implements: F-PREP-001
設計思想:
- 変数タイプ（連続/カテゴリ/バイナリ）を自動検出
- 初心者には隠蔽、熟練者には設定可能（プログレッシブ開示）
- scikit-learn Pipeline互換で再利用可能

参考文献:
- Feature Engineering for Machine Learning (Zheng & Casari, 2018)
- scikit-learn ColumnTransformer documentation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


# スケーラー定義
SCALERS = {
    'standard': StandardScaler,
    'power': lambda: PowerTransformer(method='yeo-johnson', standardize=True),
    'quantile': lambda: QuantileTransformer(output_distribution='normal', random_state=42),
    'robust': RobustScaler,
    'minmax': MinMaxScaler,
    'none': 'passthrough',
}

# エンコーダー定義
ENCODERS = {
    'onehot': lambda: OneHotEncoder(handle_unknown='ignore', sparse_output=False),
    'ordinal': lambda: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
}

# 欠損値処理定義
IMPUTERS = {
    'mean': lambda: SimpleImputer(strategy='mean'),
    'median': lambda: SimpleImputer(strategy='median'),
    'most_frequent': lambda: SimpleImputer(strategy='most_frequent'),
    'knn': lambda: KNNImputer(n_neighbors=5),
    'zero': lambda: SimpleImputer(strategy='constant', fill_value=0),
}


@dataclass
class ColumnTypeInfo:
    """カラムタイプ分類結果"""
    continuous: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    binary: List[str] = field(default_factory=list)
    integer_count: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {
            'continuous': self.continuous,
            'categorical': self.categorical,
            'binary': self.binary,
            'integer_count': self.integer_count,
        }
    
    def summary(self) -> str:
        return (
            f"連続変数: {len(self.continuous)}件, "
            f"カテゴリ変数: {len(self.categorical)}件, "
            f"バイナリ変数: {len(self.binary)}件, "
            f"整数カウント: {len(self.integer_count)}件"
        )


class OutlierHandler(BaseEstimator, TransformerMixin):
    """外れ値処理トランスフォーマー"""
    
    def __init__(
        self, 
        method: Literal['iqr', 'zscore', 'clip', 'none'] = 'iqr',
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.0,
    ):
        self.method = method
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None
        
    def fit(self, X: np.ndarray, y=None) -> 'OutlierHandler':
        if self.method == 'iqr':
            Q1 = np.nanpercentile(X, 25, axis=0)
            Q3 = np.nanpercentile(X, 75, axis=0)
            IQR = Q3 - Q1
            self.lower_bounds_ = Q1 - self.iqr_factor * IQR
            self.upper_bounds_ = Q3 + self.iqr_factor * IQR
        elif self.method == 'zscore':
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            std = np.where(std == 0, 1, std)  # ゼロ除算防止
            self.lower_bounds_ = mean - self.zscore_threshold * std
            self.upper_bounds_ = mean + self.zscore_threshold * std
        elif self.method == 'clip':
            self.lower_bounds_ = np.nanpercentile(X, 1, axis=0)
            self.upper_bounds_ = np.nanpercentile(X, 99, axis=0)
        else:
            self.lower_bounds_ = np.full(X.shape[1], -np.inf)
            self.upper_bounds_ = np.full(X.shape[1], np.inf)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_bounds_ is None:
            return X
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class SmartPreprocessor(BaseEstimator, TransformerMixin):
    """
    変数タイプを自動検出し適切な前処理を構築するスマートプリプロセッサ
    
    Features:
    - 変数タイプの自動検出（連続/カテゴリ/バイナリ/整数カウント）
    - 複数のスケーラー/エンコーダーから選択可能
    - 欠損値処理・外れ値処理の統合
    - scikit-learn Pipeline互換
    
    Example (初心者向け - すべてデフォルト):
        >>> preprocessor = SmartPreprocessor()
        >>> X_transformed = preprocessor.fit_transform(X, y)
    
    Example (熟練者向け - カスタマイズ):
        >>> preprocessor = SmartPreprocessor(
        ...     continuous_scaler='power',
        ...     categorical_encoder='ordinal',
        ...     handle_missing='knn',
        ...     handle_outliers='zscore',
        ... )
        >>> X_transformed = preprocessor.fit_transform(X, y)
    """
    
    def __init__(
        self,
        continuous_scaler: Literal['standard', 'power', 'quantile', 'robust', 'minmax', 'none'] = 'standard',
        categorical_encoder: Literal['onehot', 'ordinal'] = 'onehot',
        handle_missing: Literal['mean', 'median', 'most_frequent', 'knn', 'zero'] = 'median',
        handle_outliers: Literal['iqr', 'zscore', 'clip', 'none'] = 'iqr',
        binary_threshold: int = 2,
        categorical_threshold: int = 20,
        passthrough_smiles: bool = True,
    ):
        """
        Args:
            continuous_scaler: 連続変数のスケーリング方法
            categorical_encoder: カテゴリ変数のエンコーディング方法
            handle_missing: 欠損値処理方法
            handle_outliers: 外れ値処理方法
            binary_threshold: これ以下のユニーク数をバイナリとみなす
            categorical_threshold: これ以下のユニーク数（整数）をカテゴリとみなす
            passthrough_smiles: SMILESカラムをパススルーするか
        """
        self.continuous_scaler = continuous_scaler
        self.categorical_encoder = categorical_encoder
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.binary_threshold = binary_threshold
        self.categorical_threshold = categorical_threshold
        self.passthrough_smiles = passthrough_smiles
        
        # 学習後の状態
        self.column_types_: Optional[ColumnTypeInfo] = None
        self.transformer_: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[List[str]] = None
        
    def detect_column_types(self, X: pd.DataFrame) -> ColumnTypeInfo:
        """
        カラムタイプを自動検出
        
        Args:
            X: 入力DataFrame
            
        Returns:
            ColumnTypeInfo: 分類されたカラム情報
        """
        info = ColumnTypeInfo()
        
        for col in X.columns:
            # SMILESカラムはスキップ（文字列型の場合のみチェック）
            if isinstance(col, str) and col.upper() in ['SMILES', 'SMILES_CANONICAL', 'MOL']:
                continue
            
            dtype = X[col].dtype
            nunique = X[col].nunique()
            
            # カテゴリ/オブジェクト型
            if dtype == 'object' or dtype.name == 'category':
                if nunique <= self.binary_threshold:
                    info.binary.append(col)
                else:
                    info.categorical.append(col)
            # 数値型
            elif np.issubdtype(dtype, np.number):
                if nunique <= self.binary_threshold:
                    info.binary.append(col)
                elif dtype == np.int64 and nunique <= self.categorical_threshold:
                    info.integer_count.append(col)
                else:
                    info.continuous.append(col)
            else:
                # 不明な型はカテゴリとして扱う
                info.categorical.append(col)
        
        logger.info(f"カラムタイプ検出: {info.summary()}")
        return info
    
    def build_pipeline(self, column_types: ColumnTypeInfo) -> ColumnTransformer:
        """
        ColumnTransformerパイプラインを構築
        
        Args:
            column_types: カラムタイプ情報
            
        Returns:
            ColumnTransformer: 構築されたトランスフォーマー
        """
        transformers = []
        
        # 連続変数パイプライン
        if column_types.continuous:
            continuous_pipeline = Pipeline([
                ('imputer', self._get_imputer()),
                ('outlier', OutlierHandler(method=self.handle_outliers)),
                ('scaler', self._get_scaler()),
            ])
            transformers.append(('continuous', continuous_pipeline, column_types.continuous))
        
        # カテゴリ変数パイプライン
        if column_types.categorical:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', self._get_encoder()),
            ])
            transformers.append(('categorical', categorical_pipeline, column_types.categorical))
        
        # バイナリ変数（パススルーまたはラベルエンコード）
        if column_types.binary:
            binary_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
            ])
            transformers.append(('binary', binary_pipeline, column_types.binary))
        
        # 整数カウント変数
        if column_types.integer_count:
            integer_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
            ])
            transformers.append(('integer_count', integer_pipeline, column_types.integer_count))
        
        return ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # 未処理カラムは除去
            verbose_feature_names_out=False,
        )
    
    def _get_imputer(self):
        """欠損値処理器を取得"""
        if self.handle_missing in IMPUTERS:
            return IMPUTERS[self.handle_missing]()
        return IMPUTERS['median']()
    
    def _get_scaler(self):
        """スケーラーを取得"""
        if self.continuous_scaler in SCALERS:
            scaler = SCALERS[self.continuous_scaler]
            if callable(scaler):
                return scaler()
            return scaler
        return StandardScaler()
    
    def _get_encoder(self):
        """エンコーダーを取得"""
        if self.categorical_encoder in ENCODERS:
            return ENCODERS[self.categorical_encoder]()
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    def fit(self, X: pd.DataFrame, y=None) -> 'SmartPreprocessor':
        """
        データに基づいて前処理パイプラインを構築・学習
        
        Args:
            X: 入力DataFrame
            y: ターゲット変数（一部のエンコーダーで使用）
            
        Returns:
            self
        """
        # カラムタイプを検出
        self.column_types_ = self.detect_column_types(X)
        
        # パイプラインを構築
        self.transformer_ = self.build_pipeline(self.column_types_)
        
        # 学習
        self.transformer_.fit(X, y)
        
        # 出力特徴量名を取得
        try:
            self.feature_names_out_ = list(self.transformer_.get_feature_names_out())
        except Exception:
            # フォールバック
            self.feature_names_out_ = None
        
        logger.info(f"前処理パイプライン学習完了")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        学習済みパイプラインで変換
        
        Args:
            X: 入力DataFrame
            
        Returns:
            pd.DataFrame: 変換後のDataFrame
        """
        if self.transformer_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        # 変換
        X_transformed = self.transformer_.transform(X)
        
        # DataFrameに変換
        if self.feature_names_out_:
            columns = self.feature_names_out_
        else:
            columns = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        return pd.DataFrame(X_transformed, columns=columns, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """fit + transform"""
        return self.fit(X, y).transform(X)
    
    def get_params_summary(self) -> Dict[str, Any]:
        """現在の設定サマリーを取得"""
        return {
            '連続変数スケーラー': self.continuous_scaler,
            'カテゴリエンコーダー': self.categorical_encoder,
            '欠損値処理': self.handle_missing,
            '外れ値処理': self.handle_outliers,
        }
    
    def get_column_types_summary(self) -> Optional[Dict[str, List[str]]]:
        """カラムタイプ分類結果を取得"""
        if self.column_types_:
            return self.column_types_.to_dict()
        return None


class PreprocessorFactory:
    """
    プリセット設定でプリプロセッサを生成するファクトリ
    
    Example:
        >>> preprocessor = PreprocessorFactory.create('robust')
        >>> X_transformed = preprocessor.fit_transform(X)
    """
    
    PRESETS = {
        'default': {
            'continuous_scaler': 'standard',
            'categorical_encoder': 'onehot',
            'handle_missing': 'median',
            'handle_outliers': 'iqr',
        },
        'robust': {
            'continuous_scaler': 'robust',
            'categorical_encoder': 'onehot',
            'handle_missing': 'median',
            'handle_outliers': 'iqr',
        },
        'normalized': {
            'continuous_scaler': 'quantile',
            'categorical_encoder': 'onehot',
            'handle_missing': 'knn',
            'handle_outliers': 'clip',
        },
        'minimal': {
            'continuous_scaler': 'none',
            'categorical_encoder': 'ordinal',
            'handle_missing': 'zero',
            'handle_outliers': 'none',
        },
        'tree_optimized': {
            # 木系モデル向け（スケーリング不要、OrdinalEncoder推奨）
            'continuous_scaler': 'none',
            'categorical_encoder': 'ordinal',
            'handle_missing': 'median',
            'handle_outliers': 'none',
        },
        'neural': {
            # ニューラルネットワーク向け
            'continuous_scaler': 'quantile',
            'categorical_encoder': 'onehot',
            'handle_missing': 'knn',
            'handle_outliers': 'clip',
        },
    }
    
    @classmethod
    def create(cls, preset: str = 'default', **overrides) -> SmartPreprocessor:
        """
        プリセットからプリプロセッサを生成
        
        Args:
            preset: プリセット名
            **overrides: 上書きパラメータ
            
        Returns:
            SmartPreprocessor: 設定済みプリプロセッサ
        """
        if preset not in cls.PRESETS:
            logger.warning(f"不明なプリセット '{preset}'、'default'を使用")
            preset = 'default'
        
        params = {**cls.PRESETS[preset], **overrides}
        return SmartPreprocessor(**params)
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """利用可能なプリセット一覧"""
        return list(cls.PRESETS.keys())
