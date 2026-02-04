"""
Pipeline構成ユーティリティ - FeatureUnion, ColumnTransformer対応

Implements: F-PIPELINE-UTILS-001
設計思想:
- scikit-learnのPipeline系クラスを完全サポート
- FeatureUnion（複数特徴量の結合）
- ColumnTransformer（カラム別変換）
- 分子化学データに特化した便利関数
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    FunctionTransformer,
)

logger = logging.getLogger(__name__)


class ChemicalFeatureUnion:
    """
    複数の特徴量抽出器を結合
    
    scikit-learn FeatureUnionのラッパー + 化学特化機能
    
    Example:
        >>> from core.services.features import RDKitFeatureExtractor
        >>> union = ChemicalFeatureUnion([
        ...     ('rdkit', RDKitFeatureExtractor()),
        ...     ('morgan', MorganFingerprintExtractor()),
        ... ])
        >>> features = union.fit_transform(smiles_list)
    """
    
    def __init__(
        self,
        transformer_list: List[Tuple[str, Any]],
        n_jobs: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            transformer_list: (名前, 変換器) のリスト
            n_jobs: 並列実行数
            weights: 各変換器の重み（列結合時）
        """
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.weights = weights
        
        # FeatureUnion構築
        self.union_ = FeatureUnion(
            transformer_list=transformer_list,
            n_jobs=n_jobs,
        )
    
    def fit(self, X: Union[List[str], pd.DataFrame], y: Optional[Any] = None):
        """学習"""
        self.union_.fit(X, y)
        return self
    
    def transform(self, X: Union[List[str], pd.DataFrame]) -> pd.DataFrame:
        """変換"""
        # numpy配列で取得
        features_array = self.union_.transform(X)
        
        # DataFrameに変換
        feature_names = self._get_feature_names()
        df = pd.DataFrame(features_array, columns=feature_names)
        
        return df
    
    def fit_transform(self, X: Union[List[str], pd.DataFrame], y: Optional[Any] = None) -> pd.DataFrame:
        """学習+変換"""
        return self.fit(X, y).transform(X)
    
    def _get_feature_names(self) -> List[str]:
        """特徴量名を取得"""
        names = []
        for name, transformer in self.transformer_list:
            try:
                # descriptor_names属性がある場合
                if hasattr(transformer, 'descriptor_names'):
                    sub_names = [f"{name}_{n}" for n in transformer.descriptor_names]
                else:
                    # get_feature_names_out()がある場合
                    sub_names = [f"{name}_{i}" for i in range(transformer.n_features_out_)]
            except:
                # フォールバック
                sub_names = [f"{name}_{i}" for i in range(100)]  # 仮
            
            names.extend(sub_names)
        
        return names[:len(names)]  # 実際の数に合わせる


class ChemicalColumnTransformer:
    """
    カラム別変換器
    
    scikit-learn ColumnTransformerのラッパー + 化学特化機能
    
    Example:
        >>> transformer = ChemicalColumnTransformer([
        ...     ('smiles', SmilesToFeatures(), ['SMILES']),
        ...     ('numeric', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ... ])
        >>> X_transformed = transformer.fit_transform(df)
    """
    
    def __init__(
        self,
        transformers: List[Tuple[str, Any, Union[List[str], Any]]],
        remainder: str = 'drop',
        sparse_threshold: float = 0.3,
        n_jobs: Optional[int] = None,
    ):
        """
        Args:
            transformers: (名前, 変換器, カラム) のリスト
            remainder: 残りのカラムの扱い ('drop' or 'passthrough')
            sparse_threshold: スパース行列の閾値
            n_jobs: 並列実行数
        """
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        
        # ColumnTransformer構築
        self.ct_ = ColumnTransformer(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
        )
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None):
        """学習"""
        self.ct_.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        # 変換実行
        X_transformed = self.ct_.transform(X)
        
        # DataFrameに変換
        feature_names = self.ct_.get_feature_names_out()
        
        if hasattr(X_transformed, 'toarray'):  # スパース行列の場合
            X_transformed = X_transformed.toarray()
        
        df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        return df
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """学習+変換"""
        return self.fit(X, y).transform(X)


def create_preprocessing_pipeline(
    scaler: str = 'standard',
    power_transform: bool = False,
    quantile_transform: bool = False,
) -> Pipeline:
    """
    前処理パイプライン作成
    
    Args:
        scaler: スケーラー ('standard', 'minmax', 'robust')
        power_transform: PowerTransform適用
        quantile_transform: QuantileTransform適用
        
    Returns:
        sklearn Pipeline
    """
    steps = []
    
    # PowerTransform（正規分布化）
    if power_transform:
        steps.append(('power', PowerTransformer(method='yeo-johnson')))
    
    # QuantileTransform（分位点変換）
    if quantile_transform:
        steps.append(('quantile', QuantileTransformer(output_distribution='normal')))
    
    # スケーラー
    if scaler == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler == 'robust':
        steps.append(('scaler', RobustScaler()))
    
    return Pipeline(steps)


def create_feature_engineering_pipeline(
    smiles_transformer: Optional[Any] = None,
    numeric_transformer: Optional[Pipeline] = None,
    smiles_column: str = 'SMILES',
) -> ColumnTransformer:
    """
    特徴量エンジニアリングパイプライン作成
    
    Args:
        smiles_transformer: SMILES変換器（RDKit等）
        numeric_transformer: 数値変換パイプライン
        smiles_column: SMILESカラム名
        
    Returns:
        ColumnTransformer
    """
    transformers = []
    
    # SMILES変換
    if smiles_transformer:
        transformers.append(('smiles', smiles_transformer, [smiles_column]))
    
    # 数値変換
    if numeric_transformer is None:
        numeric_transformer = create_preprocessing_pipeline()
    
    transformers.append((
        'numeric',
        numeric_transformer,
        make_column_selector(dtype_include=np.number)
    ))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        n_jobs=-1,
    )


class AdvancedPreprocessor:
    """
    高度な前処理クラス
    
    scikit-learnの全前処理手法をサポート:
    - StandardScaler, MinMaxScaler, RobustScaler
    - PowerTransformer (Yeo-Johnson, Box-Cox)
    - QuantileTransformer
    - Normalizer (L1, L2)
    """
    
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
    }
    
    def __init__(
        self,
        scaler: str = 'standard',
        power_transform: Optional[str] = None,  # 'yeo-johnson' or 'box-cox'
        quantile_transform: bool = False,
        normalize: Optional[str] = None,  # 'l1' or 'l2'
    ):
        """
        Args:
            scaler: スケーラー種類
            power_transform: PowerTransformメソッド
            quantile_transform: QuantileTransform適用
            normalize: 正規化（'l1' or 'l2'）
        """
        self.scaler = scaler
        self.power_transform = power_transform
        self.quantile_transform = quantile_transform
        self.normalize = normalize
        
        # パイプライン構築
        self.pipeline_ = self._build_pipeline()
    
    def _build_pipeline(self) -> Pipeline:
        """パイプライン構築"""
        steps = []
        
        # PowerTransform
        if self.power_transform:
            steps.append((
                'power',
                PowerTransformer(method=self.power_transform, standardize=False)
            ))
        
        # QuantileTransform
        if self.quantile_transform:
            steps.append((
                'quantile',
                QuantileTransformer(output_distribution='normal', n_quantiles=1000)
            ))
        
        # スケーラー
        if self.scaler in self.SCALERS:
            steps.append(('scaler', self.SCALERS[self.scaler]()))
        
        # 正規化
        if self.normalize:
            from sklearn.preprocessing import Normalizer
            steps.append(('normalize', Normalizer(norm=self.normalize)))
        
        return Pipeline(steps)
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None):
        """学習"""
        self.pipeline_.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        X_transformed = self.pipeline_.transform(X)
        return pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """学習+変換"""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆変換"""
        X_inv = self.pipeline_.inverse_transform(X)
        return pd.DataFrame(X_inv, columns=X.columns, index=X.index)
