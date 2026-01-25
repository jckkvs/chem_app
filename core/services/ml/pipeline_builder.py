"""
sklearn Pipeline完全統合

Implements: F-PIPELINE-001
設計思想:
- sklearn.pipelineの全機能をサポート
- 前処理→特徴選択→モデルのワークフロー自動化
- 化学ML向けのパイプライン構築支援

参考文献:
- scikit-learn pipeline documentation
- Pipeline User Guide (sklearn)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    sklearn Pipeline完全ラッパー
    
    Features:
    - Pipeline構築支援
    - FeatureUnion（並列特徴変換）
    - make_pipeline/make_union簡易作成
    - カスタムTransformer統合
    
    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> 
        >>> builder = PipelineBuilder()
        >>> pipeline = builder.build_pipeline([
        ...     ('scaler', StandardScaler()),
        ...     ('pca', PCA(n_components=10)),
        ...     ('clf', RandomForestClassifier())
        ... ])
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    
    def __init__(self):
        """初期化"""
        pass
    
    def build_pipeline(
        self,
        steps: List[Tuple[str, BaseEstimator]],
        memory: Optional[str] = None,
        verbose: bool = False,
    ) -> Pipeline:
        """
        Pipelineを構築
        
        Args:
            steps: (名前, Transformer/Estimator)のリスト
            memory: キャッシュディレクトリ
            verbose: 詳細ログ
        
        Returns:
            Pipeline
        """
        logger.info(f"Pipeline構築: {len(steps)} steps")
        
        pipeline = Pipeline(steps=steps, memory=memory, verbose=verbose)
        
        return pipeline
    
    def build_feature_union(
        self,
        transformers: List[Tuple[str, BaseEstimator]],
        n_jobs: int = -1,
        transformer_weights: Optional[Dict[str, float]] = None,
    ) -> FeatureUnion:
        """
        FeatureUnion（並列特徴変換）を構築
        
        Args:
            transformers: (名前, Transformer)のリスト
            n_jobs: 並列実行数
            transformer_weights: 各Transformerの重み
        
        Returns:
            FeatureUnion
        """
        logger.info(f"FeatureUnion構築: {len(transformers)} transformers")
        
        feature_union = FeatureUnion(
            transformer_list=transformers,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
        )
        
        return feature_union
    
    def make_simple_pipeline(
        self,
        *steps: BaseEstimator,
    ) -> Pipeline:
        """
        簡易Pipeline作成（名前自動生成）
        
        Args:
            *steps: Transformer/Estimatorの可変長引数
        
        Returns:
            Pipeline
        """
        logger.info(f"簡易Pipeline作成: {len(steps)} steps")
        
        return make_pipeline(*steps)
    
    def make_simple_union(
        self,
        *transformers: BaseEstimator,
        n_jobs: int = -1,
    ) -> FeatureUnion:
        """
        簡易FeatureUnion作成
        
        Args:
            *transformers: Transformerの可変長引数
            n_jobs: 並列実行数
        
        Returns:
            FeatureUnion
        """
        logger.info(f"簡易FeatureUnion作成: {len(transformers)} transformers")
        
        return make_union(*transformers, n_jobs=n_jobs)


class FunctionTransformer(BaseEstimator, TransformerMixin):
    """
    カスタム関数をTransformerに変換
    
    Example:
        >>> def log_transform(X):
        ...     return np.log1p(X)
        >>> 
        >>> transformer = FunctionTransformer(log_transform)
        >>> X_transformed = transformer.fit_transform(X)
    """
    
    def __init__(self, func: callable, validate: bool = False):
        """
        Args:
            func: 変換関数
            validate: 入力検証
        """
        from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer
        
        self.transformer = SKFunctionTransformer(func=func, validate=validate)
    
    def fit(self, X, y=None):
        """フィット（何もしない）"""
        self.transformer.fit(X, y)
        return self
    
    def transform(self, X):
        """変換"""
        return self.transformer.transform(X)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    DataFrameから特定カラムを選択するTransformer
    
    Example:
        >>> selector = DataFrameSelector(['col1', 'col2', 'col3'])
        >>> X_selected = selector.fit_transform(df)
    """
    
    def __init__(self, columns: List[str]):
        """
        Args:
            columns: 選択するカラム名
        """
        self.columns = columns
    
    def fit(self, X, y=None):
        """フィット（何もしない）"""
        return self
    
    def transform(self, X):
        """変換"""
        if isinstance(X, pd.DataFrame):
            return X[self.columns].values
        else:
            return X


# =============================================================================
# ヘルパー関数
# =============================================================================

def create_preprocessing_pipeline(
    scaler: Optional[str] = 'standard',
    feature_selector: Optional[str] = None,
    n_features: Optional[int] = None,
) -> Pipeline:
    """
    前処理パイプラインを自動作成
    
    Args:
        scaler: スケーラー種類（'standard', 'minmax', 'robust'）
        feature_selector: 特徴選択手法（'k_best', 'percentile'）
        n_features: 選択する特徴数
    
    Returns:
        Pipeline
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    
    steps = []
    
    # スケーラー追加
    if scaler == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler == 'robust':
        steps.append(('scaler', RobustScaler()))
    
    # 特徴選択追加
    if feature_selector == 'k_best' and n_features:
        from sklearn.feature_selection import SelectKBest, f_regression
        steps.append(('selector', SelectKBest(f_regression, k=n_features)))
    elif feature_selector == 'percentile' and n_features:
        from sklearn.feature_selection import SelectPercentile, f_regression
        steps.append(('selector', SelectPercentile(f_regression, percentile=n_features)))
    
    return Pipeline(steps=steps) if steps else None


def create_ml_pipeline(
    preprocessor: Optional[BaseEstimator] = None,
    feature_selector: Optional[BaseEstimator] = None,
    model: Optional[BaseEstimator] = None,
) -> Pipeline:
    """
    完全なMLパイプラインを作成
    
    Args:
        preprocessor: 前処理Transformer
        feature_selector: 特徴選択Transformer
        model: モデル
    
    Returns:
        Pipeline
    """
    steps = []
    
    if preprocessor is not None:
        steps.append(('preprocessor', preprocessor))
    
    if feature_selector is not None:
        steps.append(('feature_selector', feature_selector))
    
    if model is not None:
        steps.append(('model', model))
    
    if not steps:
        raise ValueError("At least one step must be provided")
    
    return Pipeline(steps=steps)


def create_feature_engineering_pipeline(
    transformers: List[Tuple[str, BaseEstimator]],
    final_estimator: Optional[BaseEstimator] = None,
) -> Union[FeatureUnion, Pipeline]:
    """
    特徴量エンジニアリングパイプラインを作成（並列変換）
    
    Args:
        transformers: (名前, Transformer)のリスト
        final_estimator: 最終的なEstimator（オプション）
    
    Returns:
        FeatureUnion or Pipeline
    """
    feature_union = FeatureUnion(transformer_list=transformers)
    
    if final_estimator is not None:
        pipeline = Pipeline([
            ('features', feature_union),
            ('estimator', final_estimator)
        ])
        return pipeline
    
    return feature_union


def get_pipeline_steps(pipeline: Pipeline) -> List[Tuple[str, BaseEstimator]]:
    """
    Pipelineのステップを取得
    
    Args:
        pipeline: Pipeline
    
    Returns:
        List[(名前, Estimator)]
    """
    return pipeline.steps


def get_pipeline_params(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Pipelineのパラメータを取得
    
    Args:
        pipeline: Pipeline
    
    Returns:
        Dict: パラメータ
    """
    return pipeline.get_params()


def set_pipeline_params(pipeline: Pipeline, **params) -> Pipeline:
    """
    Pipelineのパラメータを設定
    
    Args:
        pipeline: Pipeline
        **params: パラメータ
    
    Returns:
        Pipeline
    """
    pipeline.set_params(**params)
    return pipeline
