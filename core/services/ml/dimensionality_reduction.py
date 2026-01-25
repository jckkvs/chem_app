"""
sklearn Dimensionality Reduction完全統合

Implements: F-DIM-REDUCTION-001
設計思想:
- sklearn.decomposition + sklearn.manifoldの全手法をサポート
- 線形/非線形次元削減
- 化学特徴量の可視化・圧縮

参考文献:
- scikit-learn decomposition documentation
- Dimensionality Reduction: A Comparative Review (van der Maaten et al., 2009)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """
    sklearn Dimensionality Reduction完全ラッパー
    
    Features:
    - Linear Methods: PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD
    - Matrix Factorization: NMF, FactorAnalysis, LatentDirichletAllocation
    - Manifold Learning: TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
    - Dictionary Learning: DictionaryLearning, MiniBatchDictionaryLearning
    - 化学特徴量向けの最適化
    
    Example:
        >>> reducer = DimensionalityReducer(method='pca', n_components=2)
        >>> X_reduced = reducer.fit_transform(X)
        >>> print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
    """
    
    def __init__(
        self,
        method: str = 'pca',
        n_components: Union[int, float, str] = 2,
        **params
    ):
        """
        Args:
            method: 次元削減手法
            n_components: 削減後の次元数（または説明分散比）
            **params: 各手法固有のパラメータ
        """
        self.method = method
        self.n_components = n_components
        self.params = params
        
        self.reducer_: Optional[BaseEstimator] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
    
    def _create_reducer(self) -> BaseEstimator:
        """手法に応じたreducerを作成"""
        
        # ===== Linear Methods =====
        if self.method == 'pca':
            from sklearn.decomposition import PCA
            return PCA(n_components=self.n_components, **self.params)
        
        elif self.method == 'incremental_pca':
            from sklearn.decomposition import IncrementalPCA
            batch_size = self.params.pop('batch_size', None)
            return IncrementalPCA(
                n_components=self.n_components,
                batch_size=batch_size,
                **self.params
            )
        
        elif self.method == 'kernel_pca':
            from sklearn.decomposition import KernelPCA
            kernel = self.params.pop('kernel', 'rbf')
            return KernelPCA(
                n_components=self.n_components,
                kernel=kernel,
                **self.params
            )
        
        elif self.method == 'sparse_pca':
            from sklearn.decomposition import SparsePCA
            alpha = self.params.pop('alpha', 1.0)
            return SparsePCA(
                n_components=self.n_components,
                alpha=alpha,
                **self.params
            )
        
        elif self.method == 'truncated_svd':
            from sklearn.decomposition import TruncatedSVD
            return TruncatedSVD(n_components=self.n_components, **self.params)
        
        # ===== Matrix Factorization =====
        elif self.method == 'nmf':
            from sklearn.decomposition import NMF
            init = self.params.pop('init', 'nndsvda')
            return NMF(n_components=self.n_components, init=init, **self.params)
        
        elif self.method == 'factor_analysis':
            from sklearn.decomposition import FactorAnalysis
            return FactorAnalysis(n_components=self.n_components, **self.params)
        
        elif self.method == 'lda':
            # Latent Dirichlet Allocation
            from sklearn.decomposition import LatentDirichletAllocation
            return LatentDirichletAllocation(
                n_components=self.n_components, **self.params
            )
        
        # ===== Manifold Learning =====
        elif self.method == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = self.params.pop('perplexity', 30.0)
            return TSNE(
                n_components=self.n_components,
                perplexity=perplexity,
                **self.params
            )
        
        elif self.method == 'isomap':
            from sklearn.manifold import Isomap
            n_neighbors = self.params.pop('n_neighbors', 5)
            return Isomap(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                **self.params
            )
        
        elif self.method == 'lle':
            # Locally Linear Embedding
            from sklearn.manifold import LocallyLinearEmbedding
            n_neighbors = self.params.pop('n_neighbors', 5)
            method = self.params.pop('lle_method', 'standard')
            return LocallyLinearEmbedding(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                method=method,
                **self.params
            )
        
        elif self.method == 'mds':
            # Multidimensional Scaling
            from sklearn.manifold import MDS
            metric = self.params.pop('metric', True)
            return MDS(
                n_components=self.n_components,
                metric=metric,
                **self.params
            )
        
        elif self.method == 'spectral_embedding':
            from sklearn.manifold import SpectralEmbedding
            n_neighbors = self.params.pop('n_neighbors', None)
            return SpectralEmbedding(
                n_components=self.n_components,
                n_neighbors=n_neighbors,
                **self.params
            )
        
        # ===== Dictionary Learning =====
        elif self.method == 'dictionary_learning':
            from sklearn.decomposition import DictionaryLearning
            alpha = self.params.pop('alpha', 1.0)
            return DictionaryLearning(
                n_components=self.n_components,
                alpha=alpha,
                **self.params
            )
        
        elif self.method == 'minibatch_dictionary_learning':
            from sklearn.decomposition import MiniBatchDictionaryLearning
            alpha = self.params.pop('alpha', 1.0)
            batch_size = self.params.pop('batch_size', 3)
            return MiniBatchDictionaryLearning(
                n_components=self.n_components,
                alpha=alpha,
                batch_size=batch_size,
                **self.params
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DimensionalityReducer':
        """
        次元削減をフィット
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット（一部の手法で使用）
        
        Returns:
            self
        """
        self.reducer_ = self._create_reducer()
        
        logger.info(
            f"Dimensionality reduction開始: "
            f"method={self.method}, {X.shape[1]} → {self.n_components} dims"
        )
        
        # フィット
        self.reducer_.fit(X.values, y.values if y is not None else None)
        
        # 説明分散比を取得（可能な場合）
        if hasattr(self.reducer_, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_
            total_variance = np.sum(self.explained_variance_ratio_)
            logger.info(f"Explained variance ratio: {total_variance:.4f}")
        
        logger.info("Dimensionality reduction完了")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        次元削減を適用
        
        Args:
            X: 特徴量DataFrame
        
        Returns:
            削減後のDataFrame
        """
        if self.reducer_ is None:
            raise ValueError("Reducer not fitted. Call fit() first.")
        
        X_reduced = self.reducer_.transform(X.values)
        
        # カラム名生成
        if isinstance(self.n_components, int):
            n_cols = self.n_components
        else:
            n_cols = X_reduced.shape[1]
        
        columns = [f'{self.method}_{i+1}' for i in range(n_cols)]
        
        return pd.DataFrame(
            X_reduced,
            columns=columns,
            index=X.index,
        )
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """フィット＆変換"""
        self.reducer_ = self._create_reducer()
        
        logger.info(
            f"Dimensionality reduction開始: "
            f"method={self.method}, {X.shape[1]} → {self.n_components} dims"
        )
        
        # transformメソッドを持たない手法（TSNE/MDS等）はfit_transformのみ
        if hasattr(self.reducer_, 'fit_transform'):
            X_reduced = self.reducer_.fit_transform(
                X.values, y.values if y is not None else None
            )
        else:
            # 通常のfit + transform
            self.reducer_.fit(X.values, y.values if y is not None else None)
            X_reduced = self.reducer_.transform(X.values)
        
        # 説明分散比を取得（可能な場合）
        if hasattr(self.reducer_, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_
            total_variance = np.sum(self.explained_variance_ratio_)
            logger.info(f"Explained variance ratio: {total_variance:.4f}")
        
        # カラム名生成
        if isinstance(self.n_components, int):
            n_cols = self.n_components
        else:
            n_cols = X_reduced.shape[1]
        
        columns = [f'{self.method}_{i+1}' for i in range(n_cols)]
        
        logger.info("Dimensionality reduction完了")
        
        return pd.DataFrame(
            X_reduced,
            columns=columns,
            index=X.index,
        )
    
    def inverse_transform(self, X_reduced: pd.DataFrame) -> pd.DataFrame:
        """逆変換（可能な場合）"""
        if self.reducer_ is None:
            raise ValueError("Reducer not fitted.")
        
        if not hasattr(self.reducer_, 'inverse_transform'):
            raise ValueError(f"{self.method} does not support inverse_transform")
        
        X_reconstructed = self.reducer_.inverse_transform(X_reduced.values)
        
        return pd.DataFrame(X_reconstructed, index=X_reduced.index)
    
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """説明分散比を取得"""
        return self.explained_variance_ratio_


# =============================================================================
# ヘルパー関数
# =============================================================================

def auto_reduce_dimensions(
    X: pd.DataFrame,
    method: str = 'pca',
    target_variance: float = 0.95,
    max_components: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    自動次元削減（目標説明分散比を達成）
    
    Args:
        X: 特徴量DataFrame
        method: 次元削減手法
        target_variance: 目標説明分散比
        max_components: 最大コンポーネント数
    
    Returns:
        Tuple[削減後DataFrame, 情報Dict]
    """
    reducer = DimensionalityReducer(
        method=method,
        n_components=max_components or min(X.shape[0], X.shape[1]),
    )
    
    X_reduced = reducer.fit_transform(X)
    
    # 目標分散を達成する最小コンポーネント数を見つける
    if reducer.explained_variance_ratio_ is not None:
        cumsum_variance = np.cumsum(reducer.explained_variance_ratio_)
        n_components_needed = np.argmax(cumsum_variance >= target_variance) + 1
        
        # 必要なコンポーネントのみ保持
        X_reduced = X_reduced.iloc[:, :n_components_needed]
        
        info = {
            'method': method,
            'n_components': n_components_needed,
            'explained_variance': cumsum_variance[n_components_needed - 1],
            'target_variance': target_variance,
        }
    else:
        info = {
            'method': method,
            'n_components': X_reduced.shape[1],
            'explained_variance': None,
        }
    
    return X_reduced, info


def visualize_2d(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    method: str = 'tsne',
    **kwargs
) -> pd.DataFrame:
    """
    2次元可視化用の次元削減
    
    Args:
        X: 特徴量DataFrame
        y: ターゲット（色分け用）
        method: 次元削減手法
        **kwargs: 追加パラメータ
    
    Returns:
        2次元DataFrame（x, yカラム付き）
    """
    reducer = DimensionalityReducer(method=method, n_components=2, **kwargs)
    X_2d = reducer.fit_transform(X, y)
    
    X_2d.columns = ['x', 'y']
    
    return X_2d
