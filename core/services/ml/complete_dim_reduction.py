"""
次元削減の完全実装 - 教科書レベル全対応

Implements: F-DIM-REDUCTION-COMPLETE-001
設計思想:
- PCA, t-SNE, UMAP（既存）
- ICA, NMF, LDA, Factor Analysis
- Isomap, LLE, MDS, Spectral Embedding
- すべてを統一インターフェースで提供
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    KernelPCA,
    SparsePCA,
    TruncatedSVD,
    FastICA,
    NMF,
    FactorAnalysis,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import (
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    MDS,
    SpectralEmbedding,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CompleteDimensionalityReducer:
    """
    完全な次元削減クラス
    
    サポート手法（教科書レベル全対応）:
    
    1. Linear Methods:
       - PCA (Principal Component Analysis)
       - IncrementalPCA (大規模データ用)
       - KernelPCA (非線形)
       - SparsePCA (スパース)
       - TruncatedSVD (SVD)
       - ICA (Independent Component Analysis)
       - NMF (Non-negative Matrix Factorization)
       - LDA (Linear Discriminant Analysis)
       - Factor Analysis
    
    2. Manifold Learning:
       - t-SNE
       - UMAP（別モジュール）
       - Isomap
       - LLE (Locally Linear Embedding)
       - MDS (Multidimensional Scaling)
       - Spectral Embedding
    
    Example:
        >>> # PCA
        >>> reducer = CompleteDimensionalityReducer('pca', n_components=10)
        >>> X_reduced = reducer.fit_transform(X)
        
        >>> # t-SNE
        >>> reducer = CompleteDimensionalityReducer('tsne', n_components=2, perplexity=30)
        >>> X_2d = reducer.fit_transform(X)
    """
    
    METHODS = {
        # Linear
        'pca': PCA,
        'incremental_pca': IncrementalPCA,
        'kernel_pca': KernelPCA,
        'sparse_pca': SparsePCA,
        'truncated_svd': TruncatedSVD,
        'ica': FastICA,
        'nmf': NMF,
        'lda': LinearDiscriminantAnalysis,
        'factor_analysis': FactorAnalysis,
        
        # Manifold
        'tsne': TSNE,
        'isomap': Isomap,
        'lle': LocallyLinearEmbedding,
        'mds': MDS,
        'spectral': SpectralEmbedding,
    }
    
    def __init__(
        self,
        method: str = 'pca',
        n_components: int = 2,
        auto_scale: bool = True,
        **kwargs
    ):
        """
        Args:
            method: 次元削減手法
            n_components: 次元数
            auto_scale: 自動スケーリング
            **kwargs: 各手法固有のパラメータ
        """
        self.method = method
        self.n_components = n_components
        self.auto_scale = auto_scale
        self.kwargs = kwargs
        
        self.scaler_ = None
        self.reducer_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """学習"""
        # スケーリング
        if self.auto_scale:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Reducer作成
        self.reducer_ = self._create_reducer(y)
        
        # 学習（yが必要な手法）
        if self.method == 'lda' and y is not None:
            self.reducer_.fit(X_scaled, y)
        else:
            self.reducer_.fit(X_scaled)
        
        # 分散寄与率（PCA系のみ）
        if hasattr(self.reducer_, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        if self.reducer_ is None:
            raise ValueError("fit()を先に実行してください")
        
        # スケーリング
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X.values
        
        # 変換
        X_reduced = self.reducer_.transform(X_scaled)
        
        # DataFrame化
        columns = [f"{self.method.upper()}_{i}" for i in range(X_reduced.shape[1])]
        return pd.DataFrame(X_reduced, columns=columns, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """学習+変換"""
        # スケーリング
        if self.auto_scale:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Reducer作成
        self.reducer_ = self._create_reducer(y)
        
        # fit_transform（yが必要な手法）
        if self.method == 'lda' and y is not None:
            X_reduced = self.reducer_.fit_transform(X_scaled, y)
        else:
            X_reduced = self.reducer_.fit_transform(X_scaled)
        
        # 分散寄与率
        if hasattr(self.reducer_, 'explained_variance_ratio_'):
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_
        
        # DataFrame化
        columns = [f"{self.method.upper()}_{i}" for i in range(X_reduced.shape[1])]
        return pd.DataFrame(X_reduced, columns=columns, index=X.index)
    
    def _create_reducer(self, y: Optional[pd.Series] = None):
        """Reducer作成"""
        if self.method not in self.METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        
        ReducerClass = self.METHODS[self.method]
        
        # パラメータ設定
        params = {'n_components': self.n_components}
        params.update(self.kwargs)
        
        # 手法ごとの特殊処理
        if self.method == 'tsne':
            params.setdefault('perplexity', 30)
            params.setdefault('learning_rate', 200)
            params.setdefault('n_iter', 1000)
            params.setdefault('random_state', 42)
        
        elif self.method == 'kernel_pca':
            params.setdefault('kernel', 'rbf')
            params.setdefault('gamma', None)
        
        elif self.method == 'ica':
            params.setdefault('max_iter', 200)
            params.setdefault('random_state', 42)
        
        elif self.method == 'nmf':
            params.setdefault('init', 'nndsvda')
            params.setdefault('max_iter', 200)
            params.setdefault('random_state', 42)
        
        elif self.method == 'lle':
            params.setdefault('n_neighbors', 5)
            params.setdefault('method', 'standard')
        
        elif self.method == 'isomap':
            params.setdefault('n_neighbors', 5)
        
        elif self.method == 'lda':
            # LDAはn_componentsに制約あり
            if y is not None:
                max_components = len(np.unique(y)) - 1
                params['n_components'] = min(self.n_components, max_components)
        
        return ReducerClass(**params)
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """分散寄与率取得（PCA系のみ）"""
        return self.explained_variance_ratio_
    
    def plot_explained_variance(self, savepath: Optional[str] = None):
        """分散寄与率プロット（PCA系のみ）"""
        if self.explained_variance_ratio_ is None:
            logger.warning("分散寄与率が計算されていません（PCA系のみ利用可能）")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 個別寄与率
        ax1.bar(range(1, len(self.explained_variance_ratio_) + 1), self.explained_variance_ratio_)
        ax1.set_xlabel('主成分')
        ax1.set_ylabel('分散寄与率')
        ax1.set_title('各主成分の分散寄与率')
        ax1.grid(True, alpha=0.3)
        
        # 累積寄与率
        cumsum = np.cumsum(self.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
        ax2.set_xlabel('主成分数')
        ax2.set_ylabel('累積寄与率')
        ax2.set_title('累積寄与率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if savepath:
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# 便利関数
def quick_pca(X: pd.DataFrame, n_components: int = 10, plot: bool = True) -> pd.DataFrame:
    """PCAクイック実行"""
    reducer = CompleteDimensionalityReducer('pca', n_components=n_components)
    X_reduced = reducer.fit_transform(X)
    
    if plot:
        reducer.plot_explained_variance()
    
    return X_reduced


def quick_tsne(X: pd.DataFrame, n_components: int = 2, perplexity: int = 30) -> pd.DataFrame:
    """t-SNEクイック実行"""
    reducer = CompleteDimensionalityReducer('tsne', n_components=n_components, perplexity=perplexity)
    return reducer.fit_transform(X)


def quick_lda(X: pd.DataFrame, y: pd.Series, n_components: int = 2) -> pd.DataFrame:
    """LDAクイック実行"""
    reducer = CompleteDimensionalityReducer('lda', n_components=n_components)
    return reducer.fit_transform(X, y)
