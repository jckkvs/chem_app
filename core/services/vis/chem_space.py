"""
化学空間マップ可視化エンジン

Implements: F-CHEMSPACE-001
設計思想:
- UMAP/t-SNEによる次元削減可視化
- インタラクティブなプロット
- 分子構造ホバー表示
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


class ChemicalSpaceMapper:
    """
    化学空間マップ生成エンジン
    
    Features:
    - UMAP/t-SNEによる2D/3D可視化
    - ターゲット値による色分け
    - クラスタ検出
    - インタラクティブHTML生成
    
    Example:
        >>> mapper = ChemicalSpaceMapper(method='umap')
        >>> coords = mapper.fit_transform(X)
        >>> fig = mapper.plot(coords, y=target, smiles=smiles_list)
    """
    
    def __init__(
        self,
        method: Literal['umap', 'tsne'] = 'umap',
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        perplexity: float = 30.0,
        random_state: int = 42,
    ):
        """
        Args:
            method: 次元削減手法 ('umap' or 'tsne')
            n_components: 出力次元 (2 or 3)
            n_neighbors: UMAP近傍数
            min_dist: UMAP最小距離
            perplexity: t-SNE perplexity
            random_state: 乱数シード
        """
        self.method = method
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.perplexity = perplexity
        self.random_state = random_state
        
        self.reducer_ = None
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        次元削減を実行
        
        Args:
            X: 特徴量行列
            y: ターゲット（Supervised UMAPで使用）
            
        Returns:
            np.ndarray: 2D/3D座標
        """
        if self.method == 'umap':
            self.reducer_ = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state,
            )
            if y is not None:
                return self.reducer_.fit_transform(X, y=y)
            return self.reducer_.fit_transform(X)
        
        else:  # tsne
            self.reducer_ = TSNE(
                n_components=self.n_components,
                perplexity=min(self.perplexity, len(X) - 1),
                random_state=self.random_state,
            )
            return self.reducer_.fit_transform(X)
    
    def plot(
        self,
        coords: np.ndarray,
        y: Optional[np.ndarray] = None,
        smiles: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        title: str = "Chemical Space Map",
        colormap: str = "viridis",
        figsize: Tuple[int, int] = (10, 8),
        show: bool = False,
    ) -> plt.Figure:
        """
        化学空間マップをプロット
        
        Args:
            coords: 2D/3D座標
            y: ターゲット値（色分け用）
            smiles: SMILESリスト（ツールチップ用）
            labels: ラベルリスト
            title: タイトル
            colormap: カラーマップ
            figsize: フィギュアサイズ
            show: 表示するか
            
        Returns:
            plt.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if y is not None:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=y,
                cmap=colormap,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5,
            )
            plt.colorbar(scatter, ax=ax, label='Target Value')
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5,
            )
        
        ax.set_xlabel(f'{self.method.upper()} 1')
        ax.set_ylabel(f'{self.method.upper()} 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
        return fig
    
    def plot_3d(
        self,
        coords: np.ndarray,
        y: Optional[np.ndarray] = None,
        title: str = "3D Chemical Space",
        colormap: str = "viridis",
        figsize: Tuple[int, int] = (12, 10),
        show: bool = False,
    ) -> plt.Figure:
        """3D化学空間マップ"""
        if coords.shape[1] < 3:
            raise ValueError("3D plot requires n_components=3")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if y is not None:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2],
                c=y,
                cmap=colormap,
                alpha=0.7,
            )
            plt.colorbar(scatter, ax=ax, label='Target Value', shrink=0.5)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], alpha=0.7)
        
        ax.set_xlabel(f'{self.method.upper()} 1')
        ax.set_ylabel(f'{self.method.upper()} 2')
        ax.set_zlabel(f'{self.method.upper()} 3')
        ax.set_title(title)
        
        if show:
            plt.show()
        
        return fig
    
    def generate_interactive_html(
        self,
        coords: np.ndarray,
        y: Optional[np.ndarray] = None,
        smiles: Optional[List[str]] = None,
        title: str = "Chemical Space Map",
    ) -> str:
        """
        Plotlyを使ったインタラクティブHTML生成
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly required for interactive HTML")
            return "<p>Plotly required for interactive visualization</p>"
        
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'smiles': smiles if smiles else [''] * len(coords),
            'target': y if y is not None else [0] * len(coords),
        })
        
        fig = px.scatter(
            df, x='x', y='y',
            color='target',
            hover_data=['smiles'],
            title=title,
            color_continuous_scale='viridis',
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            xaxis_title=f'{self.method.upper()} 1',
            yaxis_title=f'{self.method.upper()} 2',
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def find_neighbors(
        self,
        coords: np.ndarray,
        query_idx: int,
        n_neighbors: int = 5,
    ) -> List[int]:
        """
        座標空間で近傍分子を検索
        
        Args:
            coords: 座標
            query_idx: クエリインデックス
            n_neighbors: 近傍数
            
        Returns:
            近傍インデックスリスト
        """
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(coords)
        
        distances, indices = nn.kneighbors([coords[query_idx]])
        
        # 自分自身を除く
        return [i for i in indices[0] if i != query_idx][:n_neighbors]
