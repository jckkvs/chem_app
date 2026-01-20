"""
拡張データ可視化ツール

Implements: F-VISEXT-001
設計思想:
- 分子化学空間の可視化
- 予測結果の詳細プロット
- インタラクティブなダッシュボード

機能拡張:
- Plotlyベースのインタラクティブ可視化
- 化学空間マッピング（t-SNE/UMAP）
- 信頼区間付き予測プロット
- 活性クリフ可視化
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any, Tuple
import io
import base64

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChemicalSpaceVisualizer:
    """
    化学空間可視化
    
    分子を2D空間にマッピングして可視化。
    
    Usage:
        vis = ChemicalSpaceVisualizer()
        fig = vis.create_scatter(smiles_list, values)
    """
    
    def __init__(
        self,
        method: str = 'tsne',
        n_components: int = 2,
        random_state: int = 42,
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
    
    def _compute_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """SMILESからフィンガープリントを計算"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            logger.warning("RDKit not available")
            return np.random.randn(len(smiles_list), 128)
        
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(1024))
        
        return np.array(fps)
    
    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """次元削減"""
        if self.method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(
                    n_components=self.n_components,
                    random_state=self.random_state,
                )
                return reducer.fit_transform(X)
            except ImportError:
                logger.warning("UMAP not available, using t-SNE")
        
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            perplexity=min(30, max(5, len(X) // 4)),
        )
        return reducer.fit_transform(X)
    
    def compute_coordinates(self, smiles_list: List[str]) -> np.ndarray:
        """SMILESから2D座標を計算"""
        fps = self._compute_fingerprints(smiles_list)
        return self._reduce_dimensions(fps)
    
    def create_scatter_plotly(
        self,
        smiles_list: List[str],
        values: np.ndarray = None,
        labels: List[str] = None,
        title: str = "Chemical Space",
    ):
        """Plotlyで散布図を作成"""
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError("Plotly is required")
        
        coords = self.compute_coordinates(smiles_list)
        
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'smiles': smiles_list,
        })
        
        if values is not None:
            df['value'] = values
            fig = px.scatter(df, x='x', y='y', color='value',
                           hover_data=['smiles'], title=title)
        else:
            fig = px.scatter(df, x='x', y='y',
                           hover_data=['smiles'], title=title)
        
        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
        )
        
        return fig
    
    def create_scatter_matplotlib(
        self,
        smiles_list: List[str],
        values: np.ndarray = None,
        title: str = "Chemical Space",
    ):
        """Matplotlibで散布図を作成"""
        import matplotlib.pyplot as plt
        
        coords = self.compute_coordinates(smiles_list)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if values is not None:
            scatter = ax.scatter(coords[:, 0], coords[:, 1],
                               c=values, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Value')
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
        
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(title)
        
        return fig


class PredictionResultVisualizer:
    """
    予測結果可視化
    
    Usage:
        vis = PredictionResultVisualizer()
        fig = vis.scatter_with_error(y_true, y_pred, errors)
    """
    
    def scatter_with_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        errors: np.ndarray = None,
        title: str = "Prediction Results",
    ):
        """誤差付き散布図"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return self._scatter_matplotlib(y_true, y_pred, title)
        
        fig = go.Figure()
        
        # 散布図
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode='markers',
            marker=dict(
                size=8,
                color=errors if errors is not None else 'blue',
                colorscale='Reds' if errors is not None else None,
                showscale=errors is not None,
            ),
            name='Predictions',
        ))
        
        # 理想線
        line_range = [min(y_true.min(), y_pred.min()),
                     max(y_true.max(), y_pred.max())]
        fig.add_trace(go.Scatter(
            x=line_range, y=line_range,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ideal',
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Actual",
            yaxis_title="Predicted",
        )
        
        return fig
    
    def _scatter_matplotlib(self, y_true, y_pred, title):
        """Matplotlibバージョン"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.7)
        
        lims = [min(y_true.min(), y_pred.min()),
               max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--')
        
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        
        return fig
    
    def interval_plot(
        self,
        indices: List[Any],
        means: np.ndarray,
        lowers: np.ndarray,
        uppers: np.ndarray,
        actuals: np.ndarray = None,
        title: str = "Predictions with Confidence Intervals",
    ):
        """信頼区間プロット"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required")
        
        fig = go.Figure()
        
        # 信頼区間
        fig.add_trace(go.Scatter(
            x=list(indices) + list(indices)[::-1],
            y=list(uppers) + list(lowers)[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
        ))
        
        # 予測値
        fig.add_trace(go.Scatter(
            x=indices, y=means,
            mode='lines+markers',
            name='Prediction',
        ))
        
        # 実測値
        if actuals is not None:
            fig.add_trace(go.Scatter(
                x=indices, y=actuals,
                mode='markers',
                marker=dict(color='red', size=10),
                name='Actual',
            ))
        
        fig.update_layout(title=title)
        
        return fig


class ActivityCliffVisualizer:
    """
    活性クリフ可視化
    
    構造的に類似だが活性が大きく異なる分子ペアを可視化。
    """
    
    def plot_cliffs(
        self,
        cliff_data: pd.DataFrame,
        title: str = "Activity Cliffs",
    ):
        """活性クリフをネットワーク図で可視化"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly is required")
        
        fig = go.Figure()
        
        # ノード（分子）
        smiles_set = set(cliff_data['smiles_1'].tolist() + 
                        cliff_data['smiles_2'].tolist())
        smiles_list = list(smiles_set)
        
        # 簡易レイアウト（円形）
        n = len(smiles_list)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x_pos = {smi: np.cos(ang) for smi, ang in zip(smiles_list, angles)}
        y_pos = {smi: np.sin(ang) for smi, ang in zip(smiles_list, angles)}
        
        # エッジ
        for _, row in cliff_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[x_pos[row['smiles_1']], x_pos[row['smiles_2']]],
                y=[y_pos[row['smiles_1']], y_pos[row['smiles_2']]],
                mode='lines',
                line=dict(width=row.get('activity_ratio', 1)),
                hoverinfo='none',
            ))
        
        # ノード
        fig.add_trace(go.Scatter(
            x=[x_pos[smi] for smi in smiles_list],
            y=[y_pos[smi] for smi in smiles_list],
            mode='markers+text',
            text=[smi[:10] for smi in smiles_list],
            textposition='top center',
            marker=dict(size=15),
            name='Molecules',
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
        )
        
        return fig


class MoleculeGridVisualizer:
    """分子グリッド可視化"""
    
    def create_grid(
        self,
        smiles_list: List[str],
        values: List[float] = None,
        n_cols: int = 4,
        size: Tuple[int, int] = (200, 150),
    ) -> str:
        """分子グリッドをHTML形式で返す"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
        except ImportError:
            return "<p>RDKit not available</p>"
        
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        legends = [f"{smi[:20]}" for smi in smiles_list]
        
        if values is not None:
            legends = [f"{smi[:10]}: {v:.2f}" 
                      for smi, v in zip(smiles_list, values)]
        
        valid_mols = [(m, l) for m, l in zip(mols, legends) if m is not None]
        if not valid_mols:
            return "<p>No valid molecules</p>"
        
        mols, legends = zip(*valid_mols)
        
        img = Draw.MolsToGridImage(
            list(mols),
            molsPerRow=n_cols,
            subImgSize=size,
            legends=list(legends),
        )
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{b64}" />'


def create_interactive_report(
    smiles_list: List[str],
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    feature_importances: Dict[str, float] = None,
    output_path: str = "report.html",
) -> str:
    """
    インタラクティブHTMLレポートを生成
    
    Args:
        smiles_list: SMILESリスト
        y_true: 実測値
        y_pred: 予測値
        feature_importances: 特徴量重要度
        output_path: 出力パス
        
    Returns:
        HTMLファイルパス
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Chemical ML Report</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>body { font-family: Arial; margin: 20px; }</style>",
        "</head><body>",
        "<h1>Chemical ML Report</h1>",
    ]
    
    # 分子グリッド
    grid_vis = MoleculeGridVisualizer()
    html_parts.append("<h2>Molecules</h2>")
    html_parts.append(grid_vis.create_grid(smiles_list[:20]))
    
    # 予測結果
    if y_true is not None and y_pred is not None:
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        html_parts.append("<h2>Model Performance</h2>")
        html_parts.append(f"<p>R²: {r2:.4f}, MAE: {mae:.4f}</p>")
    
    html_parts.append("</body></html>")
    
    html = "\n".join(html_parts)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path
