"""
汎用プロットエンジン - ML可視化

Implements: F-008
設計思想:
- 学習曲線、予測vs実測、残差プロット
- 混同行列、相関マトリックス
- 統一されたスタイル

参考文献:
- Matplotlib/Seaborn best practices
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# スタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PlotEngine:
    """
    汎用プロットエンジン
    
    Features:
    - 学習曲線
    - 予測 vs 実測
    - 残差プロット
    - 混同行列
    - 相関マトリックス
    - 分布プロット
    """
    
    @staticmethod
    def plot_training_curves(
        metrics_history: List[Dict[str, float]],
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        学習曲線をプロット
        
        Args:
            metrics_history: エポックごとのメトリクス辞書リスト
            show: 表示するか
        """
        if not metrics_history:
            return None
        
        df = pd.DataFrame(metrics_history)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for col in df.columns:
            if 'loss' in col.lower() or 'score' in col.lower():
                sns.lineplot(data=df, x=df.index, y=col, ax=ax, label=col, marker='o')
        
        ax.set_xlabel('Epoch / Step')
        ax.set_ylabel('Value')
        ax.set_title('Training Curves')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_predicted_vs_actual(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        task_type: str = 'regression',
        show: bool = False,
    ) -> plt.Figure:
        """
        予測 vs 実測プロット
        
        Args:
            y_true: 真値
            y_pred: 予測値
            task_type: 'regression' or 'classification'
            show: 表示するか
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if task_type == 'regression':
            # 散布図
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='white', linewidth=0.5)
            
            # 45度線
            lims = [
                min(np.min(y_true), np.min(y_pred)),
                max(np.max(y_true), np.max(y_pred)),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal')
            
            # R²を表示
            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Predicted vs Actual')
            
        else:
            # 混同行列
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
        
        ax.legend(loc='lower right')
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_residuals(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        show: bool = False,
    ) -> plt.Figure:
        """
        残差プロット
        """
        residuals = np.array(y_true) - np.array(y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 残差 vs 予測
        axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='white')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # 残差の分布
        sns.histplot(residuals, kde=True, ax=axes[1])
        axes[1].set_xlabel('Residuals')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_correlation_matrix(
        df: pd.DataFrame,
        max_features: int = 30,
        method: str = 'pearson',
        show: bool = False,
    ) -> plt.Figure:
        """
        相関マトリックス
        """
        # 数値カラムのみ
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 特徴量数制限
        if len(numeric_df.columns) > max_features:
            # 分散が大きい上位N件を選択
            variances = numeric_df.var().sort_values(ascending=False)
            top_cols = variances.head(max_features).index
            numeric_df = numeric_df[top_cols]
        
        corr = numeric_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr, mask=mask, annot=False, cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_feature_distribution(
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_cols: int = 3,
        show: bool = False,
    ) -> plt.Figure:
        """
        特徴量分布プロット
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns[:12].tolist()
        
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            if feature in df.columns:
                sns.histplot(df[feature].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(feature)
        
        # 余分なaxesを非表示
        for ax in axes[n_features:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_target_distribution(
        y: Union[pd.Series, np.ndarray],
        task_type: str = 'regression',
        show: bool = False,
    ) -> plt.Figure:
        """
        ターゲット変数の分布
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if task_type == 'regression':
            sns.histplot(y, kde=True, ax=ax)
            ax.axvline(np.mean(y), color='red', linestyle='--', label=f'Mean: {np.mean(y):.2f}')
            ax.axvline(np.median(y), color='green', linestyle='--', label=f'Median: {np.median(y):.2f}')
        else:
            y_series = pd.Series(y)
            sns.countplot(x=y_series, ax=ax)
        
        ax.set_title('Target Distribution')
        ax.legend()
        
        if show:
            plt.show()
        
        return fig
    
    @staticmethod
    def plot_cv_scores(
        cv_scores: List[float],
        show: bool = False,
    ) -> plt.Figure:
        """
        クロスバリデーションスコアの可視化
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # スコア推移
        axes[0].plot(range(1, len(cv_scores) + 1), cv_scores, 'o-', markersize=10)
        axes[0].axhline(np.mean(cv_scores), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(cv_scores):.4f}')
        axes[0].fill_between(
            range(1, len(cv_scores) + 1),
            np.mean(cv_scores) - np.std(cv_scores),
            np.mean(cv_scores) + np.std(cv_scores),
            alpha=0.2, color='red',
        )
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('CV Scores by Fold')
        axes[0].legend()
        
        # スコア分布
        axes[1].boxplot(cv_scores)
        axes[1].set_ylabel('Score')
        axes[1].set_title('CV Score Distribution')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig
