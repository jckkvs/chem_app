"""
Partial Dependence Plot エンジン

Implements: F-010
設計思想:
- scikit-learn PartialDependenceDisplay利用
- 1D/2D PDP対応
- ICE (Individual Conditional Expectation) プロット対応
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

logger = logging.getLogger(__name__)


class PDPEngine:
    """
    Partial Dependence Plot (PDP) エンジン
    
    Features:
    - 1D PDP: 単一特徴量の効果
    - 2D PDP: 特徴量相互作用
    - ICE Plot: 個別サンプルの軌跡
    
    Example:
        >>> engine = PDPEngine()
        >>> fig = engine.plot_pdp(model, X, ['feature1', 'feature2'])
    """
    
    def __init__(self, grid_resolution: int = 50):
        """
        Args:
            grid_resolution: グリッドの解像度
        """
        self.grid_resolution = grid_resolution
    
    def plot_pdp(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        features: Union[List[str], List[int]],
        kind: str = 'average',  # 'average', 'individual', 'both'
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Partial Dependence Plotを生成
        
        Args:
            model: 訓練済みモデル
            X: 特徴量DataFrame
            features: プロットする特徴量リスト
            kind: 'average'=PDP, 'individual'=ICE, 'both'=両方
            show: 表示するか
            
        Returns:
            plt.Figure: 生成された図（エラー時None）
        """
        # 有効な特徴量のみ抽出
        valid_features = [f for f in features if f in X.columns]
        
        if not valid_features:
            logger.warning("有効な特徴量がありません")
            return None
        
        try:
            # サブサンプリング（大規模データ対策）
            if len(X) > 500:
                X_sample = X.sample(n=500, random_state=42)
            else:
                X_sample = X
            
            n_features = len(valid_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(
                n_rows, n_cols, 
                figsize=(5 * n_cols, 4 * n_rows),
                squeeze=False,
            )
            axes = axes.flatten()
            
            PartialDependenceDisplay.from_estimator(
                model, X_sample, valid_features,
                kind=kind,
                grid_resolution=self.grid_resolution,
                ax=axes[:n_features],
            )
            
            # 余分なaxesを非表示
            for ax in axes[n_features:]:
                ax.set_visible(False)
            
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"PDP生成エラー: {e}")
            return None
    
    def plot_2d_pdp(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_pair: Tuple[str, str],
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        2D Partial Dependence Plot（相互作用可視化）
        
        Args:
            model: 訓練済みモデル
            X: 特徴量DataFrame
            feature_pair: 特徴量ペア
            show: 表示するか
        """
        if feature_pair[0] not in X.columns or feature_pair[1] not in X.columns:
            logger.warning(f"特徴量が見つかりません: {feature_pair}")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            PartialDependenceDisplay.from_estimator(
                model, X, [feature_pair],
                kind='average',
                grid_resolution=self.grid_resolution,
                ax=ax,
            )
            
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"2D PDP生成エラー: {e}")
            return None
    
    def plot_ice(
        self,
        model: Any,
        X: pd.DataFrame,
        feature: str,
        n_samples: int = 50,
        show: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Individual Conditional Expectation (ICE) Plot
        
        各サンプルの予測変化を個別に可視化
        """
        if feature not in X.columns:
            logger.warning(f"特徴量が見つかりません: {feature}")
            return None
        
        try:
            # サンプリング
            if len(X) > n_samples:
                X_sample = X.sample(n=n_samples, random_state=42)
            else:
                X_sample = X
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            PartialDependenceDisplay.from_estimator(
                model, X_sample, [feature],
                kind='both',  # PDP + ICE
                grid_resolution=self.grid_resolution,
                ax=ax,
            )
            
            ax.set_title(f'ICE Plot: {feature}')
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return fig
            
        except Exception as e:
            logger.error(f"ICE生成エラー: {e}")
            return None
    
    def get_partial_dependence_values(
        self,
        model: Any,
        X: pd.DataFrame,
        feature: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Partial Dependence値を数値として取得
        
        Returns:
            Tuple[grid_values, pd_values]: グリッド値とPD値
        """
        if feature not in X.columns:
            return None
        
        try:
            result = partial_dependence(
                model, X, [feature],
                grid_resolution=self.grid_resolution,
            )
            
            return result['grid_values'][0], result['average'][0]
            
        except Exception as e:
            logger.error(f"PD値取得エラー: {e}")
            return None
