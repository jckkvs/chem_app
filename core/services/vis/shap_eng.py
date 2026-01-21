"""
SHAP説明性エンジン

Implements: F-009
設計思想:
- TreeSHAP/KernelSHAPの自動選択
- 複数可視化タイプ対応
- 分類タスク対応

参考文献:
- A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)
- DOI: 10.48550/arXiv.1705.07874
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class SHAPEngine:
    """
    SHAP (SHapley Additive exPlanations) エンジン
    
    Features:
    - TreeExplainer（木系モデル用、高速）
    - KernelExplainer（汎用、低速）
    - Summary Plot, Dependence Plot, Force Plot
    
    Example:
        >>> engine = SHAPEngine()
        >>> shap_values, explainer = engine.explain(model, X)
        >>> fig = engine.plot_summary(shap_values, X)
    """
    
    def __init__(self, max_samples: int = 100):
        """
        Args:
            max_samples: 計算に使用する最大サンプル数
        """
        self.max_samples = max_samples
    
    def explain(
        self, 
        model: Any, 
        X: pd.DataFrame,
        is_classifier: bool = False,
    ) -> Tuple[np.ndarray, shap.Explainer]:
        """
        SHAP値を計算
        
        Args:
            model: 訓練済みモデル
            X: 特徴量DataFrame
            is_classifier: 分類タスクかどうか
            
        Returns:
            Tuple[np.ndarray, Explainer]: SHAP値と説明器
        """
        # サンプリング
        if len(X) > self.max_samples:
            X_sample = X.sample(n=self.max_samples, random_state=42)
        else:
            X_sample = X
        
        # 説明器の選択
        explainer = self._get_explainer(model, X_sample)
        
        # SHAP値計算
        try:
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            logger.warning(f"shap_values失敗、__call__を試行: {e}")
            explanation = explainer(X_sample)
            shap_values = explanation.values
        
        # 分類タスクの場合、正クラスのSHAP値のみ返す
        if is_classifier and isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        return shap_values, explainer
    
    def _get_explainer(
        self, 
        model: Any, 
        X: pd.DataFrame
    ) -> shap.Explainer:
        """適切な説明器を選択"""
        model_type = type(model).__name__
        
        # TreeExplainer対応モデル
        tree_models = [
            'RandomForest', 'GradientBoosting', 'XGB', 'LGBM', 'LightGBM',
            'ExtraTrees', 'DecisionTree', 'CatBoost',
        ]
        
        if any(t in model_type for t in tree_models):
            try:
                return shap.TreeExplainer(model)
            except Exception as e:
                logger.warning(f"TreeExplainer失敗、KernelExplainerを使用: {e}")
        
        # フォールバック: KernelExplainer
        # 背景データをサブサンプリング
        background = shap.sample(X, min(50, len(X)))
        return shap.KernelExplainer(model.predict, background)
    
    def plot_summary(
        self, 
        shap_values: np.ndarray, 
        X: pd.DataFrame,
        max_display: int = 20,
        show: bool = False,
    ) -> plt.Figure:
        """
        Summary Plot（特徴量重要度＋分布）
        
        Args:
            shap_values: SHAP値
            X: 特徴量DataFrame
            max_display: 表示する特徴量数
            show: 表示するか
            
        Returns:
            plt.Figure
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X, 
            max_display=max_display, 
            show=show,
        )
        return plt.gcf()
    
    def plot_bar(
        self, 
        shap_values: np.ndarray, 
        X: pd.DataFrame,
        max_display: int = 20,
        show: bool = False,
    ) -> plt.Figure:
        """
        Bar Plot（特徴量重要度のみ）
        """
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            plot_type="bar",
            max_display=max_display,
            show=show,
        )
        return plt.gcf()
    
    def plot_dependence(
        self, 
        shap_values: np.ndarray, 
        X: pd.DataFrame,
        feature: Union[int, str],
        interaction_feature: Optional[Union[int, str]] = None,
        show: bool = False,
    ) -> plt.Figure:
        """
        Dependence Plot（特徴量とSHAP値の関係）
        """
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, shap_values, X,
            interaction_index=interaction_feature,
            show=show,
        )
        return plt.gcf()
    
    def plot_force(
        self, 
        explainer: shap.Explainer,
        shap_values: np.ndarray,
        X_instance: pd.Series,
        show: bool = False,
    ) -> plt.Figure:
        """
        Force Plot（単一サンプルの説明）
        """
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
        
        plt.figure(figsize=(20, 3))
        shap.force_plot(
            expected_value,
            shap_values[0] if shap_values.ndim > 1 else shap_values,
            X_instance,
            matplotlib=True,
            show=show,
        )
        return plt.gcf()
    
    def get_feature_importance(
        self, 
        shap_values: np.ndarray, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        SHAP値から特徴量重要度を計算
        
        Returns:
            pd.DataFrame: 特徴量名と重要度のDataFrame
        """
        importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=False)
        
        return df
