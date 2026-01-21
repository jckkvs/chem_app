"""
A/Bテストフレームワーク

Implements: F-ABTEST-001
設計思想:
- モデル比較実験
- 統計的有意性検定
- 結果可視化
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """A/Bテスト結果"""
    experiment_name: str
    variant_a: str
    variant_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    is_significant: bool
    winner: Optional[str] = None
    confidence_level: float = 0.95


class ABTester:
    """
    A/Bテストフレームワーク
    
    Features:
    - 2群比較
    - t検定/Mann-Whitney検定
    - 効果量計算
    
    Example:
        >>> tester = ABTester()
        >>> result = tester.compare(
        ...     predictions_a, predictions_b,
        ...     ground_truth, metric='rmse'
        ... )
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compare(
        self,
        values_a: List[float],
        values_b: List[float],
        experiment_name: str = "Experiment",
        variant_a: str = "A",
        variant_b: str = "B",
        metric: str = "metric",
        test_type: str = "ttest",
    ) -> ABTestResult:
        """2群を比較"""
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)
        
        mean_a = arr_a.mean()
        mean_b = arr_b.mean()
        std_a = arr_a.std()
        std_b = arr_b.std()
        
        if test_type == "ttest":
            _, p_value = stats.ttest_ind(arr_a, arr_b)
        elif test_type == "mannwhitney":
            _, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
        else:
            _, p_value = stats.ttest_ind(arr_a, arr_b)
        
        is_significant = p_value < self.alpha
        
        winner = None
        if is_significant:
            winner = variant_a if mean_a > mean_b else variant_b
        
        return ABTestResult(
            experiment_name=experiment_name,
            variant_a=variant_a,
            variant_b=variant_b,
            metric=metric,
            mean_a=float(mean_a),
            mean_b=float(mean_b),
            std_a=float(std_a),
            std_b=float(std_b),
            p_value=float(p_value),
            is_significant=is_significant,
            winner=winner,
            confidence_level=self.confidence_level,
        )
    
    def compare_models(
        self,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> ABTestResult:
        """モデル予測を比較"""
        errors_a = np.abs(y_true - y_pred_a)
        errors_b = np.abs(y_true - y_pred_b)
        
        return self.compare(
            errors_a.tolist(),
            errors_b.tolist(),
            experiment_name="Model Comparison",
            variant_a=model_a,
            variant_b=model_b,
            metric="MAE",
        )
    
    def effect_size(
        self,
        values_a: List[float],
        values_b: List[float],
    ) -> float:
        """効果量（Cohen's d）を計算"""
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)
        
        pooled_std = np.sqrt((arr_a.var() + arr_b.var()) / 2)
        return (arr_a.mean() - arr_b.mean()) / (pooled_std + 1e-9)
    
    def sample_size_required(
        self,
        effect_size: float,
        power: float = 0.8,
    ) -> int:
        """必要サンプルサイズを計算"""
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
