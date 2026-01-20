"""
メトリクス計算

Implements: F-METRICS-001
設計思想:
- 回帰/分類メトリクス
- カスタムメトリクス
- 信頼区間
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """メトリクスレポート"""
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, tuple] = None


class MetricsCalculator:
    """
    メトリクス計算
    
    Features:
    - 回帰: R², RMSE, MAE, MAPE
    - 分類: Accuracy, Precision, Recall, F1, AUC
    - 信頼区間（Bootstrap）
    
    Example:
        >>> calc = MetricsCalculator()
        >>> report = calc.regression_metrics(y_true, y_pred)
    """
    
    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        with_ci: bool = False,
    ) -> MetricsReport:
        """回帰メトリクス"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        metrics['r2'] = 1 - (ss_res / (ss_tot + 1e-9))
        
        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # MAPE
        mask = y_true != 0
        if mask.any():
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Max Error
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # Median Absolute Error
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        ci = None
        if with_ci:
            ci = self._bootstrap_ci(y_true, y_pred)
        
        return MetricsReport(metrics=metrics, confidence_intervals=ci)
    
    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> MetricsReport:
        """分類メトリクス"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        # Binary metrics
        if len(np.unique(y_true)) == 2:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            metrics['precision'] = tp / (tp + fp + 1e-9)
            metrics['recall'] = tp / (tp + fn + 1e-9)
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-9)
            metrics['specificity'] = tn / (tn + fp + 1e-9)
            
            # AUC (if probabilities available)
            if y_prob is not None:
                try:
                    from sklearn.metrics import roc_auc_score
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
                except Exception:
                    pass
        
        return MetricsReport(metrics=metrics)
    
    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> Dict[str, tuple]:
        """Bootstrap信頼区間"""
        n = len(y_true)
        alpha = (1 - confidence) / 2
        
        r2_samples = []
        rmse_samples = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            y_t = y_true[indices]
            y_p = y_pred[indices]
            
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - y_t.mean()) ** 2)
            r2_samples.append(1 - (ss_res / (ss_tot + 1e-9)))
            rmse_samples.append(np.sqrt(np.mean((y_t - y_p) ** 2)))
        
        return {
            'r2': (np.percentile(r2_samples, alpha * 100), np.percentile(r2_samples, (1 - alpha) * 100)),
            'rmse': (np.percentile(rmse_samples, alpha * 100), np.percentile(rmse_samples, (1 - alpha) * 100)),
        }
    
    def compare_models(
        self,
        results: Dict[str, MetricsReport],
    ) -> str:
        """モデル比較"""
        lines = ["Model Comparison", "=" * 50]
        
        for name, report in results.items():
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in report.metrics.items())
            lines.append(f"{name}: {metrics_str}")
        
        return "\n".join(lines)
