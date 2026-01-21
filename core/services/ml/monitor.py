"""
モデル監視（Evidently/WhyLabs inspired）

Implements: F-MONITOR-001
設計思想:
- データドリフト検出
- モデル性能監視
- アラート生成
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """ドリフトレポート"""
    timestamp: str
    feature_drifts: Dict[str, float]
    drifted_features: List[str]
    overall_drift_score: float
    is_drifted: bool


@dataclass
class PerformanceReport:
    """性能レポート"""
    timestamp: str
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    degraded_metrics: List[str]
    is_degraded: bool


class ModelMonitor:
    """
    モデル監視（Evidently inspired）
    
    Features:
    - データドリフト検出
    - 性能劣化検知
    - アラート生成
    
    Example:
        >>> monitor = ModelMonitor()
        >>> monitor.set_baseline(X_train, metrics_baseline)
        >>> report = monitor.check_drift(X_new)
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.05,
        performance_threshold: float = 0.1,
    ):
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        self.baseline_stats_: Optional[Dict[str, Dict[str, float]]] = None
        self.baseline_metrics_: Optional[Dict[str, float]] = None
        self.reports_: List[Any] = []
    
    def set_baseline(
        self,
        X: pd.DataFrame,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """ベースラインを設定"""
        self.baseline_stats_ = {}
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.baseline_stats_[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                }
        
        self.baseline_metrics_ = metrics
        logger.info(f"Set baseline with {len(self.baseline_stats_)} features")
    
    def check_drift(self, X: pd.DataFrame) -> DriftReport:
        """データドリフトをチェック"""
        if self.baseline_stats_ is None:
            raise ValueError("Baseline not set. Call set_baseline first.")
        
        feature_drifts = {}
        drifted_features = []
        
        for col in X.columns:
            if col not in self.baseline_stats_:
                continue
            
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            
            baseline = self.baseline_stats_[col]
            current_mean = X[col].mean()
            current_std = X[col].std()
            
            # 標準化差分
            mean_diff = abs(current_mean - baseline['mean']) / (baseline['std'] + 1e-9)
            std_ratio = current_std / (baseline['std'] + 1e-9)
            
            drift_score = mean_diff + abs(1 - std_ratio)
            feature_drifts[col] = drift_score
            
            if drift_score > self.drift_threshold * 10:
                drifted_features.append(col)
        
        overall_drift = np.mean(list(feature_drifts.values())) if feature_drifts else 0
        
        report = DriftReport(
            timestamp=datetime.now().isoformat(),
            feature_drifts=feature_drifts,
            drifted_features=drifted_features,
            overall_drift_score=float(overall_drift),
            is_drifted=len(drifted_features) > 0,
        )
        
        self.reports_.append(report)
        return report
    
    def check_performance(
        self,
        current_metrics: Dict[str, float],
    ) -> PerformanceReport:
        """性能劣化をチェック"""
        if self.baseline_metrics_ is None:
            raise ValueError("Baseline metrics not set.")
        
        degraded_metrics = []
        
        for metric, current_value in current_metrics.items():
            if metric not in self.baseline_metrics_:
                continue
            
            baseline_value = self.baseline_metrics_[metric]
            
            # R²などは高いほど良い
            if metric in ['r2', 'accuracy', 'f1']:
                if current_value < baseline_value * (1 - self.performance_threshold):
                    degraded_metrics.append(metric)
            # RMSEなどは低いほど良い
            else:
                if current_value > baseline_value * (1 + self.performance_threshold):
                    degraded_metrics.append(metric)
        
        report = PerformanceReport(
            timestamp=datetime.now().isoformat(),
            current_metrics=current_metrics,
            baseline_metrics=self.baseline_metrics_,
            degraded_metrics=degraded_metrics,
            is_degraded=len(degraded_metrics) > 0,
        )
        
        self.reports_.append(report)
        return report
    
    def get_alerts(self) -> List[str]:
        """アラートを取得"""
        alerts = []
        
        for report in self.reports_:
            if isinstance(report, DriftReport) and report.is_drifted:
                alerts.append(f"⚠️ Data drift detected: {report.drifted_features}")
            elif isinstance(report, PerformanceReport) and report.is_degraded:
                alerts.append(f"🔴 Performance degraded: {report.degraded_metrics}")
        
        return alerts
