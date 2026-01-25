"""
sklearn Metrics完全統合

Implements: F-METRICS-001
設計思想:
- sklearn.metricsの全評価指標をサポート
- 回帰/分類/クラスタリングメトリクスの統一的な計算
- 化学MLに適した評価指標の提供

参考文献:
- scikit-learn metrics documentation
- Model Evaluation Metrics (sklearn user guide)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    sklearn Metrics完全ラッパー
    
    Features:
    - 回帰メトリクス（10+種類）
    - 分類メトリクス（15+種類）
    - クラスタリングメトリクス（8+種類）
    - 一括計算機能
    
    Example:
        >>> calc = MetricsCalculator(task_type='regression')
        >>> metrics = calc.calculate_all(y_true, y_pred)
        >>> print(metrics['r2'], metrics['rmse'])
    """
    
    def __init__(self, task_type: Literal['regression', 'classification', 'clustering'] = 'regression'):
        """
        Args:
            task_type: タスクタイプ
        """
        self.task_type = task_type
    
    # ===== 回帰メトリクス =====
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        回帰メトリクスを一括計算
        
        Args:
            y_true: 真値
            y_pred: 予測値
        
        Returns:
            Dict: 全回帰メトリクス
        """
        from sklearn.metrics import (
            explained_variance_score,
            max_error,
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            median_absolute_error,
            r2_score,
        )
        
        metrics = {
            # 基本メトリクス
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            
            # 追加メトリクス
            'explained_variance': explained_variance_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred),
            'max_error': max_error(y_true, y_pred),
        }
        
        # Mean Squared Log Error（非負値のみ）
        if np.all(y_true >= 0) and np.all(y_pred >= 0):
            from sklearn.metrics import mean_squared_log_error
            metrics['msle'] = mean_squared_log_error(y_true, y_pred)
            metrics['rmsle'] = np.sqrt(metrics['msle'])
        
        return metrics
    
    # ===== 分類メトリクス =====
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        average: str = 'binary',
    ) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        分類メトリクスを一括計算
        
        Args:
            y_true: 真値
            y_pred: 予測クラス
            y_pred_proba: 予測確率（オプション）
            average: 多クラスの平均方法
        
        Returns:
            Dict: 全分類メトリクス
        """
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
            hamming_loss,
            jaccard_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )
        
        # クラス数判定
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
        
        metrics = {
            # 基本メトリクス
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            'recall': recall_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            
            # 追加メトリクス
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'hamming_loss': hamming_loss(y_true, y_pred),
        }
        
        # Jaccardスコア（バイナリまたはマルチクラス）
        try:
            metrics['jaccard'] = jaccard_score(
                y_true, y_pred, average=average, zero_division=0
            )
        except (ValueError, ZeroDivisionError) as e:
            logger.debug(f"Jaccard score calculation failed: {e}")
        
        # 混同行列
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # 分類レポート（Dict形式）
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        # 確率ベースのメトリクス
        if y_pred_proba is not None:
            if is_binary:
                from sklearn.metrics import (
                    average_precision_score,
                    brier_score_loss,
                    log_loss,
                    roc_auc_score,
                )
                
                # バイナリ分類
                if y_pred_proba.ndim == 2:
                    y_proba_pos = y_pred_proba[:, 1]
                else:
                    y_proba_pos = y_pred_proba
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_pos)
                metrics['average_precision'] = average_precision_score(
                    y_true, y_proba_pos
                )
                metrics['brier_score'] = brier_score_loss(y_true, y_proba_pos)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
            else:
                # マルチクラス分類
                from sklearn.metrics import log_loss, roc_auc_score
                
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='macro'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovo', average='macro'
                    )
                except (ValueError, ZeroDivisionError) as e:
                    logger.debug(f"ROC AUC calculation failed: {e}")
                
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        
        return metrics
    
    # ===== クラスタリングメトリクス =====
    
    def calculate_clustering_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        labels_true: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        クラスタリングメトリクスを一括計算
        
        Args:
            X: 特徴量
            labels: 予測クラスタラベル
            labels_true: 真のラベル（オプション）
        
        Returns:
            Dict: 全クラスタリングメトリクス
        """
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
        
        metrics = {
            # 内部指標（ラベル不要）
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
        }
        
        # 外部指標（真のラベル必要）
        if labels_true is not None:
            from sklearn.metrics import (
                adjusted_mutual_info_score,
                adjusted_rand_score,
                completeness_score,
                fowlkes_mallows_score,
                homogeneity_score,
                mutual_info_score,
                normalized_mutual_info_score,
                rand_score,
                v_measure_score,
            )
            
            metrics.update({
                'adjusted_rand_score': adjusted_rand_score(labels_true, labels),
                'adjusted_mutual_info': adjusted_mutual_info_score(
                    labels_true, labels
                ),
                'mutual_info': mutual_info_score(labels_true, labels),
                'normalized_mutual_info': normalized_mutual_info_score(
                    labels_true, labels
                ),
                'homogeneity': homogeneity_score(labels_true, labels),
                'completeness': completeness_score(labels_true, labels),
                'v_measure': v_measure_score(labels_true, labels),
                'fowlkes_mallows': fowlkes_mallows_score(labels_true, labels),
                'rand_score': rand_score(labels_true, labels),
            })
        
        return metrics
    
    # ===== 統合計算 =====
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        タスクタイプに応じて全メトリクスを計算
        
        Args:
            y_true: 真値
            y_pred: 予測値
            y_pred_proba: 予測確率（分類のみ）
            X: 特徴量（クラスタリングのみ）
        
        Returns:
            Dict: 全メトリクス
        """
        if self.task_type == 'regression':
            return self.calculate_regression_metrics(y_true, y_pred)
        
        elif self.task_type == 'classification':
            return self.calculate_classification_metrics(
                y_true, y_pred, y_pred_proba
            )
        
        elif self.task_type == 'clustering':
            if X is None:
                raise ValueError("X is required for clustering metrics")
            return self.calculate_clustering_metrics(X, y_pred, y_true)
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")
    
    def get_best_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        最良メトリクスを取得（タスクタイプ別）
        
        Args:
            metrics: メトリクスDict
        
        Returns:
            Dict: ベストメトリクスの情報
        """
        if self.task_type == 'regression':
            # 回帰: R2が最も代表的
            return {
                'best_metric': 'r2',
                'best_value': metrics.get('r2', 0.0),
                'mae': metrics.get('mae', 0.0),
                'rmse': metrics.get('rmse', 0.0),
            }
        
        elif self.task_type == 'classification':
            # 分類: Accuracyが最も代表的
            return {
                'best_metric': 'accuracy',
                'best_value': metrics.get('accuracy', 0.0),
                'f1': metrics.get('f1', 0.0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
            }
        
        elif self.task_type == 'clustering':
            # クラスタリング: Silhouetteが最も代表的
            return {
                'best_metric': 'silhouette',
                'best_value': metrics.get('silhouette', 0.0),
                'calinski_harabasz': metrics.get('calinski_harabasz', 0.0),
                'davies_bouldin': metrics.get('davies_bouldin', 0.0),
            }


# =============================================================================
# ヘルパー関数
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'regression',
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    モデル評価（簡易インターフェース）
    
    Args:
        y_true: 真値
        y_pred: 予測値
        task_type: タスクタイプ
        y_pred_proba: 予測確率
    
    Returns:
        Dict: 全メトリクス
    """
    calc = MetricsCalculator(task_type=task_type)
    return calc.calculate_all(y_true, y_pred, y_pred_proba)


def print_metrics_summary(metrics: Dict[str, Union[float, np.ndarray, Dict]]) -> None:
    """メトリクスサマリーを表示"""
    print("=== Metrics Summary ===")
    
    for key, value in metrics.items():
        # 混同行列やレポートはスキップ
        if isinstance(value, (list, np.ndarray, dict)):
            continue
        
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
