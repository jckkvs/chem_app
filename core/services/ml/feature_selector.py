"""
sklearn Feature Selection完全統合

Implements: F-FEATURE-SELECT-001
設計思想:
- sklearn.feature_selectionの全手法をサポート
- Filter/Wrapper/Embedded methodsの統合
- 化学特徴量に適した選択戦略

参考文献:
- Feature selection (sklearn documentation)
- An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    # Filter Methods
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    # Wrapper Methods
    RFE,
    RFECV,
    SequentialFeatureSelector,
    # Embedded Methods
    SelectFromModel,
    # Scoring Functions
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    sklearn Feature Selection完全ラッパー
    
    Features:
    - Filter Methods: 統計検定ベース
    - Wrapper Methods: モデルベース反復選択
    - Embedded Methods: モデル組み込み重要度
    - 化学特徴量に適した戦略
    
    Example:
        >>> selector = FeatureSelector(method='k_best', k=50)
        >>> X_selected = selector.fit_transform(X, y)
        >>> selected_features = selector.get_selected_features()
    """
    
    def __init__(
        self,
        method: str = 'k_best',
        task_type: Literal['regression', 'classification'] = 'regression',
        **params
    ):
        """
        Args:
            method: 選択手法 ('variance', 'k_best', 'percentile', 'rfe', 'rfecv', 
                    'sequential', 'from_model', etc.)
            task_type: タスクタイプ
            **params: 各手法固有のパラメータ
        """
        self.method = method
        self.task_type = task_type
        self.params = params
        
        self.selector_: Optional[BaseEstimator] = None
        self.selected_features_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
    
    def _get_scoring_function(self):
        """タスクタイプに応じたスコアリング関数を取得"""
        if self.task_type == 'classification':
            # 分類用スコアリング関数
            scoring_func = self.params.get('score_func', f_classif)
            if scoring_func == 'chi2':
                return chi2
            elif scoring_func == 'mutual_info':
                return mutual_info_classif
            else:
                return f_classif
        else:
            # 回帰用スコアリング関数
            scoring_func = self.params.get('score_func', f_regression)
            if scoring_func == 'mutual_info':
                return mutual_info_regression
            else:
                return f_regression
    
    def _create_selector(self) -> BaseEstimator:
        """手法に応じたselectorを作成"""
        
        # ===== Filter Methods =====
        if self.method == 'variance':
            # Variance Threshold
            threshold = self.params.get('threshold', 0.0)
            return VarianceThreshold(threshold=threshold)
        
        elif self.method == 'k_best':
            # Select K Best
            k = self.params.get('k', 10)
            score_func = self._get_scoring_function()
            return SelectKBest(score_func=score_func, k=k)
        
        elif self.method == 'percentile':
            # Select Percentile
            percentile = self.params.get('percentile', 50)
            score_func = self._get_scoring_function()
            return SelectPercentile(score_func=score_func, percentile=percentile)
        
        elif self.method == 'fpr':
            # Select FPR (False Positive Rate)
            alpha = self.params.get('alpha', 0.05)
            score_func = self._get_scoring_function()
            return SelectFpr(score_func=score_func, alpha=alpha)
        
        elif self.method == 'fdr':
            # Select FDR (False Discovery Rate)
            alpha = self.params.get('alpha', 0.05)
            score_func = self._get_scoring_function()
            return SelectFdr(score_func=score_func, alpha=alpha)
        
        elif self.method == 'fwe':
            # Select FWE (Family-Wise Error)
            alpha = self.params.get('alpha', 0.05)
            score_func = self._get_scoring_function()
            return SelectFwe(score_func=score_func, alpha=alpha)
        
        elif self.method == 'generic_univariate':
            # Generic Univariate Select
            mode = self.params.get('mode', 'percentile')
            param = self.params.get('param', 1e-5)
            score_func = self._get_scoring_function()
            return GenericUnivariateSelect(
                score_func=score_func, mode=mode, param=param
            )
        
        # ===== Wrapper Methods =====
        elif self.method == 'rfe':
            # Recursive Feature Elimination
            estimator = self.params.get('estimator')
            if estimator is None:
                raise ValueError("estimator required for RFE")
            n_features_to_select = self.params.get('n_features_to_select', None)
            step = self.params.get('step', 1)
            return RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=step,
            )
        
        elif self.method == 'rfecv':
            # RFE with Cross-Validation
            estimator = self.params.get('estimator')
            if estimator is None:
                raise ValueError("estimator required for RFECV")
            cv = self.params.get('cv', 5)
            step = self.params.get('step', 1)
            scoring = self.params.get('scoring', None)
            return RFECV(
                estimator=estimator,
                cv=cv,
                step=step,
                scoring=scoring,
            )
        
        elif self.method == 'sequential':
            # Sequential Feature Selector (forward/backward)
            estimator = self.params.get('estimator')
            if estimator is None:
                raise ValueError("estimator required for SequentialFeatureSelector")
            n_features_to_select = self.params.get('n_features_to_select', 'auto')
            direction = self.params.get('direction', 'forward')
            cv = self.params.get('cv', 5)
            return SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                direction=direction,
                cv=cv,
            )
        
        # ===== Embedded Methods =====
        elif self.method == 'from_model':
            # Select From Model
            estimator = self.params.get('estimator')
            if estimator is None:
                raise ValueError("estimator required for SelectFromModel")
            threshold = self.params.get('threshold', None)
            prefit = self.params.get('prefit', False)
            max_features = self.params.get('max_features', None)
            return SelectFromModel(
                estimator=estimator,
                threshold=threshold,
                prefit=prefit,
                max_features=max_features,
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        特徴選択をフィット
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
        
        Returns:
            self
        """
        self.selector_ = self._create_selector()
        
        logger.info(f"Feature selection開始: method={self.method}, shape={X.shape}")
        
        # フィット
        self.selector_.fit(X, y)
        
        # 選択された特徴のマスク取得
        support_mask = self.selector_.get_support()
        self.selected_features_ = X.columns[support_mask].tolist()
        
        # 重要度取得（可能な場合）
        if hasattr(self.selector_, 'scores_'):
            self.feature_importances_ = self.selector_.scores_
        elif hasattr(self.selector_, 'ranking_'):
            # RFEの場合、ランキングを逆にして重要度とする
            self.feature_importances_ = -self.selector_.ranking_
        
        logger.info(
            f"Feature selection完了: {len(self.selected_features_)}/{X.shape[1]} features selected"
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        選択された特徴のみ保持
        
        Args:
            X: 特徴量DataFrame
        
        Returns:
            選択された特徴のみのDataFrame
        """
        if self.selector_ is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        X_selected = self.selector_.transform(X)
        
        return pd.DataFrame(
            X_selected,
            columns=self.selected_features_,
            index=X.index,
        )
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """フィット＆変換"""
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """選択された特徴名を取得"""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted.")
        return self.selected_features_
    
    def get_feature_scores(self) -> Optional[pd.DataFrame]:
        """特徴スコア/重要度を取得"""
        if self.feature_importances_ is None:
            return None
        
        # 元のDataFrameの全特徴に対するスコア
        return pd.DataFrame({
            'feature': self.selector_.feature_names_in_ if hasattr(self.selector_, 'feature_names_in_') else range(len(self.feature_importances_)),
            'score': self.feature_importances_,
        }).sort_values('score', ascending=False)
    
    def get_support_mask(self) -> np.ndarray:
        """選択された特徴のbooleanマスクを取得"""
        if self.selector_ is None:
            raise ValueError("Selector not fitted.")
        return self.selector_.get_support()


# =============================================================================
# ヘルパー関数
# =============================================================================

def auto_select_features(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'regression',
    target_n_features: int = 50,
    method: str = 'k_best',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    自動特徴選択（簡易インターフェース）
    
    Args:
        X: 特徴量DataFrame
        y: ターゲット
        task_type: タスクタイプ
        target_n_features: 目標特徴数
        method: 選択手法
    
    Returns:
        Tuple[選択後DataFrame, 選択された特徴名リスト]
    """
    selector = FeatureSelector(
        method=method,
        task_type=task_type,
        k=target_n_features,
    )
    
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_selected_features()
    
    return X_selected, selected_features
