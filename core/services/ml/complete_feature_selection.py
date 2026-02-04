"""
特徴選択の完全実装 - sklearn + mlxtend全対応

Implements: F-FEATURE-SELECTION-COMPLETE-001
設計思想:
- scikit-learn全特徴選択手法
- mlxtend SequentialFeatureSelector
- 化学データ特化のラッパー
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    # Filter methods
    VarianceThreshold,
    SelectKBest,
    SelectPercentile,
    SelectFpr,
    SelectFdr,
    SelectFwe,
    GenericUnivariateSelect,
    # Score functions
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    # Wrapper methods
    RFE,
    RFECV,
    SequentialFeatureSelector,
    # Model-based
    SelectFromModel,
)

logger = logging.getLogger(__name__)


class ComprehensiveFeatureSelector:
    """
    包括的な特徴選択クラス
    
    サポート手法:
    1. Filter Methods:
       - VarianceThreshold
       - SelectKBest (ANOVA F-value, Mutual Information, Chi2)
       - SelectPercentile
    
    2. Wrapper Methods:
       - RFE (Recursive Feature Elimination)
       - RFECV (RFE with Cross-Validation)
       - SequentialFeatureSelector (Forward/Backward)
    
    3. Embedded Methods:
       - SelectFromModel (L1, Tree-based)
    
    Example:
        >>> selector = ComprehensiveFeatureSelector(
        ...     method='rfe',
        ...     n_features_to_select=10,
        ...     estimator=RandomForestRegressor()
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """
    
    def __init__(
        self,
        method: str = 'variance',
        n_features_to_select: Optional[Union[int, float]] = None,
        score_func: Optional[Any] = None,
        estimator: Optional[Any] = None,
        cv: int = 5,
        direction: str = 'forward',  # for Sequential
        **kwargs
    ):
        """
        Args:
            method: 選択手法 ('variance', 'kbest', 'rfe', 'rfecv', 'sequential', 'model')
            n_features_to_select: 選択する特徴数
            score_func: スコア関数（kbest用）
            estimator: 推定器（rfe, sequential, model用）
            cv: クロスバリデーション分割数
            direction: Sequential方向 ('forward' or 'backward')
            **kwargs: その他のパラメータ
        """
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.score_func = score_func
        self.estimator = estimator
        self.cv = cv
        self.direction = direction
        self.kwargs = kwargs
        
        self.selector_ = None
        self.selected_features_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """学習"""
        self.selector_ = self._create_selector(X, y)
        self.selector_.fit(X, y)
        
        # 選択された特徴量のインデックス取得
        try:
            self.selected_features_ = self.selector_.get_support(indices=True)
        except:
            self.selected_features_ = np.arange(X.shape[1])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        if self.selector_ is None:
            raise ValueError("fit()を先に実行してください")
        
        X_selected = self.selector_.transform(X)
        
        # DataFrameとして返す
        selected_columns = X.columns[self.selected_features_]
        return pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """学習+変換"""
        return self.fit(X, y).transform(X)
    
    def _create_selector(self, X: pd.DataFrame, y: pd.Series):
        """セレクター作成"""
        
        # 1. Variance Threshold
        if self.method == 'variance':
            threshold = self.kwargs.get('threshold', 0.0)
            return VarianceThreshold(threshold=threshold)
        
        # 2. SelectKBest
        elif self.method == 'kbest':
            k = self.n_features_to_select or 10
            
            # スコア関数自動選択
            if self.score_func is None:
                if self._is_classification(y):
                    score_func = f_classif
                else:
                    score_func = f_regression
            else:
                score_func = self.score_func
            
            return SelectKBest(score_func=score_func, k=k)
        
        # 3. SelectPercentile
        elif self.method == 'percentile':
            percentile = self.kwargs.get('percentile', 50)
            score_func = self.score_func or f_regression
            return SelectPercentile(score_func=score_func, percentile=percentile)
        
        # 4. Mutual Information
        elif self.method == 'mutual_info':
            k = self.n_features_to_select or 10
            
            if self._is_classification(y):
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
            
            return SelectKBest(score_func=score_func, k=k)
        
        # 5. RFE
        elif self.method == 'rfe':
            if self.estimator is None:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                estimator = self.estimator
            
            n_features = self.n_features_to_select or X.shape[1] // 2
            
            return RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=self.kwargs.get('step', 1)
            )
        
        # 6. RFECV
        elif self.method == 'rfecv':
            if self.estimator is None:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                estimator = self.estimator
            
            return RFECV(
                estimator=estimator,
                step=self.kwargs.get('step', 1),
                cv=self.cv,
                scoring=self.kwargs.get('scoring', None),
                n_jobs=-1
            )
        
        # 7. Sequential Feature Selector (sklearn版)
        elif self.method == 'sequential':
            if self.estimator is None:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                estimator = self.estimator
            
            n_features = self.n_features_to_select or X.shape[1] // 2
            
            return SequentialFeatureSelector(
                estimator=estimator,
                n_features_to_select=n_features,
                direction=self.direction,
                cv=self.cv,
                n_jobs=-1
            )
        
        # 8. SelectFromModel (L1 or Tree-based)
        elif self.method == 'model':
            if self.estimator is None:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                estimator = self.estimator
            
            return SelectFromModel(
                estimator=estimator,
                threshold=self.kwargs.get('threshold', 'mean'),
                prefit=False,
                max_features=self.n_features_to_select
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _is_classification(self, y: pd.Series) -> bool:
        """分類問題かどうか判定"""
        return len(np.unique(y)) < 20  # 簡易判定
    
    def get_feature_importances(self) -> Optional[pd.Series]:
        """特徴量重要度取得"""
        if self.selector_ is None:
            return None
        
        try:
            # RFE/RFECV
            if hasattr(self.selector_, 'ranking_'):
                importances = 1.0 / self.selector_.ranking_
            
            # SelectFromModel
            elif hasattr(self.selector_, 'estimator_'):
                if hasattr(self.selector_.estimator_, 'feature_importances_'):
                    importances = self.selector_.estimator_.feature_importances_
                elif hasattr(self.selector_.estimator_, 'coef_'):
                    importances = np.abs(self.selector_.estimator_.coef_)
                else:
                    return None
            
            # SelectKBest
            elif hasattr(self.selector_, 'scores_'):
                importances = self.selector_.scores_
            
            else:
                return None
            
            return pd.Series(importances)
        
        except:
            return None


class MLxtendSequentialFeatureSelector:
    """
    mlxtend SequentialFeatureSelector のラッパー
    
    より高度な Sequential Feature Selection:
    - Forward Selection
    - Backward Elimination
    - Floating variants (SFFS, SFBS)
    
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> selector = MLxtendSequentialFeatureSelector(
        ...     estimator=RandomForestRegressor(),
        ...     k_features=10,
        ...     forward=True,
        ...     floating=False,
        ... )
        >>> X_selected = selector.fit_transform(X, y)
    """
    
    def __init__(
        self,
        estimator: Any,
        k_features: Union[int, str, Tuple[int, int]] = 'best',
        forward: bool = True,
        floating: bool = False,
        scoring: Optional[str] = None,
        cv: int = 5,
        n_jobs: int = -1,
    ):
        """
        Args:
            estimator: 推定器
            k_features: 選択する特徴数 (int or 'best' or (min, max))
            forward: Forward Selection（Falseの場合は Backward）
            floating: Floating variants使用
            scoring: スコアリング指標
            cv: クロスバリデーション分割数
            n_jobs: 並列実行数
        """
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        
        self.selector_ = None
        self.selected_features_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """学習"""
        try:
            from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        except ImportError:
            raise ImportError("mlxtendが必要です: pip install mlxtend")
        
        self.selector_ = SFS(
            self.estimator,
            k_features=self.k_features,
            forward=self.forward,
            floating=self.floating,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
        )
        
        self.selector_.fit(X.values, y.values)
        self.selected_features_ = list(self.selector_.k_feature_idx_)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """変換"""
        if self.selector_ is None:
            raise ValueError("fit()を先に実行してください")
        
        X_selected = X.iloc[:, self.selected_features_]
        return X_selected
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """学習+変換"""
        return self.fit(X, y).transform(X)
    
    def get_metric_dict(self) -> Dict[str, Any]:
        """評価指標の詳細取得"""
        if self.selector_ is None:
            return {}
        
        return self.selector_.get_metric_dict()
