"""
mlxtend Sequential Feature Selection統合

Implements: F-MLXTEND-SFS-001
設計思想:
- mlxtendのSequential Feature Selectionを統合
- sklearn SequentialFeatureSelectorより高速で柔軟
- Floating variants（SFFS/SBFS）もサポート

参考文献:
- mlxtend documentation (http://rasbt.github.io/mlxtend/)
- Sequential Feature Selection Algorithms (Pudil et al., 1994)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

# mlxtendの可用性チェック
try:
    from mlxtend.feature_selection import (
        ExhaustiveFeatureSelector,
        SequentialFeatureSelector,
    )
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    logger.warning("mlxtend not installed. Sequential FS features will be limited.")


class MLXtendSequentialFS:
    """
    mlxtend Sequential Feature Selection統合クラス
    
    Features:
    - SFS (Sequential Forward Selection)
    - SBS (Sequential Backward Selection)
    - SFFS (Sequential Floating Forward Selection)
    - SBFS (Sequential Floating Backward Selection)
    - EFS (Exhaustive Feature Selector)
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> 
        >>> selector = MLXtendSequentialFS(
        ...     method='sfs',
        ...     k_features=5,
        ...     estimator=RandomForestClassifier()
        ... )
        >>> selector.fit(X_train, y_train)
        >>> X_selected = selector.transform(X_test)
        >>> print(selector.get_selected_features())
    """
    
    def __init__(
        self,
        method: Literal['sfs', 'sbs', 'sffs', 'sbfs', 'efs'] = 'sfs',
        estimator: Optional[BaseEstimator] = None,
        k_features: Union[int, str, Tuple[int, int]] = 'best',
        forward: bool = True,
        floating: bool = False,
        scoring: str = 'accuracy',
        cv: int = 5,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Args:
            method: 選択手法（'sfs', 'sbs', 'sffs', 'sbfs', 'efs'）
            estimator: ベースモデル
            k_features: 選択する特徴数（'best', 'parsimonious', または整数/範囲）
            forward: Forward selection（True）かBackward（False）
            floating: Floating variants使用
            scoring: スコアリング指標
            cv: Cross-validation分割数
            n_jobs: 並列実行数
            **kwargs: その他のパラメータ
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend is not installed. "
                "Install it with: pip install mlxtend"
            )
        
        self.method = method
        self.estimator = estimator
        self.k_features = k_features
        self.forward = forward
        self.floating = floating
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        self.selector_: Optional[BaseEstimator] = None
        self.selected_features_: Optional[List[str]] = None
        self.subsets_: Optional[Dict] = None
    
    def _create_selector(self) -> BaseEstimator:
        """手法に応じたselectorを作成"""
        
        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = self.estimator
        
        if self.method == 'efs':
            # Exhaustive Feature Selector
            if isinstance(self.k_features, tuple):
                min_features, max_features = self.k_features
            elif isinstance(self.k_features, int):
                min_features = max_features = self.k_features
            else:
                min_features, max_features = 1, None
            
            selector = ExhaustiveFeatureSelector(
                estimator=estimator,
                min_features=min_features,
                max_features=max_features,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        else:
            # Sequential Feature Selector
            # method別にforward/floatingを設定
            if self.method == 'sfs':
                forward = True
                floating = False
            elif self.method == 'sbs':
                forward = False
                floating = False
            elif self.method == 'sffs':
                forward = True
                floating = True
            elif self.method == 'sbfs':
                forward = False
                floating = True
            else:
                # デフォルト設定から
                forward = self.forward
                floating = self.floating
            
            selector = SequentialFeatureSelector(
                estimator=estimator,
                k_features=self.k_features,
                forward=forward,
                floating=floating,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                **self.kwargs
            )
        
        return selector
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLXtendSequentialFS':
        """
        特徴選択をフィット
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
        
        Returns:
            self
        """
        self.selector_ = self._create_selector()
        
        logger.info(
            f"mlxtend {self.method.upper()} feature selection開始: "
            f"{X.shape[1]} features"
        )
        
        # フィット
        self.selector_.fit(X.values, y.values)
        
        # 選択された特徴のインデックス取得
        selected_idx = list(self.selector_.k_feature_idx_)
        self.selected_features_ = X.columns[selected_idx].tolist()
        
        # サブセット情報取得
        self.subsets_ = self.selector_.subsets_
        
        logger.info(
            f"mlxtend {self.method.upper()} 完了: "
            f"{len(self.selected_features_)} features selected"
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
        
        X_selected = self.selector_.transform(X.values)
        
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
    
    def get_metric_dict(self) -> Dict[int, Dict[str, Any]]:
        """各ステップのメトリクス情報を取得"""
        if self.subsets_ is None:
            raise ValueError("Selector not fitted.")
        return self.subsets_
    
    def get_best_score(self) -> float:
        """ベストスコアを取得"""
        if self.selector_ is None:
            raise ValueError("Selector not fitted.")
        return self.selector_.k_score_


# =============================================================================
# ヘルパー関数
# =============================================================================

def sequential_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: Optional[BaseEstimator] = None,
    method: str = 'sfs',
    k_features: Union[int, str] = 5,
    cv: int = 5,
) -> Tuple[pd.DataFrame, List[str], float]:
    """
    Sequential Feature Selection（簡易インターフェース）
    
    Args:
        X: 特徴量DataFrame
        y: ターゲット
        estimator: ベースモデル
        method: 選択手法
        k_features: 選択する特徴数
        cv: Cross-validation分割数
    
    Returns:
        Tuple[選択後DataFrame, 選択された特徴名, ベストスコア]
    """
    selector = MLXtendSequentialFS(
        method=method,
        estimator=estimator,
        k_features=k_features,
        cv=cv,
    )
    
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_selected_features()
    best_score = selector.get_best_score()
    
    return X_selected, selected_features, best_score


def compare_sequential_methods(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: Optional[BaseEstimator] = None,
    k_features: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    複数のSequential FS手法を比較
    
    Args:
        X: 特徴量DataFrame
        y: ターゲット
        estimator: ベースモデル
        k_features: 選択する特徴数
    
    Returns:
        Dict: 各手法の結果
    """
    methods = ['sfs', 'sbs', 'sffs', 'sbfs']
    results = {}
    
    for method in methods:
        try:
            selector = MLXtendSequentialFS(
                method=method,
                estimator=estimator,
                k_features=k_features,
                cv=3  # 比較のため高速化
            )
            selector.fit(X, y)
            
            results[method] = {
                'selected_features': selector.get_selected_features(),
                'best_score': selector.get_best_score(),
                'n_features': len(selector.get_selected_features()),
            }
        except Exception as e:
            logger.warning(f"{method} failed: {e}")
            results[method] = {'error': str(e)}
    
    return results
