"""
Nested Cross-Validation実装

Implements: F-NESTED-CV-001
設計思想:
- 外側のCVでモデル評価の汎化性能を推定
- 内側のCVでハイパーパラメータ最適化
- バイアスを避けた正確な性能推定

参考文献:
- Nested versus non-nested cross-validation (Varma & Simon, 2006)
- DOI: 10.1093/bioinformatics/btl214
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

logger = logging.getLogger(__name__)


class NestedCV:
    """
    Nested Cross-Validation
    
    Features:
    - 外側CV: モデル評価（汎化性能）
    - 内側CV: ハイパーパラメータ最適化
    - 各種CVストラテジー対応
    - スコアリング指標カスタマイズ可能
    
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> param_grid = {'n_estimators': [50, 100, 200]}
        >>> 
        >>> nested_cv = NestedCV(
        ...     outer_cv=5,
        ...     inner_cv=3,
        ...     scoring='neg_mean_squared_error'
        ... )
        >>> 
        >>> results = nested_cv.fit(model, X, y, param_grid)
        >>> print(f"Nested CV Score: {results['outer_scores'].mean():.3f}")
    """
    
    def __init__(
        self,
        outer_cv: int | BaseCrossValidator = 5,
        inner_cv: int | BaseCrossValidator = 3,
        scoring: str = 'neg_mean_squared_error',
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Args:
            outer_cv: 外側CVの分割数またはCVオブジェクト
            inner_cv: 内側CVの分割数またはCVオブジェクト
            scoring: スコアリング指標
            random_state: 乱数シード
            n_jobs: 並列実行数
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.outer_scores_: Optional[List[float]] = None
        self.best_params_: Optional[List[Dict[str, Any]]] = None
    
    def _get_cv_splitter(
        self, 
        cv: int | BaseCrossValidator,
        task_type: str = 'regression'
    ) -> BaseCrossValidator:
        """CVスプリッターを取得"""
        if isinstance(cv, int):
            if task_type == 'classification':
                return StratifiedKFold(
                    n_splits=cv, 
                    shuffle=True, 
                    random_state=self.random_state
                )
            else:
                return KFold(
                    n_splits=cv, 
                    shuffle=True, 
                    random_state=self.random_state
                )
        return cv
    
    def fit(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        task_type: str = 'regression',
    ) -> Dict[str, Any]:
        """
        Nested CVを実行
        
        Args:
            estimator: ベースモデル
            X: 特徴量
            y: ターゲット
            param_grid: ハイパーパラメータグリッド
            task_type: タスクタイプ ('regression' or 'classification')
        
        Returns:
            Dict: 'outer_scores', 'best_params', 'mean_score', 'std_score'
        """
        from sklearn.model_selection import GridSearchCV
        
        # CVスプリッター取得
        outer_cv_splitter = self._get_cv_splitter(self.outer_cv, task_type)
        inner_cv_splitter = self._get_cv_splitter(self.inner_cv, task_type)
        
        outer_scores = []
        best_params_list = []
        
        logger.info(
            f"Nested CV開始: outer_cv={self.outer_cv}, inner_cv={self.inner_cv}"
        )
        
        # 外側CVループ
        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv_splitter.split(X, y)
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 内側CVでハイパーパラメータ最適化
            grid_search = GridSearchCV(
                estimator=clone(estimator),
                param_grid=param_grid,
                cv=inner_cv_splitter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                verbose=0,
            )
            
            grid_search.fit(X_train, y_train)
            
            # 最良モデルで外側テストセット評価
            best_model = grid_search.best_estimator_
            test_score = best_model.score(X_test, y_test)
            
            outer_scores.append(test_score)
            best_params_list.append(grid_search.best_params_)
            
            logger.info(
                f"Fold {fold_idx + 1}: "
                f"Score={test_score:.4f}, "
                f"Best params={grid_search.best_params_}"
            )
        
        # 結果保存
        self.outer_scores_ = outer_scores
        self.best_params_ = best_params_list
        
        results = {
            'outer_scores': np.array(outer_scores),
            'best_params': best_params_list,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
        }
        
        logger.info(
            f"Nested CV完了: "
            f"Mean Score={results['mean_score']:.4f} "
            f"(±{results['std_score']:.4f})"
        )
        
        return results
    
    def get_summary(self) -> str:
        """結果サマリーを取得"""
        if self.outer_scores_ is None:
            return "Not fitted yet"
        
        lines = [
            "=== Nested CV Results ===",
            f"Mean Score: {np.mean(self.outer_scores_):.4f}",
            f"Std Score: {np.std(self.outer_scores_):.4f}",
            f"Min Score: {np.min(self.outer_scores_):.4f}",
            f"Max Score: {np.max(self.outer_scores_):.4f}",
            "\nBest Parameters per Fold:",
        ]
        
        for i, params in enumerate(self.best_params_):
            lines.append(f"  Fold {i+1}: {params}")
        
        return "\n".join(lines)
