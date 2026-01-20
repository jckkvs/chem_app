"""
マルチタスク学習エンジン

Implements: F-MULTITASK-001
設計思想:
- 複数物性の同時予測
- ハードパラメータ共有
- タスク間相関の活用
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class MultiTaskLearner:
    """
    マルチタスク学習エンジン
    
    Features:
    - 複数ターゲットの同時予測
    - 共有表現学習
    - タスク間相関分析
    
    Example:
        >>> mtl = MultiTaskLearner(base_model='lightgbm')
        >>> mtl.fit(X, y_multi)  # y_multi: DataFrame with multiple columns
        >>> predictions = mtl.predict(X_test)  # Dict[task_name, predictions]
    """
    
    def __init__(
        self,
        base_model: str = 'lightgbm',
        task_type: str = 'regression',
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """
        Args:
            base_model: 'lightgbm', 'xgboost', 'random_forest'
            task_type: 'regression' or 'classification'
            n_estimators: 推定器数
            random_state: 乱数シード
        """
        self.base_model = base_model
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self.model_: Optional[Any] = None
        self.task_names_: List[str] = []
        self.task_correlations_: Optional[pd.DataFrame] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> 'MultiTaskLearner':
        """
        マルチタスクモデルを学習
        
        Args:
            X: 特徴量
            y: 複数ターゲット（DataFrame）
        """
        self.task_names_ = list(y.columns)
        
        # タスク間相関を計算
        self.task_correlations_ = y.corr()
        
        # ベースモデルを作成
        base = self._create_base_model()
        
        # MultiOutput wrapper
        if self.task_type == 'regression':
            self.model_ = MultiOutputRegressor(base, n_jobs=-1)
        else:
            self.model_ = MultiOutputClassifier(base, n_jobs=-1)
        
        self.model_.fit(X, y)
        
        logger.info(f"MultiTask model fitted: {len(self.task_names_)} tasks")
        return self
    
    def _create_base_model(self) -> BaseEstimator:
        """ベースモデルを作成"""
        if self.base_model == 'lightgbm':
            from lightgbm import LGBMRegressor, LGBMClassifier
            if self.task_type == 'regression':
                return LGBMRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    verbose=-1,
                )
            return LGBMClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbose=-1,
            )
        
        elif self.base_model == 'xgboost':
            from xgboost import XGBRegressor, XGBClassifier
            if self.task_type == 'regression':
                return XGBRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    verbosity=0,
                )
            return XGBClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                verbosity=0,
            )
        
        else:  # random_forest
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            if self.task_type == 'regression':
                return RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                )
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        全タスクを予測
        
        Returns:
            Dict[task_name, predictions]
        """
        if self.model_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        predictions = self.model_.predict(X)
        
        return {
            name: predictions[:, i]
            for i, name in enumerate(self.task_names_)
        }
    
    def predict_single_task(self, X: pd.DataFrame, task_name: str) -> np.ndarray:
        """特定タスクのみ予測"""
        all_preds = self.predict(X)
        if task_name not in all_preds:
            raise ValueError(f"Unknown task: {task_name}")
        return all_preds[task_name]
    
    def get_task_correlations(self) -> pd.DataFrame:
        """タスク間相関行列を取得"""
        if self.task_correlations_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        return self.task_correlations_
    
    def get_feature_importances(self) -> Dict[str, pd.Series]:
        """各タスクの特徴量重要度"""
        if self.model_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        importances = {}
        for i, (name, estimator) in enumerate(zip(self.task_names_, self.model_.estimators_)):
            if hasattr(estimator, 'feature_importances_'):
                importances[name] = pd.Series(
                    estimator.feature_importances_,
                    name=name,
                )
        
        return importances
