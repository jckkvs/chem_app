"""
不確実性定量化エンジン

Implements: F-UQ-001
設計思想:
- 予測に信頼区間を付与
- Quantile Regression、Bootstrap、NGBoost対応
- 意思決定支援
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    予測不確実性定量化エンジン
    
    Features:
    - Quantile Regression: 予測分位点
    - Bootstrap Ensemble: 複数モデルのばらつき
    - Prediction Intervals: 信頼区間
    
    Example:
        >>> uq = UncertaintyQuantifier(method='quantile')
        >>> uq.fit(X_train, y_train)
        >>> mean, lower, upper = uq.predict_with_interval(X_test)
    """
    
    def __init__(
        self,
        method: Literal['quantile', 'bootstrap', 'ensemble'] = 'quantile',
        confidence_level: float = 0.9,
        n_estimators: int = 100,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ):
        """
        Args:
            method: 不確実性推定手法
            confidence_level: 信頼水準 (0.9 = 90%)
            n_estimators: 木の数
            n_bootstrap: Bootstrapサンプル数
            random_state: 乱数シード
        """
        self.method = method
        self.confidence_level = confidence_level
        self.n_estimators = n_estimators
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        
        self.models_: List[Any] = []
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'UncertaintyQuantifier':
        """
        不確実性モデルを学習
        """
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        if self.method == 'quantile':
            self._fit_quantile(X_arr, y_arr)
        elif self.method == 'bootstrap':
            self._fit_bootstrap(X_arr, y_arr)
        else:  # ensemble
            self._fit_ensemble(X_arr, y_arr)
        
        self.is_fitted_ = True
        return self
    
    def _fit_quantile(self, X: np.ndarray, y: np.ndarray) -> None:
        """Quantile Regression"""
        alpha = (1 - self.confidence_level) / 2
        quantiles = [alpha, 0.5, 1 - alpha]
        
        for q in quantiles:
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            model.fit(X, y)
            self.models_.append(model)
    
    def _fit_bootstrap(self, X: np.ndarray, y: np.ndarray) -> None:
        """Bootstrap Ensemble"""
        from lightgbm import LGBMRegressor
        
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            model = LGBMRegressor(
                n_estimators=self.n_estimators // 2,
                random_state=self.random_state + i,
                verbose=-1,
            )
            model.fit(X_boot, y_boot)
            self.models_.append(model)
    
    def _fit_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Diverse Ensemble"""
        from lightgbm import LGBMRegressor
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        
        models = [
            LGBMRegressor(n_estimators=self.n_estimators, random_state=self.random_state, verbose=-1),
            XGBRegressor(n_estimators=self.n_estimators, random_state=self.random_state, verbosity=0),
            RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state),
        ]
        
        for model in models:
            model.fit(X, y)
            self.models_.append(model)
    
    def predict_with_interval(
        self,
        X: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        信頼区間付き予測
        
        Returns:
            (mean, lower, upper)
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()を先に呼び出してください")
        
        X_arr = np.array(X)
        
        if self.method == 'quantile':
            lower = self.models_[0].predict(X_arr)
            mean = self.models_[1].predict(X_arr)
            upper = self.models_[2].predict(X_arr)
        else:
            # Bootstrap/Ensemble: 全モデルの予測を集約
            predictions = np.array([m.predict(X_arr) for m in self.models_])
            mean = predictions.mean(axis=0)
            
            alpha = (1 - self.confidence_level) / 2
            lower = np.percentile(predictions, alpha * 100, axis=0)
            upper = np.percentile(predictions, (1 - alpha) * 100, axis=0)
        
        return mean, lower, upper
    
    def get_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """不確実性スコア（区間幅）を取得"""
        _, lower, upper = self.predict_with_interval(X)
        return upper - lower
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """通常の予測（中央値）"""
        mean, _, _ = self.predict_with_interval(X)
        return mean
