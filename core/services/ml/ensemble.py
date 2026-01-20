"""
モデルアンサンブルエンジン

Implements: F-ENSEMBLE-001
設計思想:
- 複数モデルの組み合わせ
- 加重平均、スタッキング
- モデル多様性評価
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    モデルアンサンブルエンジン
    
    Features:
    - 加重平均アンサンブル
    - スタッキング（メタ学習）
    - モデル多様性評価
    
    Example:
        >>> ensemble = EnsembleModel(method='weighted')
        >>> ensemble.fit(X, y, models=[lgb, xgb, rf])
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        method: Literal['average', 'weighted', 'stacking'] = 'weighted',
        weights: Optional[List[float]] = None,
    ):
        self.method = method
        self.weights = weights
        self.models_: List[Any] = []
        self.meta_model_: Optional[Any] = None
        self.is_fitted_ = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Optional[List[Any]] = None,
    ) -> 'EnsembleModel':
        """アンサンブルを学習"""
        if models is None:
            models = self._create_default_models()
        
        self.models_ = []
        for model in models:
            cloned = clone(model)
            cloned.fit(X, y)
            self.models_.append(cloned)
        
        if self.method == 'weighted' and self.weights is None:
            # CVスコアで重み決定
            self.weights = self._calculate_weights(X, y)
        
        if self.method == 'stacking':
            self._fit_stacking(X, y)
        
        self.is_fitted_ = True
        return self
    
    def _create_default_models(self) -> List[Any]:
        """デフォルトモデル"""
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor
        
        return [
            LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            RandomForestRegressor(n_estimators=100, random_state=42),
        ]
    
    def _calculate_weights(self, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """CVスコアベースの重み計算"""
        from sklearn.model_selection import cross_val_score
        
        scores = []
        for model in self.models_:
            cv_score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            scores.append(-cv_score.mean())  # MSEを正にして小さいほど良い
        
        # 逆数で重み化
        inv_scores = [1 / (s + 1e-9) for s in scores]
        total = sum(inv_scores)
        return [s / total for s in inv_scores]
    
    def _fit_stacking(self, X: pd.DataFrame, y: pd.Series) -> None:
        """スタッキング"""
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import Ridge
        
        # メタ特徴量生成
        meta_features = []
        for model in self.models_:
            oof_preds = cross_val_predict(model, X, y, cv=3)
            meta_features.append(oof_preds)
        
        meta_X = np.column_stack(meta_features)
        
        self.meta_model_ = Ridge(alpha=1.0)
        self.meta_model_.fit(meta_X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if not self.is_fitted_:
            raise RuntimeError("fit()を先に呼び出してください")
        
        predictions = np.array([m.predict(X) for m in self.models_])
        
        if self.method == 'average':
            return predictions.mean(axis=0)
        
        elif self.method == 'weighted':
            weighted = np.zeros(predictions.shape[1])
            for i, w in enumerate(self.weights):
                weighted += w * predictions[i]
            return weighted
        
        else:  # stacking
            meta_X = predictions.T
            return self.meta_model_.predict(meta_X)
    
    def get_model_contributions(self) -> Dict[str, float]:
        """各モデルの貢献度"""
        if self.method == 'weighted':
            return {
                f"Model_{i}": w
                for i, w in enumerate(self.weights)
            }
        return {}
