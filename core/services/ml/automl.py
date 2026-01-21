"""
AutoML エンジン - Optuna統合

Implements: F-AUTOML-001
設計思想:
- ハイパーパラメータ自動最適化
- 初心者にも使いやすいワンクリック最適化
- 複数モデルタイプ対応
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class AutoMLEngine:
    """
    AutoML ハイパーパラメータ最適化エンジン
    
    Features:
    - Optuna TPEサンプラーによる効率的な探索
    - 複数モデルタイプ対応（LightGBM, XGBoost, RandomForest）
    - Early Stopping対応
    - 探索結果の可視化
    
    Example (シンプル):
        >>> automl = AutoMLEngine()
        >>> best_params, best_score = automl.optimize(X, y)
    
    Example (カスタマイズ):
        >>> automl = AutoMLEngine(
        ...     model_type='lightgbm',
        ...     n_trials=100,
        ...     timeout=600,
        ... )
        >>> best_params, best_score = automl.optimize(X, y)
    """
    
    def __init__(
        self,
        model_type: Literal['lightgbm', 'xgboost', 'random_forest'] = 'lightgbm',
        task_type: Literal['regression', 'classification'] = 'regression',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            model_type: モデルタイプ
            task_type: タスクタイプ
            n_trials: 試行回数
            timeout: タイムアウト秒数
            cv_folds: クロスバリデーション分割数
            random_state: 乱数シード
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for AutoML. Run: pip install optuna")
        
        self.model_type = model_type
        self.task_type = task_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # 探索結果
        self.study_: Optional[optuna.Study] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> tuple[Dict[str, Any], float]:
        """
        ハイパーパラメータを最適化
        
        Args:
            X: 特徴量
            y: ターゲット
            callback: 進捗コールバック (trial_number, score) -> None
            
        Returns:
            (best_params, best_score)
        """
        direction = "minimize" if self.task_type == "regression" else "maximize"
        
        self.study_ = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.random_state),
        )
        
        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            model = self._create_model(params)
            
            scoring = "neg_mean_squared_error" if self.task_type == "regression" else "accuracy"
            
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=-1,
            )
            
            score = scores.mean()
            
            if callback:
                callback(trial.number, score)
            
            return -score if self.task_type == "regression" else score
        
        # Optunaのログを抑制
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value
        
        logger.info(f"AutoML completed: best_score={self.best_score_:.4f}")
        
        return self.best_params_, self.best_score_
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """モデルタイプに応じたパラメータを提案"""
        if self.model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'verbose': -1,
                'n_jobs': -1,
            }
        
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
            }
        
        else:  # random_forest
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state,
                'n_jobs': -1,
            }
    
    def _create_model(self, params: Dict[str, Any]):
        """パラメータからモデルを作成"""
        if self.model_type == 'lightgbm':
            if self.task_type == 'regression':
                return lgb.LGBMRegressor(**params)
            return lgb.LGBMClassifier(**params)
        
        elif self.model_type == 'xgboost':
            if self.task_type == 'regression':
                return XGBRegressor(**params)
            return XGBClassifier(**params)
        
        else:  # random_forest
            if self.task_type == 'regression':
                return RandomForestRegressor(**params)
            return RandomForestClassifier(**params)
    
    def get_best_model(self):
        """最適パラメータでモデルを作成"""
        if self.best_params_ is None:
            raise RuntimeError("optimize()を先に実行してください")
        return self._create_model(self.best_params_)
    
    def get_optimization_history(self) -> pd.DataFrame:
        """最適化履歴を取得"""
        if self.study_ is None:
            return pd.DataFrame()
        
        trials = self.study_.trials
        data = []
        
        for trial in trials:
            row = {
                'trial': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                **trial.params,
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_param_importances(self) -> Dict[str, float]:
        """パラメータ重要度を取得"""
        if self.study_ is None:
            return {}
        
        try:
            return optuna.importance.get_param_importances(self.study_)
        except Exception:
            return {}
