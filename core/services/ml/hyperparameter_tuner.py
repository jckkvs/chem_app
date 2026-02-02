"""
自動ハイパーパラメータ調整

RandomizedSearchCV/GridSearchCVを使用してモデルを最適化
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """
    チューニング結果
    """
    best_params: Dict[str, Any]
    best_score: float
    tuning_history: List[Dict[str, Any]] = field(default_factory=list)
    total_trials: int = 0
    total_time: float = 0.0  # 秒
    improvement: float = 0.0  # デフォルトからの改善率
    cv_results: Optional[Dict[str, Any]] = None


class HyperparameterTuner:
    """
    ハイパーパラメータ自動調整
    
    Features:
    - RandomizedSearchCV統合（高速）
    - GridSearchCV統合（精密）
    - モデル別探索空間
    - プログレスコールバック
    - 結果可視化
    
    Example:
        >>> tuner = HyperparameterTuner()
        >>> result = tuner.tune(
        ...     model_type='lightgbm',
        ...     X=X_train, y=y_train,
        ...     method='random',
        ...     n_iter=20
        ... )
        >>> print(result.best_params)
    """
    
    # モデル別探索空間（RandomizedSearchCV用）
    SEARCH_SPACES_RANDOM = {
        'random_forest': {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [5, 10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
            'num_leaves': [15, 31, 63, 127, 255],
            'max_depth': [3, 5, 7, 10, 15, -1],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200, 300, 400],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 10, 12],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.5, 1.0]
        }
    }
    
    # GridSearchCV用（より細かい探索）
    SEARCH_SPACES_GRID = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'lightgbm': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.03, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'max_depth': [5, 10, -1]
        },
        'xgboost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [5, 7, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 乱数シード
        """
        self.random_state = random_state
    
    def tune(
        self,
        model_type: str,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'random',
        n_iter: int = 20,
        cv: int = 5,
        scoring: Optional[str] = None,
        custom_search_space: Optional[Dict[str, List]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        n_jobs: int = -1
    ) -> TuningResult:
        """
        ハイパーパラメータ調整を実行
        
        Args:
            model_type: モデルタイプ ('random_forest', 'lightgbm', 'xgboost')
            X: 特徴量
            y: ターゲット
            method: 探索方法 ('random', 'grid')
            n_iter: RandomizedSearchの試行回数
            cv: クロスバリデーションfold数
            scoring: スコアリング方法（Noneで自動判定）
            custom_search_space: カスタム探索空間
            progress_callback: プログレスコールバック関数
            n_jobs: 並列度
        
        Returns:
            TuningResult
        
        Raises:
            ValueError: サポートされていないモデルタイプ
        """
        start_time = time.time()
        
        # モデルタイプ検証
        if model_type not in self.SEARCH_SPACES_RANDOM:
            available = ', '.join(self.SEARCH_SPACES_RANDOM.keys())
            raise ValueError(
                f"Model type '{model_type}' not supported. "
                f"Available: {available}"
            )
        
        # スコアリング自動判定
        if scoring is None:
            scoring = self._auto_detect_scoring(y)
        
        # 探索空間取得
        if custom_search_space is not None:
            search_space = custom_search_space
        elif method == 'random':
            search_space = self.SEARCH_SPACES_RANDOM[model_type]
        else:  # grid
            search_space = self.SEARCH_SPACES_GRID[model_type]
        
        # ベースモデル作成
        base_model = self._create_base_model(model_type)
        
        # デフォルトスコア計算（比較用）
        from sklearn.model_selection import cross_val_score
        default_scores = cross_val_score(
            base_model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        default_score = np.mean(default_scores)
        
        logger.info(f"Default score: {default_score:.4f}")
        
        # サーチ実行
        if method == 'random':
            searcher = RandomizedSearchCV(
                base_model,
                param_distributions=search_space,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:  # grid
            searcher = GridSearchCV(
                base_model,
                param_grid=search_space,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        
        # フィット
        logger.info(f"Starting {method} search with {n_iter if method == 'random' else 'all'} iterations...")
        searcher.fit(X, y)
        
        # 結果収集
        best_params = searcher.best_params_
        best_score = searcher.best_score_
        improvement = ((best_score - default_score) / abs(default_score)) * 100
        
        # 履歴作成
        history = self._extract_history(searcher.cv_results_)
        
        total_time = time.time() - start_time
        
        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            tuning_history=history,
            total_trials=len(history),
            total_time=total_time,
            improvement=improvement,
            cv_results=searcher.cv_results_
        )
        
        logger.info(
            f"Tuning completed: best_score={best_score:.4f}, "
            f"improvement={improvement:.2f}%, time={total_time:.1f}s"
        )
        
        return result
    
    def _create_base_model(self, model_type: str):
        """ベースモデルを作成"""
        if model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(random_state=self.random_state)
        
        elif model_type == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                random_state=self.random_state,
                verbose=-1
            )
        
        elif model_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBRegressor(
                random_state=self.random_state,
                verbosity=0
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _auto_detect_scoring(self, y: pd.Series) -> str:
        """スコアリング方法を自動判定"""
        # 分類 vs 回帰
        if y.dtype == 'object' or len(y.unique()) < 20:
            # 分類
            return 'accuracy'
        else:
            # 回帰
            return 'r2'
    
    def _extract_history(
        self,
        cv_results: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """CV結果から履歴を抽出"""
        history = []
        
        n_trials = len(cv_results['mean_test_score'])
        
        for i in range(n_trials):
            trial = {
                'iter': i + 1,
                'score': cv_results['mean_test_score'][i],
                'std': cv_results['std_test_score'][i],
                'params': cv_results['params'][i]
            }
            history.append(trial)
        
        # スコア順にソート
        history.sort(key=lambda x: x['score'], reverse=True)
        
        return history
    
    def get_preset_config(
        self,
        level: str = 'fast'
    ) -> Dict[str, Any]:
        """
        プリセット設定を取得
        
        Args:
            level: 'fast', 'balanced', 'thorough'
        
        Returns:
            設定辞書
        """
        presets = {
            'fast': {
                'method': 'random',
                'n_iter': 20,
                'cv': 3,
                'estimated_time': '5-10分'
            },
            'balanced': {
                'method': 'random',
                'n_iter': 50,
                'cv': 5,
                'estimated_time': '15-25分'
            },
            'thorough': {
                'method': 'grid',
                'n_iter': None,  # 全探索
                'cv': 5,
                'estimated_time': '30-60分'
            }
        }
        
        if level not in presets:
            raise ValueError(f"Unknown level: {level}. Choose from {list(presets.keys())}")
        
        return presets[level]
