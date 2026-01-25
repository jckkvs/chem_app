"""
mlxtend Ensemble Methods統合

Implements: F-MLXTEND-ENSEMBLE-001
設計思想:
- mlxtendのStacking/Voting機能を統合
- sklearnのStackingVotingより柔軟
- 化学ML向けのアンサンブル戦略

参考文献:
- mlxtend documentation (http://rasbt.github.io/mlxtend/)
- Stacked Generalization (Wolpert, 1992)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

# mlxtendの可用性チェック
try:
    from mlxtend.classifier import EnsembleVoteClassifier, StackingClassifier
    from mlxtend.regressor import StackingRegressor
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    logger.warning("mlxtend not installed. Ensemble features will be limited.")


class MLXtendEnsembles:
    """
    mlxtend Ensemble Methods統合クラス
    
    Features:
    - Stacking（StackingClassifier/StackingRegressor）
    - Voting（EnsembleVoteClassifier）
    - 柔軟なメタラーナー設定
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> base_models = [
        ...     RandomForestClassifier(n_estimators=10),
        ...     LogisticRegression()
        ... ]
        >>> meta_model = LogisticRegression()
        >>> 
        >>> ensemble = MLXtendEnsembles(
        ...     method='stacking',
        ...     base_models=base_models,
        ...     meta_model=meta_model,
        ...     task_type='classification'
        ... )
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        method: str = 'stacking',
        base_models: Optional[List[BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        task_type: str = 'classification',
        voting: str = 'soft',  # for EnsembleVoteClassifier
        weights: Optional[List[float]] = None,
        use_probas: bool = True,  # for Stacking
        use_features_in_secondary: bool = False,
        **kwargs
    ):
        """
        Args:
            method: 'stacking' or 'voting'
            base_models: ベースモデルのリスト
            meta_model: メタモデル（Stackingの場合）
            task_type: 'classification' or 'regression'
            voting: 'soft' or 'hard'（Votingの場合）
            weights: 各モデルの重み
            use_probas: 確率を使うか（Stacking）
            use_features_in_secondary: 元特徴量もメタモデルに渡すか
            **kwargs: その他のパラメータ
        """
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend is not installed. "
                "Install it with: pip install mlxtend"
            )
        
        self.method = method
        self.base_models = base_models or []
        self.meta_model = meta_model
        self.task_type = task_type
        self.voting = voting
        self.weights = weights
        self.use_probas = use_probas
        self.use_features_in_secondary = use_features_in_secondary
        self.kwargs = kwargs
        
        self.ensemble_: Optional[BaseEstimator] = None
    
    def _create_ensemble(self) -> BaseEstimator:
        """アンサンブルモデルを作成"""
        
        if self.method == 'stacking':
            # Stacking
            if self.task_type == 'classification':
                if self.meta_model is None:
                    from sklearn.linear_model import LogisticRegression
                    meta_model = LogisticRegression()
                else:
                    meta_model = self.meta_model
                
                ensemble = StackingClassifier(
                    classifiers=self.base_models,
                    meta_classifier=meta_model,
                    use_probas=self.use_probas,
                    use_features_in_secondary=self.use_features_in_secondary,
                    **self.kwargs
                )
            
            elif self.task_type == 'regression':
                if self.meta_model is None:
                    from sklearn.linear_model import Ridge
                    meta_model = Ridge()
                else:
                    meta_model = self.meta_model
                
                ensemble = StackingRegressor(
                    regressors=self.base_models,
                    meta_regressor=meta_model,
                    use_features_in_secondary=self.use_features_in_secondary,
                    **self.kwargs
                )
            
            else:
                raise ValueError(f"Unknown task_type: {self.task_type}")
        
        elif self.method == 'voting':
            # Voting（分類のみ）
            if self.task_type != 'classification':
                raise ValueError("Voting is only supported for classification")
            
            ensemble = EnsembleVoteClassifier(
                clfs=self.base_models,
                voting=self.voting,
                weights=self.weights,
                **self.kwargs
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return ensemble
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MLXtendEnsembles':
        """
        アンサンブルモデルを学習
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
        
        Returns:
            self
        """
        logger.info(
            f"mlxtend {self.method.capitalize()} training: "
            f"{len(self.base_models)} base models"
        )
        
        self.ensemble_ = self._create_ensemble()
        self.ensemble_.fit(X.values, y.values)
        
        logger.info(f"mlxtend {self.method.capitalize()} training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.ensemble_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.ensemble_.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """確率予測（分類のみ）"""
        if self.ensemble_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only for classification")
        
        return self.ensemble_.predict_proba(X.values)
    
    def get_params(self) -> Dict[str, Any]:
        """パラメータ取得"""
        return {
            'method': self.method,
            'task_type': self.task_type,
            'n_base_models': len(self.base_models),
            'voting': self.voting if self.method == 'voting' else None,
            'use_probas': self.use_probas if self.method == 'stacking' else None,
        }


# =============================================================================
# ヘルパー関数
# =============================================================================

def create_stacking_ensemble(
    base_models: List[BaseEstimator],
    meta_model: Optional[BaseEstimator] = None,
    task_type: str = 'classification',
    use_probas: bool = True,
) -> MLXtendEnsembles:
    """
    Stackingアンサンブルを簡易作成
    
    Args:
        base_models: ベースモデルのリスト
        meta_model: メタモデル
        task_type: タスクタイプ
        use_probas: 確率を使うか
    
    Returns:
        MLXtendEnsembles
    """
    return MLXtendEnsembles(
        method='stacking',
        base_models=base_models,
        meta_model=meta_model,
        task_type=task_type,
        use_probas=use_probas,
    )


def create_voting_ensemble(
    base_models: List[BaseEstimator],
    voting: str = 'soft',
    weights: Optional[List[float]] = None,
) -> MLXtendEnsembles:
    """
    Votingアンサンブルを簡易作成
    
    Args:
        base_models: ベースモデルのリスト
        voting: 'soft' or 'hard'
        weights: 各モデルの重み
    
    Returns:
        MLXtendEnsembles
    """
    return MLXtendEnsembles(
        method='voting',
        base_models=base_models,
        task_type='classification',
        voting=voting,
        weights=weights,
    )


def auto_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'classification',
    method: str = 'stacking',
    n_base_models: int = 3,
) -> Tuple[MLXtendEnsembles, Dict[str, Any]]:
    """
    自動アンサンブル（デフォルトモデルで構築）
    
    Args:
        X: 特徴量DataFrame
        y: ターゲット
        task_type: タスクタイプ
        method: 'stacking' or 'voting'
        n_base_models: ベースモデル数
    
    Returns:
        Tuple[学習済みアンサンブル, 詳細情報]
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    
    # デフォルトベースモデル
    if task_type == 'classification':
        base_models = [
            RandomForestClassifier(n_estimators=50, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000),
            SVC(probability=True, random_state=42),
        ][:n_base_models]
        meta_model = LogisticRegression(random_state=42)
    else:
        base_models = [
            RandomForestRegressor(n_estimators=50, random_state=42),
            Ridge(random_state=42),
            SVR(),
        ][:n_base_models]
        meta_model = Ridge(random_state=42)
    
    ensemble = MLXtendEnsembles(
        method=method,
        base_models=base_models,
        meta_model=meta_model,
        task_type=task_type,
    )
    
    ensemble.fit(X, y)
    
    info = {
        'method': method,
        'n_base_models': len(base_models),
        'base_model_types': [type(m).__name__ for m in base_models],
        'meta_model_type': type(meta_model).__name__ if method == 'stacking' else None,
    }
    
    return ensemble, info
