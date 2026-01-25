"""
sklearn.neural_network統一ラッパー

メンテナブル設計:
- 全引数カスタマイズ可能（**kwargs透過）
- 250行程度に抑制
- 単一責任: neural_networkのみ担当

Implements: sklearn.neural_network完全対応
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class NeuralNetworkWrapper:
    """
    sklearn.neural_networkの統一ラッパー
    
    全3種類のニューラルネットワークモデルをサポート:
    - MLPClassifier: 多層パーセプトロン分類器
    - MLPRegressor: 多層パーセプトロン回帰器
    - BernoulliRBM: 制限ボルツマンマシン（特徴抽出用）
    
    全引数カスタマイズ可能（**kwargs透過）
    
    Example:
        >>> # MLPClassifier
        >>> nn = NeuralNetworkWrapper(
        ...     model_type='classifier',
        ...     hidden_layer_sizes=(100, 50),
        ...     activation='relu',
        ...     learning_rate_init=0.001,
        ...     max_iter=200
        ... )
        >>> nn.fit(X_train, y_train)
        >>> 
        >>> # MLPRegressor
        >>> nn = NeuralNetworkWrapper(
        ...     model_type='regressor',
        ...     hidden_layer_sizes=(64, 32),
        ...     solver='adam',
        ...     early_stopping=True
        ... )
    """
    
    def __init__(
        self,
        model_type: Literal['classifier', 'regressor', 'rbm'] = 'classifier',
        **params
    ):
        """
        Args:
            model_type: モデルタイプ
            **params: sklearn.neural_network全パラメータ
                - hidden_layer_sizes: 隠れ層のサイズ (default: (100,))
                - activation: 活性化関数 ('relu', 'tanh', 'logistic')
                - solver: 最適化アルゴリズム ('adam', 'sgd', 'lbfgs')
                - alpha: L2正則化項
                - learning_rate_init: 初期学習率
                - max_iter: 最大イテレーション数
                - early_stopping: early stopping使用
                - その他全パラメータ対応
        """
        self.model_type = model_type
        self.params = params
        self.model_: Optional[BaseEstimator] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> 'NeuralNetworkWrapper':
        """
        学習
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
        
        Returns:
            self
        """
        # モデル作成
        self.model_ = self._create_model()
        
        logger.info(f"NeuralNetwork学習開始: type={self.model_type}")
        self.model_.fit(X.values, y.values)
        logger.info(f"NeuralNetwork学習完了: n_iter={self.model_.n_iter_}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """確率予測（分類のみ）"""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        if not hasattr(self.model_, 'predict_proba'):
            raise ValueError(f"{self.model_type} does not support predict_proba")
        
        return self.model_.predict_proba(X.values)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """変換（RBMのみ）"""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        if not hasattr(self.model_, 'transform'):
            raise ValueError(f"{self.model_type} does not support transform")
        
        return self.model_.transform(X.values)
    
    def _create_model(self) -> BaseEstimator:
        """モデルタイプに応じてモデル作成"""
        
        if self.model_type == 'classifier':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**self.params)
        
        elif self.model_type == 'regressor':
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(**self.params)
        
        elif self.model_type == 'rbm':
            from sklearn.neural_network import BernoulliRBM
            return BernoulliRBM(**self.params)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """利用可能なモデル一覧"""
        return ['classifier', 'regressor', 'rbm']
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        info = {
            'model_type': self.model_type,
            'model_class': type(self.model_).__name__ if self.model_ else 'None',
            'params': self.params,
        }
        
        if self.model_ is not None:
            if hasattr(self.model_, 'n_iter_'):
                info['n_iterations'] = self.model_.n_iter_
            if hasattr(self.model_, 'loss_'):
                info['final_loss'] = self.model_.loss_
            if hasattr(self.model_, 'best_loss_'):
                info['best_loss'] = self.model_.best_loss_
        
        return info


# =============================================================================
# ヘルパー関数
# =============================================================================

def create_mlp_model(
    task_type: Literal['classification', 'regression'] = 'classification',
    hidden_layers: tuple = (100, 50),
    **params
) -> NeuralNetworkWrapper:
    """
    MLP（多層パーセプトロン）モデル簡易作成
    
    Args:
        task_type: タスクタイプ
        hidden_layers: 隠れ層のサイ
        **params: その他パラメータ
    
    Returns:
        NeuralNetworkWrapper
    
    Example:
        >>> # 2層MLP分類器
        >>> model = create_mlp_model('classification', (128, 64), max_iter=200)
        >>> 
        >>> # 3層MLP回帰器、early stopping有効
        >>> model = create_mlp_model(
        ...     'regression',
        ...     (100, 50, 25),
        ...     early_stopping=True,
        ...     validation_fraction=0.1
        ... )
    """
    model_type = 'classifier' if task_type == 'classification' else 'regressor'
    
    return NeuralNetworkWrapper(
        model_type=model_type,
        hidden_layer_sizes=hidden_layers,
        **params
    )
