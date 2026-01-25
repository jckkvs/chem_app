"""
XGBoost全sklearn-APIクラス統一ラッパー

メンテナブル設計:
- 全引数をユーザーが設定可能（**kwargs透過）
- 250行程度に抑制
- 単一責任: XGBoostのみ担当

Implements: XGBoost全6種類の sklearn-API完全対応
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

# XGBoost可用性チェック
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed")


class XGBoostWrapper:
    """
    XGBoost全sklearn-APIクラスの統一ラッパー
    
    全6種類のXGBoostモデルをサポート:
    - XGBClassifier: 分類（Gradient Boosting）
    - XGBRegressor: 回帰（Gradient Boosting）
    - XGBRFClassifier: 分類（Random Forest様式）
    - XGBRFRegressor: 回帰（Random Forest様式）
    - XGBRanker: ランキング学習
    - (XGBModel): 低レベルAPI（上級者向け）
    
    全引数カスタマイズ可能:
    - **kwargsで任意のXGBoostパラメータを渡せる
    - GPU/分散学習も対応
    
    Example:
        >>> # 基本
        >>> xgb_wrapper = XGBoostWrapper(
        ...     model_type='classifier',
        ...     n_estimators=100,
        ...     max_depth=6,
        ...     learning_rate=0.1
        ... )
        >>> xgb_wrapper.fit(X_train, y_train)
        >>> 
        >>> # GPU利用
        >>> xgb_wrapper = XGBoostWrapper(
        ...     model_type='regressor',
        ...     tree_method='gpu_hist',
        ...     gpu_id=0
        ... )
        >>> 
        >>> # Random Forest様式
        >>> xgb_wrapper = XGBoostWrapper(
        ...     model_type='rf_classifier',
        ...     n_estimators=100,
        ...     subsample=0.8,
        ...     colsample_bynode=0.8
        ... )
    """
    
    def __init__(
        self,
        model_type: Literal['classifier', 'regressor', 'rf_classifier', 'rf_regressor', 'ranker'] = 'classifier',
        **params
    ):
        """
        Args:
            model_type: XGBoostモデルタイプ
            **params: XGBoost全パラメータ（透過的に渡される）
                - n_estimators: ブースティングラウンド数
                - max_depth: 木の最大深さ
                - learning_rate (eta): 学習率
                - subsample: サンプリング比率
                - colsample_bytree: 特徴量サンプリング比率
                - tree_method: 'auto', 'exact', 'approx', 'hist', 'gpu_hist'
                - gpu_id: GPU ID
                - その他100+パラメータ全て対応
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install: pip install xgboost")
        
        self.model_type = model_type
        self.params = params
        self.model_: Optional[BaseEstimator] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
        **fit_params
    ) -> 'XGBoostWrapper':
        """
        学習
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
            eval_set: 評価セット（early stopping用）
            early_stopping_rounds: early stopping rounds
            verbose: ログ出力
            **fit_params: その他fitパラメータ
        
        Returns:
            self
        """
        # モデル作成
        self.model_ = self._create_model()
        
        logger.info(f"XGBoost学習開始: type={self.model_type}")
        
        # fit引数準備
        fit_args = {'verbose': verbose, **fit_params}
        
        if eval_set is not None:
            fit_args['eval_set'] = eval_set
        
        if early_stopping_rounds is not None:
            fit_args['early_stopping_rounds'] = early_stopping_rounds
        
        self.model_.fit(X.values, y.values, **fit_args)
        
        logger.info("XGBoost学習完了")
        
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
    
    def get_feature_importance(
        self,
        importance_type: Literal['weight', 'gain', 'cover', 'total_gain', 'total_cover'] = 'weight'
    ) -> pd.Series:
        """
        特徴重要度取得
        
        Args:
            importance_type: 重要度タイプ
                - weight: 特徴が使われた回数
                - gain: 平均ゲイン
                - cover: 平均カバレッジ
                - total_gain: 合計ゲイン
                - total_cover: 合計カバレッジ
        
        Returns:
            特徴重要度Series
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        
        importance = self.model_.get_booster().get_score(importance_type=importance_type)
        
        return pd.Series(importance)
    
    def _create_model(self) -> BaseEstimator:
        """モデルタイプに応じてモデル作成"""
        
        if self.model_type == 'classifier':
            return xgb.XGBClassifier(**self.params)
        
        elif self.model_type == 'regressor':
            return xgb.XGBRegressor(**self.params)
        
        elif self.model_type == 'rf_classifier':
            # Random Forest様式（デフォルト設定）
            default_rf_params = {
                'learning_rate': 1.0,
                'subsample': 0.8,
                'colsample_bynode': 0.8,
                'num_parallel_tree': 100,
            }
            # ユーザー指定パラメータで上書き
            rf_params = {**default_rf_params, **self.params}
            return xgb.XGBRFClassifier(**rf_params)
        
        elif self.model_type == 'rf_regressor':
            default_rf_params = {
                'learning_rate': 1.0,
                'subsample': 0.8,
                'colsample_bynode': 0.8,
                'num_parallel_tree': 100,
            }
            rf_params = {**default_rf_params, **self.params}
            return xgb.XGBRFRegressor(**rf_params)
        
        elif self.model_type == 'ranker':
            return xgb.XGBRanker(**self.params)
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """利用可能なモデル一覧"""
        return ['classifier', 'regressor', 'rf_classifier', 'rf_regressor', 'ranker']
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        info = {
            'model_type': self.model_type,
            'model_class': type(self.model_).__name__ if self.model_ else 'None',
            'params': self.params,
        }
        
        if self.model_ is not None and hasattr(self.model_, 'get_params'):
            info['current_params'] = self.model_.get_params()
        
        return info


# =============================================================================
# ヘルパー関数
# =============================================================================

def create_xgboost_model(
    task_type: Literal['classification', 'regression', 'ranking'] = 'classification',
    use_rf_style: bool = False,
    **params
) -> XGBoostWrapper:
    """
    XGBoostモデル簡易作成
    
    Args:
        task_type: タスクタイプ
        use_rf_style: Random Forest様式を使用
        **params: 全XGBoostパラメータ
    
    Returns:
        XGBoostWrapper
    
    Example:
        >>> # 分類（通常）
        >>> model = create_xgboost_model('classification', n_estimators=100)
        >>> 
        >>> # 回帰（RF様式）
        >>> model = create_xgboost_model('regression', use_rf_style=True)
        >>> 
        >>> # GPU利用
        >>> model = create_xgboost_model(
        ...     'classification',
        ...     tree_method='gpu_hist',
        ...     gpu_id=0
        ... )
    """
    # モデルタイプ決定
    if task_type == 'classification':
        model_type = 'rf_classifier' if use_rf_style else 'classifier'
    elif task_type == 'regression':
        model_type = 'rf_regressor' if use_rf_style else 'regressor'
    elif task_type == 'ranking':
        model_type = 'ranker'
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    return XGBoostWrapper(model_type=model_type, **params)
