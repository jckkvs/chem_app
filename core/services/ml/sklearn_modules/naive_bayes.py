"""
sklearn.naive_bayes統一ラッパー

メンテナブル設計:
- 単一責任: naive_bayesモジュールのみ担当
- 200行程度に抑制
- 統一的なAPI提供

Implements: sklearn.naive_bayes完全対応
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class NaiveBayesWrapper:
    """
    sklearn.naive_bayesの統一ラッパー
    
    全5種類のNaive Bayesモデルをサポート:
    - GaussianNB: 連続値（正規分布仮定）
    - MultinomialNB: 離散カウント（文書分類等）
    - ComplementNB: 不均衡データ向けMultinomial
    - BernoulliNB: 二値特徴量
    - CategoricalNB: カテゴリ変数
    
    Example:
        >>> # 自動選択
        >>> nb = NaiveBayesWrapper(model_type='auto')
        >>> nb.fit(X_train, y_train)
        >>> predictions = nb.predict(X_test)
        >>> 
        >>> # 手動選択
        >>> nb = NaiveBayesWrapper(model_type='gaussian')
        >>> nb.fit(X_train, y_train)
    """
    
    def __init__(
        self,
        model_type: Literal['auto', 'gaussian', 'multinomial', 'complement', 'bernoulli', 'categorical'] = 'auto',
        **params
    ):
        """
        Args:
            model_type: Naive Bayesモデルタイプ
            **params: モデル固有のパラメータ
        """
        self.model_type = model_type
        self.params = params
        self.model_: Optional[BaseEstimator] = None
        self.selected_model_type_: Optional[str] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveBayesWrapper':
        """
        学習
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
        
        Returns:
            self
        """
        # モデルタイプ自動選択
        if self.model_type == 'auto':
            self.selected_model_type_ = self._auto_select_model_type(X)
        else:
            self.selected_model_type_ = self.model_type
        
        # モデル作成
        self.model_ = self._create_model(self.selected_model_type_)
        
        logger.info(f"NaiveBayes学習開始: type={self.selected_model_type_}")
        self.model_.fit(X.values, y.values)
        logger.info("NaiveBayes学習完了")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """確率予測"""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        return self.model_.predict_proba(X.values)
    
    def _auto_select_model_type(self, X: pd.DataFrame) -> str:
        """データに基づいてモデルタイプ自動選択"""
        
        # 二値特徴のみ → Bernoulli
        if self._is_binary(X):
            return 'bernoulli'
        
        # 非負整数のみ → Multinomial
        if self._is_nonnegative_integer(X):
            return 'multinomial'
        
        # カテゴリ変数 → Categorical
        if self._is_categorical(X):
            return 'categorical'
        
        # デフォルト: Gaussian（連続値）
        return 'gaussian'
    
    def _is_binary(self, X: pd.DataFrame) -> bool:
        """全カラムが二値か判定"""
        for col in X.columns:
            unique_vals = X[col].dropna().unique()
            if len(unique_vals) > 2:
                return False
            if not set(unique_vals).issubset({0, 1, True, False}):
                return False
        return True
    
    def _is_nonnegative_integer(self, X: pd.DataFrame) -> bool:
        """全カラムが非負整数か判定"""
        for col in X.columns:
            if not np.all(X[col] >= 0):
                return False
            if not np.all(X[col] == X[col].astype(int)):
                return False
        return True
    
    def _is_categorical(self, X: pd.DataFrame) -> bool:
        """カテゴリ変数か判定"""
        return X.dtypes.apply(lambda x: x == 'object' or x.name == 'category').all()
    
    def _create_model(self, model_type: str) -> BaseEstimator:
        """モデルタイプに応じてモデル作成"""
        
        if model_type == 'gaussian':
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(**self.params)
        
        elif model_type == 'multinomial':
            from sklearn.naive_bayes import MultinomialNB
            return MultinomialNB(**self.params)
        
        elif model_type == 'complement':
            from sklearn.naive_bayes import ComplementNB
            return ComplementNB(**self.params)
        
        elif model_type == 'bernoulli':
            from sklearn.naive_bayes import BernoulliNB
            return BernoulliNB(**self.params)
        
        elif model_type == 'categorical':
            from sklearn.naive_bayes import CategoricalNB
            return CategoricalNB(**self.params)
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """利用可能なモデル一覧"""
        return ['gaussian', 'multinomial', 'complement', 'bernoulli', 'categorical']
    
    def get_model_info(self) -> Dict[str, str]:
        """モデル情報取得"""
        return {
            'model_type': self.selected_model_type_ or 'not fitted',
            'model_class': type(self.model_).__name__ if self.model_ else 'None',
        }


# =============================================================================
# ヘルパー関数
# =============================================================================

def auto_select_naive_bayes(
    X: pd.DataFrame,
    y: pd.Series,
    **params
) -> NaiveBayesWrapper:
    """
    データに最適なNaive Bayesを自動選択・学習
    
    Args:
        X: 特徴量
        y: ターゲット
        **params: モデルパラメータ
    
    Returns:
        学習済みNaiveBayesWrapper
    
    Example:
        >>> model = auto_select_naive_bayes(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    nb = NaiveBayesWrapper(model_type='auto', **params)
    nb.fit(X, y)
    
    logger.info(f"自動選択結果: {nb.selected_model_type_}")
    
    return nb
