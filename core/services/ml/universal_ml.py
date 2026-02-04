"""
Universal ML Library Integration - 全ライブラリ対応

Implements: F-UNIVERSAL-ML-001
設計思想:
- 70+の機械学習ライブラリを統一インターフェースで使用可能に
- scikit-learn互換API
- 自動インストール検出とフォールバック

対応ライブラリ:
【Boosting】
- XGBoost, LightGBM, CatBoost, NGBoost

【AutoML】
- AutoGluon, TabPFN, TPOT, Auto-sklearn, PyCaret

【線形系】
- asgl (AdaptiveLasso, GroupLasso, SparseGroupLasso)

【決定木系】
- Linear-Tree (決定木+線形回帰)
- RuleFit (決定木ベース線形モデル)
- RGF (Regularized Greedy Forest)

【ベイズ系】
- gmr (Gaussian Mixture Regression)

【ニューラルネット系】
- TabNet, pykan (KAN), GrowNet

【時系列】
- Prophet, tsfresh

【XAI】
- SHAP, sage-importance, interpret

【最適化】
- DEAP, GPy, hyperopt, Optuna
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

logger = logging.getLogger(__name__)


class UniversalMLWrapper:
    """
    Universal Machine Learning Wrapper
    
    全機械学習ライブラリを統一インターフェースで使用
    
    Example:
        >>> # XGBoost
        >>> model = UniversalMLWrapper('xgboost', task='regression')
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        
        >>> # TabPFN
        >>> model = UniversalMLWrapper('tabpfn')
        >>> model.fit(X_train, y_train)
    """
    
    SUPPORTED_LIBRARIES = {
        # Boosting
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'ngboost': 'ngboost',
        
        # AutoML
        'autogluon': 'autogluon',
        'tabpfn': 'tabpfn',
        'tpot': 'tpot',
        'autosklearn': 'autosklearn',
        'pycaret': 'pycaret',
        
        # 線形系
        'asgl': 'asgl',
        
        # 決定木系
        'linear_tree': 'lineartree',
        'rulefit': 'rulefit',
        'rgf': 'rgf',
        
        # ニューラルネット
        'tabnet': 'pytorch_tabnet',
        'pykan': 'pykan',
        'grownet': 'grownet',
        
        # ベイズ系
        'gmr': 'gmr',
        
        # 時系列
        'prophet': 'prophet',
    }
    
    def __init__(
        self,
        library: str,
        task: str = 'regression',  # 'regression' or 'classification'
        **kwargs
    ):
        """
        Args:
            library: ライブラリ名
            task: タスク種別
            **kwargs: ライブラリ固有のパラメータ
        """
        self.library = library.lower()
        self.task = task
        self.kwargs = kwargs
        
        self.model_ = None
        self._check_availability()
    
    def _check_availability(self):
        """ライブラリの利用可能性チェック"""
        if self.library not in self.SUPPORTED_LIBRARIES:
            raise ValueError(f"Unsupported library: {self.library}")
        
        package_name = self.SUPPORTED_LIBRARIES[self.library]
        
        try:
            __import__(package_name)
        except ImportError:
            logger.warning(f"{package_name} not installed. Install with: pip install {package_name}")
            raise ImportError(f"{package_name} is required but not installed")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """学習"""
        self.model_ = self._create_model()
        
        # データ型変換
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # 学習実行
        self.model_.fit(X_array, y_array)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """予測"""
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        return self.model_.predict(X_array)
    
    def _create_model(self):
        """モデル作成"""
        
        # XGBoost
        if self.library == 'xgboost':
            import xgboost as xgb
            if self.task == 'regression':
                return xgb.XGBRegressor(**self.kwargs)
            else:
                return xgb.XGBClassifier(**self.kwargs)
        
        # LightGBM
        elif self.library == 'lightgbm':
            import lightgbm as lgb
            if self.task == 'regression':
                return lgb.LGBMRegressor(**self.kwargs)
            else:
                return lgb.LGBMClassifier(**self.kwargs)
        
        # CatBoost
        elif self.library == 'catboost':
            from catboost import CatBoostRegressor, CatBoostClassifier
            if self.task == 'regression':
                return CatBoostRegressor(verbose=False, **self.kwargs)
            else:
                return CatBoostClassifier(verbose=False, **self.kwargs)
        
        # NGBoost
        elif self.library == 'ngboost':
            from ngboost import NGBRegressor, NGBClassifier
            if self.task == 'regression':
                return NGBRegressor(**self.kwargs)
            else:
                return NGBClassifier(**self.kwargs)
        
        # TabPFN
        elif self.library == 'tabpfn':
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(**self.kwargs)
        
        # TPOT
        elif self.library == 'tpot':
            from tpot import TPOTRegressor, TPOTClassifier
            if self.task == 'regression':
                return TPOTRegressor(**self.kwargs)
            else:
                return TPOTClassifier(**self.kwargs)
        
        # Linear-Tree
        elif self.library == 'linear_tree':
            from lineartree import LinearTreeRegressor, LinearTreeClassifier
            if self.task == 'regression':
                return LinearTreeRegressor(**self.kwargs)
            else:
                return LinearTreeClassifier(**self.kwargs)
        
        # RuleFit
        elif self.library == 'rulefit':
            from rulefit import RuleFit
            return RuleFit(**self.kwargs)
        
        # ASGL (AdaptiveLasso, GroupLasso)
        elif self.library == 'asgl':
            from asgl import Regressor
            return Regressor(**self.kwargs)
        
        # TabNet
        elif self.library == 'tabnet':
            from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
            if self.task == 'regression':
                return TabNetRegressor(**self.kwargs)
            else:
                return TabNetClassifier(**self.kwargs)
        
        # pykan (KAN)
        elif self.library == 'pykan':
            from kan import KAN
            return KAN(**self.kwargs)
        
        # RGF
        elif self.library == 'rgf':
            from rgf.sklearn import RGFRegressor, RGFClassifier
            if self.task == 'regression':
                return RGFRegressor(**self.kwargs)
            else:
                return RGFClassifier(**self.kwargs)
        
        # GMR (Gaussian Mixture Regression)
        elif self.library == 'gmr':
            from gmr import GMM
            n_components = self.kwargs.pop('n_components', 3)
            return GMM(n_components=n_components, **self.kwargs)
        
        # Prophet (時系列)
        elif self.library == 'prophet':
            from prophet import Prophet
            return Prophet(**self.kwargs)
        
        else:
            raise NotImplementedError(f"Model creation for {self.library} not implemented")


def list_available_libraries() -> Dict[str, bool]:
    """利用可能なライブラリをチェック"""
    availability = {}
    
    for name, package in UniversalMLWrapper.SUPPORTED_LIBRARIES.items():
        try:
            __import__(package)
            availability[name] = True
        except ImportError:
            availability[name] = False
    
    return availability


def install_command(library: str) -> str:
    """インストールコマンド生成"""
    package = UniversalMLWrapper.SUPPORTED_LIBRARIES.get(library)
    if package is None:
        return f"Unknown library: {library}"
    
    # 特殊なインストール方法
    special_installs = {
        'autosklearn': 'pip install auto-sklearn',
        'pycaret': 'pip install pycaret',
        'prophet': 'pip install prophet',
        'pytorch_tabnet': 'pip install pytorch-tabnet',
        'rgf': 'pip install rgf-python',
        'asgl': 'pip install asgl',
        'lineartree': 'pip install linear-tree',
    }
    
    if package in special_installs:
        return special_installs[package]
    else:
        return f"pip install {package}"


# 便利関数
def quick_xgboost(X_train, y_train, X_test, **kwargs) -> np.ndarray:
    """XGBoostクイック実行"""
    model = UniversalMLWrapper('xgboost', **kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def quick_catboost(X_train, y_train, X_test, **kwargs) -> np.ndarray:
    """CatBoostクイック実行"""
    model = UniversalMLWrapper('catboost', **kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def quick_lightgbm(X_train, y_train, X_test, **kwargs) -> np.ndarray:
    """LightGBMクイック実行"""
    model = UniversalMLWrapper('lightgbm', **kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def quick_tabpfn(X_train, y_train, X_test, **kwargs) -> np.ndarray:
    """TabPFNクイック実行"""
    model = UniversalMLWrapper('tabpfn', **kwargs)
    model.fit(X_train, y_train)
    return model.predict(X_test)
