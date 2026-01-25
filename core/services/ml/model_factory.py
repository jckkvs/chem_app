"""
拡張可能なMLモデルファクトリー - マイナーライブラリ統合版

Implements: F-ML-FACTORY-001
設計思想:
- 多様なMLライブラリ（sklearn, XGBoost, LightGBM, CatBoost, imodels, rgf, linear-tree等）の統合
- 各ライブラリがインストールされていない場合のgraceful fallback
- 統一的なAPI（fit, predict, feature_importances_）
- OptunaSearchCVによるハイパーパラメータ最適化対応

参考文献:
- imodels: Interpretable ML Package (https://github.com/csinva/imodels)
- RGF: Regularized Greedy Forest (https://github.com/RGF-team/rgf)
- linear-tree: Linear Tree Models (https://github.com/cerlymarco/linear-tree)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)

# =============================================================================
# ライブラリ可用性チェック
# =============================================================================

def is_library_available(library_name: str) -> bool:
    """ライブラリがインストールされているかチェック"""
    try:
        __import__(library_name)
        return True
    except ImportError:
        return False


# 各ライブラリの可用性
AVAILABLE_LIBRARIES = {
    # Core ML
    'sklearn': is_library_available('sklearn'),
    'xgboost': is_library_available('xgboost'),
    'lightgbm': is_library_available('lightgbm'),
    'catboost': is_library_available('catboost'),
    
    # Interpretable ML
    'imodels': is_library_available('imodels'),
    'wittgenstein': is_library_available('wittgenstein'),
    'corels': is_library_available('corels'),
    'skrules': is_library_available('skrules'),  # skope-rules
    'anchor': is_library_available('anchor'),
    'alibi': is_library_available('alibi'),
    
    # Tree Models
    'rgf': is_library_available('rgf'),
    'linear_tree': is_library_available('lineartree'),
    'node': is_library_available('node'),
    'tabnet': is_library_available('pytorch_tabnet'),
    
    # GAM (Generalized Additive Models)
    'pygam': is_library_available('pygam'),
    'interpret': is_library_available('interpret'),  # Microsoft InterpretML (EBM)
    
    # Causal Inference
    'econml': is_library_available('econml'),
    'causalml': is_library_available('causalml'),
    'dowhy': is_library_available('dowhy'),
    
    # Deep Learning (Tabular)
    'pytorch_tabular': is_library_available('pytorch_tabular'),
    'rtdl': is_library_available('rtdl'),
    
    # Other
    'ngboost': is_library_available('ngboost'),
    'optuna': is_library_available('optuna'),
}


# =============================================================================
# モデルレジストリ - タスクタイプ × モデル名
# =============================================================================

class ModelRegistry:
    """モデルレジストリ - 利用可能なモデルの管理"""
    
    def __init__(self):
        self._registry: Dict[Tuple[str, str], Type[BaseEstimator]] = {}
        self._default_params: Dict[str, Dict[str, Any]] = {}
        self._initialize_registry()
    
    def _initialize_registry(self):
        """モデルレジストリを初期化"""
        
        # ===== Scikit-learn (常に利用可能) =====
        if AVAILABLE_LIBRARIES['sklearn']:
            from sklearn.ensemble import (
                AdaBoostClassifier,
                AdaBoostRegressor,
                BaggingClassifier,
                BaggingRegressor,
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )
            from sklearn.linear_model import (
                ElasticNet,
                Lasso,
                LinearRegression,
                LogisticRegression,
                Ridge,
            )
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.svm import SVC, SVR
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            
            # 線形モデル
            self.register('regression', 'linear', LinearRegression, {})
            self.register('regression', 'ridge', Ridge, {'alpha': 1.0, 'random_state': 42})
            self.register('regression', 'lasso', Lasso, {'alpha': 1.0, 'random_state': 42})
            self.register('regression', 'elasticnet', ElasticNet, {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': 42})
            self.register('classification', 'logistic', LogisticRegression, {'random_state': 42, 'max_iter': 1000})
            
            # ツリーモデル
            self.register('regression', 'decision_tree', DecisionTreeRegressor, {'random_state': 42})
            self.register('classification', 'decision_tree', DecisionTreeClassifier, {'random_state': 42})
            
            # アンサンブル
            self.register('regression', 'random_forest', RandomForestRegressor, {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1})
            self.register('classification', 'random_forest', RandomForestClassifier, {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1})
            
            self.register('regression', 'extra_trees', ExtraTreesRegressor, {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1})
            self.register('classification', 'extra_trees', ExtraTreesClassifier, {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1})
            
            self.register('regression', 'gbm', GradientBoostingRegressor, {'n_estimators': 100, 'random_state': 42})
            self.register('classification', 'gbm', GradientBoostingClassifier, {'n_estimators': 100, 'random_state': 42})
            
            self.register('regression', 'adaboost', AdaBoostRegressor, {'n_estimators': 50, 'random_state': 42})
            self.register('classification', 'adaboost', AdaBoostClassifier, {'n_estimators': 50, 'random_state': 42})
            
            self.register('regression', 'bagging', BaggingRegressor, {'n_estimators': 10, 'random_state': 42, 'n_jobs': -1})
            self.register('classification', 'bagging', BaggingClassifier, {'n_estimators': 10, 'random_state': 42, 'n_jobs': -1})
            
            # その他
            self.register('regression', 'knn', KNeighborsRegressor, {'n_neighbors': 5, 'n_jobs': -1})
            self.register('classification', 'knn', KNeighborsClassifier, {'n_neighbors': 5, 'n_jobs': -1})
            
            self.register('regression', 'svr', SVR, {'kernel': 'rbf', 'C': 1.0})
            self.register('classification', 'svc', SVC, {'kernel': 'rbf', 'C': 1.0, 'probability': True, 'random_state': 42})
        
        # ===== XGBoost =====
        if AVAILABLE_LIBRARIES['xgboost']:
            from xgboost import XGBClassifier, XGBRegressor
            
            self.register('regression', 'xgboost', XGBRegressor, {'random_state': 42, 'n_jobs': -1})
            self.register('regression', 'xgb', XGBRegressor, {'random_state': 42, 'n_jobs': -1})
            self.register('classification', 'xgboost', XGBClassifier, {'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'})
            self.register('classification', 'xgb', XGBClassifier, {'random_state': 42, 'n_jobs': -1, 'eval_metric': 'logloss'})
        
        # ===== LightGBM =====
        if AVAILABLE_LIBRARIES['lightgbm']:
            import lightgbm as lgb
            
            self.register('regression', 'lightgbm', lgb.LGBMRegressor, {'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            self.register('regression', 'lgbm', lgb.LGBMRegressor, {'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            self.register('classification', 'lightgbm', lgb.LGBMClassifier, {'random_state': 42, 'n_jobs': -1, 'verbose': -1})
            self.register('classification', 'lgbm', lgb.LGBMClassifier, {'random_state': 42, 'n_jobs': -1, 'verbose': -1})
        
        # ===== CatBoost =====
        if AVAILABLE_LIBRARIES['catboost']:
            from catboost import CatBoostClassifier, CatBoostRegressor
            
            self.register('regression', 'catboost', CatBoostRegressor, {'random_state': 42, 'verbose': 0})
            self.register('classification', 'catboost', CatBoostClassifier, {'random_state': 42, 'verbose': 0})
        
        # ===== imodels (解釈可能ML) =====
        if AVAILABLE_LIBRARIES['imodels']:
            try:
                from imodels import (
                    BayesianRuleListClassifier,
                    GreedyRuleListClassifier,
                    RuleFitRegressor,
                    SkopeRulesClassifier,
                )
                
                self.register('regression', 'rulefit', RuleFitRegressor, {'random_state': 42})
                self.register('classification', 'bayesian_rule_list', BayesianRuleListClassifier, {})
                self.register('classification', 'greedy_rule_list', GreedyRuleListClassifier, {'random_state': 42})
                self.register('classification', 'skope_rules', SkopeRulesClassifier, {'random_state': 42})
                
                logger.info("imodels統合成功: RuleFit, BayesianRuleList, GreedyRuleList, SkopeRules")
            except ImportError as e:
                logger.warning(f"imodels一部モデルのインポート失敗: {e}")
        
        # ===== RGF (Regularized Greedy Forest) =====
        if AVAILABLE_LIBRARIES['rgf']:
            try:
                from rgf.sklearn import RGFClassifier, RGFRegressor
                
                self.register('regression', 'rgf', RGFRegressor, {'max_leaf': 1000, 'test_interval': 100})
                self.register('classification', 'rgf', RGFClassifier, {'max_leaf': 1000, 'test_interval': 100})
                
                logger.info("RGF統合成功: RGFRegressor, RGFClassifier")
            except ImportError as e:
                logger.warning(f"RGFインポート失敗: {e}")
        
        # ===== linear-tree =====
        if AVAILABLE_LIBRARIES['linear_tree']:
            try:
                from lineartree import LinearTreeClassifier, LinearTreeRegressor
                
                self.register('regression', 'linear_tree', LinearTreeRegressor, {'base_estimator': None})
                self.register('classification', 'linear_tree', LinearTreeClassifier, {'base_estimator': None})
                
                logger.info("linear-tree統合成功: LinearTreeRegressor, LinearTreeClassifier")
            except ImportError as e:
                logger.warning(f"linear-treeインポート失敗: {e}")
        
        # ===== NGBoost (確率的勾配ブースティング) =====
        if AVAILABLE_LIBRARIES['ngboost']:
            try:
                from ngboost import NGBClassifier, NGBRegressor
                
                self.register('regression', 'ngboost', NGBRegressor, {'random_state': 42})
                self.register('classification', 'ngboost', NGBClassifier, {'random_state': 42})
                
                logger.info("NGBoost統合成功: NGBRegressor, NGBClassifier")
            except ImportError as e:
                logger.warning(f"NGBoostインポート失敗: {e}")
        
        # ===== wittgenstein (RIPPER/IREP) =====
        if AVAILABLE_LIBRARIES['wittgenstein']:
            try:
                from wittgenstein import RIPPER
                
                self.register('classification', 'ripper', RIPPER, {'random_state': 42})
                
                logger.info("wittgenstein統合成功: RIPPER")
            except ImportError as e:
                logger.warning(f"wittgensteinインポート失敗: {e}")
        
        # ===== CORELS =====
        if AVAILABLE_LIBRARIES['corels']:
            try:
                # CORESは分類のみサポート
                # Python bindingsが必要
                logger.info("CORELS検出されましたが、Python bindingsの確認が必要です")
            except Exception as e:
                logger.warning(f"CORELSインポート失敗: {e}")
        
        # ===== PyGAM (一般化加法モデル) =====
        if AVAILABLE_LIBRARIES['pygam']:
            try:
                from pygam import LogisticGAM, LinearGAM
                
                self.register('regression', 'linear_gam', LinearGAM, {})
                self.register('classification', 'logistic_gam', LogisticGAM, {})
                
                logger.info("PyGAM統合成功: LinearGAM, LogisticGAM")
            except ImportError as e:
                logger.warning(f"PyGAMインポート失敗: {e}")
        
        # ===== InterpretML (EBM - Explainable Boosting Machine) =====
        if AVAILABLE_LIBRARIES['interpret']:
            try:
                from interpret.glassbox import (
                    ExplainableBoostingClassifier,
                    ExplainableBoostingRegressor,
                )
                
                self.register('regression', 'ebm', ExplainableBoostingRegressor, {'random_state': 42})
                self.register('regression', 'explainable_boosting', ExplainableBoostingRegressor, {'random_state': 42})
                self.register('classification', 'ebm', ExplainableBoostingClassifier, {'random_state': 42})
                self.register('classification', 'explainable_boosting', ExplainableBoostingClassifier, {'random_state': 42})
                
                logger.info("InterpretML統合成功: EBM (Explainable Boosting Machine)")
            except ImportError as e:
                logger.warning(f"InterpretMLインポート失敗: {e}")
        
        # ===== PyTorch TabNet =====
        if AVAILABLE_LIBRARIES['tabnet']:
            try:
                from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
                
                self.register('regression', 'tabnet', TabNetRegressor, {'seed': 42})
                self.register('classification', 'tabnet', TabNetClassifier, {'seed': 42})
                
                logger.info("PyTorch TabNet統合成功: TabNetRegressor, TabNetClassifier")
            except ImportError as e:
                logger.warning(f"PyTorch TabNetインポート失敗: {e}")
        
        # ===== NODE (Neural Oblivious Decision Ensembles) =====
        if AVAILABLE_LIBRARIES['node']:
            try:
                # NODEは複雑なため、基本的な設定のみ
                logger.info("NODE検出されましたが、カスタム統合が必要です")
            except Exception as e:
                logger.warning(f"NODEインポート失敗: {e}")
        
        # ===== EconML (Microsoft因果推論) =====
        if AVAILABLE_LIBRARIES['econml']:
            try:
                from econml.dml import LinearDML, CausalForestDML
                
                # 因果推論モデルは特殊なため、'causal'タスクタイプで登録
                self.register('causal', 'linear_dml', LinearDML, {'random_state': 42})
                self.register('causal', 'causal_forest_dml', CausalForestDML, {'random_state': 42})
                
                logger.info("EconML統合成功: LinearDML, CausalForestDML")
            except ImportError as e:
                logger.warning(f"EconMLインポート失敗: {e}")
        
        # ===== CausalML (Uber因果推論) =====
        if AVAILABLE_LIBRARIES['causalml']:
            try:
                # CausalMLは多様なメタラーナーを提供
                logger.info("CausalML検出されました（カスタム統合が推奨）")
            except Exception as e:
                logger.warning(f"CausalMLインポート失敗: {e}")
        
        # ===== DoWhy (因果グラフ推論) =====
        if AVAILABLE_LIBRARIES['dowhy']:
            try:
                # DoWhyは因果グラフベースのフレームワーク
                logger.info("DoWhy検出されました（因果グラフ推論用）")
            except Exception as e:
                logger.warning(f"DoWhyインポート失敗: {e}")
        
        # ===== PyTorch Tabular =====
        if AVAILABLE_LIBRARIES['pytorch_tabular']:
            try:
                # PyTorch Tabularは設定が複雑なため、基本情報のみ
                logger.info("PyTorch Tabular検出されました（詳細設定が必要）")
            except Exception as e:
                logger.warning(f"PyTorch Tabularインポート失敗: {e}")
        
        # ===== RTDL (Research Tabular Deep Learning) =====
        if AVAILABLE_LIBRARIES['rtdl']:
            try:
                # RTDLは研究用の先端DLモデル群
                logger.info("RTDL検出されました（研究用DLモデル）")
            except Exception as e:
                logger.warning(f"RTDLインポート失敗: {e}")

    
    def register(
        self, 
        task_type: str, 
        model_name: str, 
        model_class: Type[BaseEstimator],
        default_params: Dict[str, Any]
    ):
        """モデルを登録"""
        key = (task_type, model_name)
        self._registry[key] = model_class
        self._default_params[model_name] = default_params
    
    def get(self, task_type: str, model_name: str) -> Optional[Type[BaseEstimator]]:
        """モデルクラスを取得"""
        key = (task_type, model_name)
        return self._registry.get(key)
    
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """デフォルトパラメータを取得"""
        return self._default_params.get(model_name, {}).copy()
    
    def list_models(self, task_type: Optional[str] = None) -> List[str]:
        """利用可能なモデル一覧"""
        if task_type:
            return [name for (tt, name) in self._registry.keys() if tt == task_type]
        else:
            return list(set(name for (_, name) in self._registry.keys()))
    
    def is_available(self, task_type: str, model_name: str) -> bool:
        """モデルが利用可能かチェック"""
        return (task_type, model_name) in self._registry


# グローバルレジストリインスタンス
_global_registry = ModelRegistry()


# =============================================================================
# モデルファクトリー
# =============================================================================

class ModelFactory:
    """
    MLモデルファクトリー - 統一的なモデル生成インターフェース
    
    Features:
    - 多様なライブラリのサポート（sklearn, XGBoost, LightGBM, CatBoost, imodels, rgf, linear-tree等）
    - ライブラリが利用不可の場合のフォールバック
    - デフォルトパラメータの自動設定
    - カスタムパラメータのオーバーライド
    
    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create('regression', 'xgboost', n_estimators=200)
        >>> model.fit(X, y)
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or _global_registry
    
    def create(
        self,
        task_type: Literal['regression', 'classification'],
        model_name: str,
        **custom_params
    ) -> BaseEstimator:
        """
        モデルを生成
        
        Args:
            task_type: タスクタイプ ('regression' or 'classification')
            model_name: モデル名 ('xgboost', 'lightgbm', 'random_forest', 'rulefit', 'rgf', etc.)
            **custom_params: カスタムパラメータ（デフォルトをオーバーライド）
        
        Returns:
            BaseEstimator: モデルインスタンス
        
        Raises:
            ValueError: モデルが利用不可の場合
        """
        model_class = self.registry.get(task_type, model_name)
        
        if model_class is None:
            available = self.registry.list_models(task_type)
            raise ValueError(
                f"モデル '{model_name}' (タスク: {task_type}) は利用不可です。\n"
                f"利用可能なモデル: {available}"
            )
        
        # デフォルトパラメータ + カスタムパラメータ
        params = self.registry.get_default_params(model_name)
        params.update(custom_params)
        
        logger.info(f"モデル作成: {task_type}/{model_name} with params={params}")
        return model_class(**params)
    
    def list_available_models(
        self, 
        task_type: Optional[str] = None
    ) -> List[str]:
        """利用可能なモデル一覧"""
        return self.registry.list_models(task_type)
    
    def get_library_status(self) -> Dict[str, bool]:
        """各ライブラリの利用可能状況"""
        return AVAILABLE_LIBRARIES.copy()


# =============================================================================
# HPO対応ファクトリー (Optuna統合)
# =============================================================================

def create_model_with_optuna(
    task_type: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = 42,
    **fixed_params
) -> BaseEstimator:
    """
    Optunaでハイパーパラメータ最適化されたモデルを作成
    
    Args:
        task_type: タスクタイプ
        model_name: モデル名
        X: 特徴量
        y: ターゲット
        n_trials: 試行回数
        cv: 交差検証フォールド数
        random_state: 乱数シード
        **fixed_params: 固定パラメータ
    
    Returns:
        BaseEstimator: 最適化されたモデル
    """
    if not AVAILABLE_LIBRARIES['optuna']:
        logger.warning("Optunaが利用不可のため、デフォルトパラメータでモデルを作成します")
        factory = ModelFactory()
        return factory.create(task_type, model_name, **fixed_params)
    
    try:
        import optuna
        from optuna.integration import OptunaSearchCV
        from sklearn.model_selection import KFold, StratifiedKFold
        
        # ベースモデル取得
        factory = ModelFactory()
        base_model = factory.create(task_type, model_name, **fixed_params)
        
        # パラメータ空間の定義（モデル別）
        param_distributions = _get_param_distributions(model_name)
        
        if not param_distributions:
            logger.warning(f"モデル '{model_name}' のパラメータ空間が未定義のため、デフォルトパラメータを使用")
            return base_model
        
        # CV分割
        if task_type == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # OptunaSearchCV
        optuna_search = OptunaSearchCV(
            base_model,
            param_distributions,
            n_trials=n_trials,
            cv=cv_splitter,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )
        
        logger.info(f"Optuna HPO開始: {n_trials} trials, {cv}-fold CV")
        optuna_search.fit(X, y)
        
        logger.info(f"Best score: {optuna_search.best_score_:.4f}")
        logger.info(f"Best params: {optuna_search.best_params_}")
        
        return optuna_search.best_estimator_
        
    except Exception as e:
        logger.error(f"Optuna HPO失敗: {e}、デフォルトモデルを返します")
        factory = ModelFactory()
        return factory.create(task_type, model_name, **fixed_params)


def _get_param_distributions(model_name: str) -> Dict[str, Any]:
    """モデル別のOptunaパラメータ分布を取得"""
    import optuna
    
    distributions = {}
    
    if model_name in ['xgboost', 'xgb']:
        distributions = {
            'n_estimators': optuna.distributions.IntDistribution(50, 500),
            'max_depth': optuna.distributions.IntDistribution(3, 10),
            'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            'subsample': optuna.distributions.FloatDistribution(0.6, 1.0),
            'colsample_bytree': optuna.distributions.FloatDistribution(0.6, 1.0),
        }
    
    elif model_name in ['lightgbm', 'lgbm']:
        distributions = {
            'n_estimators': optuna.distributions.IntDistribution(50, 500),
            'max_depth': optuna.distributions.IntDistribution(3, 10),
            'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            'num_leaves': optuna.distributions.IntDistribution(20, 100),
            'subsample': optuna.distributions.FloatDistribution(0.6, 1.0),
            'colsample_bytree': optuna.distributions.FloatDistribution(0.6, 1.0),
        }
    
    elif model_name == 'random_forest':
        distributions = {
            'n_estimators': optuna.distributions.IntDistribution(50, 300),
            'max_depth': optuna.distributions.IntDistribution(3, 20),
            'min_samples_split': optuna.distributions.IntDistribution(2, 20),
            'min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
        }
    
    elif model_name == 'catboost':
        distributions = {
            'iterations': optuna.distributions.IntDistribution(50, 500),
            'depth': optuna.distributions.IntDistribution(4, 10),
            'learning_rate': optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            'l2_leaf_reg': optuna.distributions.FloatDistribution(1, 10),
        }
    
    elif model_name == 'rgf':
        distributions = {
            'max_leaf': optuna.distributions.IntDistribution(500, 2000),
            'l2': optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
            'min_samples_leaf': optuna.distributions.IntDistribution(1, 20),
        }
    
    return distributions
