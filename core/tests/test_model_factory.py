"""
ModelFactoryのテスト

Implements: TEST-MODEL-FACTORY-001
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class ModelFactoryTests(unittest.TestCase):
    """ModelFactoryの基本テスト"""
    
    def setUp(self):
        from core.services.ml.model_factory import ModelFactory
        
        self.factory = ModelFactory()
    
    def test_create_sklearn_models(self):
        """Scikit-learnモデルの作成テスト"""
        # 回帰モデル
        models_regression = [
            'random_forest', 'ridge', 'lasso', 'decision_tree'
        ]
        
        for model_name in models_regression:
            with self.subTest(model=model_name):
                model = self.factory.create('regression', model_name)
                self.assertIsInstance(model, BaseEstimator)
                
                # 簡単な学習テスト
                X = pd.DataFrame(np.random.rand(10, 5))
                y = pd.Series(np.random.rand(10))
                model.fit(X, y)
                preds = model.predict(X)
                self.assertEqual(len(preds), 10)
        
        # 分類モデル
        models_classification = [
            'random_forest', 'logistic', 'decision_tree'
        ]
        
        for model_name in models_classification:
            with self.subTest(model=model_name):
                model = self.factory.create('classification', model_name)
                self.assertIsInstance(model, BaseEstimator)
                
                # 簡単な学習テスト
                X = pd.DataFrame(np.random.rand(10, 5))
                y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
                model.fit(X, y)
                preds = model.predict(X)
                self.assertEqual(len(preds), 10)
    
    def test_create_xgboost_lightgbm(self):
        """XGBoost/LightGBMモデルの作成テスト"""
        from core.services.ml.model_factory import AVAILABLE_LIBRARIES
        
        if AVAILABLE_LIBRARIES['xgboost']:
            model = self.factory.create('regression', 'xgboost', n_estimators=10)
            self.assertIsNotNone(model)
        
        if AVAILABLE_LIBRARIES['lightgbm']:
            model = self.factory.create('regression', 'lightgbm', n_estimators=10)
            self.assertIsNotNone(model)
    
    def test_custom_params(self):
        """カスタムパラメータのテスト"""
        model = self.factory.create('regression', 'random_forest', n_estimators=50, max_depth=5)
        
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.max_depth, 5)
    
    def test_unknown_model_raises_error(self):
        """不明なモデルでエラーを発生させる"""
        with self.assertRaises(ValueError):
            self.factory.create('regression', 'unknown_model_xyz')
    
    def test_list_available_models(self):
        """利用可能なモデル一覧の取得テスト"""
        all_models = self.factory.list_available_models()
        self.assertIsInstance(all_models, list)
        self.assertIn('random_forest', all_models)
        
        regression_models = self.factory.list_available_models('regression')
        self.assertIsInstance(regression_models, list)
        self.assertIn('random_forest', regression_models)
    
    def test_library_status(self):
        """ライブラリ利用可能状況の取得"""
        status = self.factory.get_library_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('sklearn', status)
        self.assertTrue(status['sklearn'])  # Scikit-learnは常に利用可能


class ModelFactoryOptionalLibrariesTests(unittest.TestCase):
    """オプションライブラリのテスト（インストール済みの場合のみ）"""
    
    def setUp(self):
        from core.services.ml.model_factory import AVAILABLE_LIBRARIES, ModelFactory
        
        self.available = AVAILABLE_LIBRARIES
        self.factory = ModelFactory()
    
    def test_catboost_if_available(self):
        """CatBoostが利用可能ならテスト"""
        if not self.available['catboost']:
            self.skipTest("CatBoost not installed")
        
        model = self.factory.create('regression', 'catboost', iterations=10)
        self.assertIsNotNone(model)
        
        X = pd.DataFrame(np.random.rand(20, 5))
        y = pd.Series(np.random.rand(20))
        model.fit(X, y, verbose=False)
        preds = model.predict(X)
        self.assertEqual(len(preds), 20)
    
    def test_imodels_if_available(self):
        """imodelsが利用可能ならテスト"""
        if not self.available['imodels']:
            self.skipTest("imodels not installed")
        
        try:
            model = self.factory.create('regression', 'rulefit')
            self.assertIsNotNone(model)
        except ValueError:
            # imodelsはインストールされているが、特定モデルが利用不可の場合
            pass
    
    def test_rgf_if_available(self):
        """RGFが利用可能ならテスト"""
        if not self.available['rgf']:
            self.skipTest("RGF not installed")
        
        try:
            model = self.factory.create('regression', 'rgf', max_leaf=100)
            self.assertIsNotNone(model)
        except ValueError:
            pass
    
    def test_linear_tree_if_available(self):
        """linear-treeが利用可能ならテスト"""
        if not self.available['linear_tree']:
            self.skipTest("linear-tree not installed")
        
        try:
            model = self.factory.create('regression', 'linear_tree')
            self.assertIsNotNone(model)
        except ValueError:
            pass
    
    def test_ngboost_if_available(self):
        """NGBoostが利用可能ならテスト"""
        if not self.available['ngboost']:
            self.skipTest("NGBoost not installed")
        
        try:
            model = self.factory.create('regression', 'ngboost')
            self.assertIsNotNone(model)
        except ValueError:
            pass


class OptunaIntegrationTests(unittest.TestCase):
    """Optuna統合のテスト"""
    
    def test_create_model_with_optuna_fallback(self):
        """Optunaが利用不可の場合のフォールバックテスト"""
        from core.services.ml.model_factory import AVAILABLE_LIBRARIES, create_model_with_optuna
        
        if AVAILABLE_LIBRARIES['optuna']:
            self.skipTest("Optuna is available, testing fallback")
        
        X = pd.DataFrame(np.random.rand(50, 5))
        y = pd.Series(np.random.rand(50))
        
        # Optunaなしでもモデルが返される
        model = create_model_with_optuna(
            'regression', 'random_forest', X, y, n_trials=5
        )
        self.assertIsNotNone(model)
    
    def test_create_model_with_optuna_if_available(self):
        """Optunaが利用可能な場合の最適化テスト"""
        from core.services.ml.model_factory import AVAILABLE_LIBRARIES, create_model_with_optuna
        
        if not AVAILABLE_LIBRARIES['optuna']:
            self.skipTest("Optuna not installed")
        
        X = pd.DataFrame(np.random.rand(50, 5))
        y = pd.Series(np.random.rand(50))
        
        model = create_model_with_optuna(
            'regression', 'random_forest', X, y, n_trials=3, cv=3
        )
        
        self.assertIsNotNone(model)
        # 最適化されたモデルで予測可能
        preds = model.predict(X)
        self.assertEqual(len(preds), 50)
