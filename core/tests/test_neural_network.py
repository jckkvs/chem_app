"""
NeuralNetworkWrapperのテスト
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


class NeuralNetworkWrapperTests(unittest.TestCase):
    """NeuralNetworkWrapperのテスト"""
    
    def setUp(self):
        # 分類データ
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
        
        # 回帰データ
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=[f'f{i}' for i in range(10)])
        self.y_reg = pd.Series(y_reg)
    
    def test_mlp_classifier(self):
        """MLPClassifierのテスト"""
        from core.services.ml.sklearn_modules.neural_network import NeuralNetworkWrapper
        
        nn = NeuralNetworkWrapper(
            model_type='classifier',
            hidden_layer_sizes=(10,),
            max_iter=100,
            random_state=42
        )
        nn.fit(self.X_clf, self.y_clf)
        predictions = nn.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
    
    def test_mlp_regressor(self):
        """MLPRegressorのテスト"""
        from core.services.ml.sklearn_modules.neural_network import NeuralNetworkWrapper
        
        nn = NeuralNetworkWrapper(
            model_type='regressor',
            hidden_layer_sizes=(10,),
            max_iter=100,
            random_state=42
        )
        nn.fit(self.X_reg, self.y_reg)
        predictions = nn.predict(self.X_reg)
        
        self.assertEqual(len(predictions), len(self.y_reg))
    
    def test_predict_proba(self):
        """確率予測のテスト"""
        from core.services.ml.sklearn_modules.neural_network import NeuralNetworkWrapper
        
        nn = NeuralNetworkWrapper(
            model_type='classifier',
            hidden_layer_sizes=(10,),
            max_iter=100,
            random_state=42
        )
        nn.fit(self.X_clf, self.y_clf)
        probas = nn.predict_proba(self.X_clf)
        
        # 確率の合計は1
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))
    
    def test_custom_architecture(self):
        """カスタムアーキテクチャのテスト"""
        from core.services.ml.sklearn_modules.neural_network import NeuralNetworkWrapper
        
        # 3層ネットワーク
        nn = NeuralNetworkWrapper(
            model_type='classifier',
            hidden_layer_sizes=(50, 25, 10),
            activation='tanh',
            solver='adam',
            learning_rate_init=0.01,
            max_iter=50,
            random_state=42
        )
        nn.fit(self.X_clf, self.y_clf)
        
        self.assertIsNotNone(nn.model_)
    
    def test_early_stopping(self):
        """Early stoppingのテスト"""
        from core.services.ml.sklearn_modules.neural_network import NeuralNetworkWrapper
        
        nn = NeuralNetworkWrapper(
            model_type='regressor',
            hidden_layer_sizes=(10,),
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=100,
            random_state=42
        )
        nn.fit(self.X_reg, self.y_reg)
        
        # Early stoppingで早期終了する可能性がある
        self.assertIsNotNone(nn.model_)
    
    def test_get_model_info(self):
        """モデル情報取得のテスト"""
        from core.services.ml.sklearn_modules.neural_network import NeuralNetworkWrapper
        
        nn = NeuralNetworkWrapper(model_type='classifier', max_iter=10, random_state=42)
        nn.fit(self.X_clf, self.y_clf)
        info = nn.get_model_info()
        
        self.assertIn('model_type', info)
        self.assertIn('n_iterations', info)
    
    def test_create_mlp_model_helper(self):
        """create_mlp_modelヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules.neural_network import create_mlp_model
        
        model = create_mlp_model('classification', (20, 10), max_iter=50, random_state=42)
        model.fit(self.X_clf, self.y_clf)
        predictions = model.predict(self.X_clf)
        
        self.assertEqual(len(predictions), len(self.y_clf))
