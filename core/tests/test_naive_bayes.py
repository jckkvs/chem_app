"""
NaiveBayesWrapperのテスト
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


class NaiveBayesWrapperTests(unittest.TestCase):
    """NaiveBayesWrapperのテスト"""
    
    def setUp(self):
        # 分類データ
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
        self.y = pd.Series(y)
    
    def test_gaussian_nb(self):
        """GaussianNBのテスト"""
        from core.services.ml.sklearn_modules import NaiveBayesWrapper
        
        nb = NaiveBayesWrapper(model_type='gaussian')
        nb.fit(self.X, self.y)
        predictions = nb.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertEqual(nb.selected_model_type_, 'gaussian')
    
    def test_multinomial_nb(self):
        """MultinomialNBのテスト"""
        from core.services.ml.sklearn_modules import NaiveBayesWrapper
        
        # 非負整数データ
        X_count = pd.DataFrame(np.random.randint(0, 10, (100, 10)))
        
        nb = NaiveBayesWrapper(model_type='multinomial')
        nb.fit(X_count, self.y)
        predictions = nb.predict(X_count)
        
        self.assertEqual(len(predictions), len(self.y))
    
    def test_auto_select(self):
        """自動選択のテスト"""
        from core.services.ml.sklearn_modules import NaiveBayesWrapper
        
        nb = NaiveBayesWrapper(model_type='auto')
        nb.fit(self.X, self.y)
        
        # 連続値データ → gaussianが選択される
        self.assertEqual(nb.selected_model_type_, 'gaussian')
    
    def test_auto_select_binary(self):
        """二値データでBernoulli自動選択のテスト"""
        from core.services.ml.sklearn_modules import NaiveBayesWrapper
        
        X_binary = pd.DataFrame(np.random.randint(0, 2, (100, 10)))
        
        nb = NaiveBayesWrapper(model_type='auto')
        nb.fit(X_binary, self.y)
        
        self.assertEqual(nb.selected_model_type_, 'bernoulli')
    
    def test_predict_proba(self):
        """確率予測のテスト"""
        from core.services.ml.sklearn_modules import NaiveBayesWrapper
        
        nb = NaiveBayesWrapper(model_type='gaussian')
        nb.fit(self.X, self.y)
        probas = nb.predict_proba(self.X)
        
        # 確率の合計は1
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))
    
    def test_get_available_models(self):
        """利用可能モデル一覧のテスト"""
        from core.services.ml.sklearn_modules import NaiveBayesWrapper
        
        models = NaiveBayesWrapper.get_available_models()
        
        self.assertEqual(len(models), 5)
        self.assertIn('gaussian', models)
        self.assertIn('multinomial', models)
    
    def test_auto_select_helper(self):
        """auto_select_naive_bayesヘルパー関数のテスト"""
        from core.services.ml.sklearn_modules import auto_select_naive_bayes
        
        model = auto_select_naive_bayes(self.X, self.y)
        predictions = model.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertIsNotNone(model.selected_model_type_)
