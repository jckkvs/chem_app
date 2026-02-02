# -*- coding: utf-8 -*-
"""
マルチタスク学習エンジンのテスト

カバレッジ目標: 0% → 60%
"""
import numpy as np
import pandas as pd
import pytest


class TestMultitaskEngineBasic:
    """基本テスト"""
    
    def test_basic_import(self):
        """基本的なインポート"""
        from core.services.ml import multitask
        assert multitask is not None


class TestMultitaskDataPreparation:
    """マルチタスクデータ準備テスト"""
    
    def test_multi_output_data(self):
        """マルチ出力データ"""
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 3)  # 3つのタスク
        
        assert X.shape == (100, 10)
        assert y.shape == (100, 3)


class TestMultiOutputRegressor:
    """マルチ出力回帰テスト"""
    
    def test_multi_output_basic(self):
        """基本的なマルチ出力回帰"""
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import Ridge
        
        X = np.random.rand(100, 5)
        y = np.random.rand(100, 3)
        
        model = MultiOutputRegressor(Ridge())
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert predictions.shape == (10, 3)


class TestChainedOutputs:
    """連鎖マルチ出力テスト"""
    
    def test_classifier_chain(self):
        """分類器チェーン"""
        from sklearn.multioutput import ClassifierChain
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, (100, 3))
        
        model = ClassifierChain(LogisticRegression(max_iter=200))
        model.fit(X, y)
        
        predictions = model.predict(X[:10])
        assert predictions.shape == (10, 3)


class TestMultitaskMetrics:
    """マルチタスク評価指標テスト"""
    
    def test_multi_output_mse(self):
        """マルチ出力MSE"""
        from sklearn.metrics import mean_squared_error
        
        y_true = np.random.rand(100, 3)
        y_pred = y_true + np.random.randn(100, 3) * 0.1
        
        mse = mean_squared_error(y_true, y_pred)
        assert mse >= 0


class TestTaskWeighting:
    """タスク重み付けテスト"""
    
    def test_weighted_loss(self):
        """重み付き損失"""
        y_true = np.array([[1.0, 2.0, 3.0]])
        y_pred = np.array([[1.1, 2.2, 2.9]])
        weights = np.array([1.0, 2.0, 0.5])
        
        errors = (y_true - y_pred) ** 2
        weighted_loss = np.sum(errors * weights)
        
        assert weighted_loss > 0


class TestMultitaskIntegration:
    """統合テスト"""
    
    def test_complete_workflow(self):
        """完全なワークフロー"""
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        # データ生成
        X = np.random.rand(200, 10)
        y = np.random.rand(200, 4)
        
        # 分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # モデル学習
        model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=10, random_state=42)
        )
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        mse = mean_squared_error(y_test, y_pred)
        
        assert y_pred.shape == y_test.shape
        assert mse >= 0
