"""
DataSplitter と NestedCV のテスト

Implements: TEST-SPLITTER-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


class DataSplitterTests(unittest.TestCase):
    """DataSplitterのテスト"""
    
    def setUp(self):
        from core.services.ml.splitter import DataSplitter
        
        # 回帰データ
        X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
        self.X_reg = pd.DataFrame(X_reg, columns=[f'f{i}' for i in range(10)])
        self.y_reg = pd.Series(y_reg)
        
        # 分類データ
        X_clf, y_clf = make_classification(n_samples=100, n_features=10, random_state=42)
        self.X_clf = pd.DataFrame(X_clf, columns=[f'f{i}' for i in range(10)])
        self.y_clf = pd.Series(y_clf)
        
        # グループデータ
        self.groups = pd.Series([i % 5 for i in range(100)])  # 5グループ
    
    def test_random_kfold(self):
        """KFold分割のテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='random', n_splits=5)
        splits = list(splitter.split(self.X_reg))
        
        self.assertEqual(len(splits), 5)
        
        # 各分割のサイズ確認
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
    
    def test_stratified_kfold(self):
        """Stratified KFold分割のテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='stratified', n_splits=5)
        splits = list(splitter.split(self.X_clf, self.y_clf))
        
        self.assertEqual(len(splits), 5)
    
    def test_group_kfold(self):
        """Group KFold分割のテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='group', n_splits=5)
        splits = list(splitter.split(self.X_reg, self.y_reg, self.groups))
        
        self.assertEqual(len(splits), 5)
    
    def test_timeseries_split(self):
        """Time Series分割のテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='timeseries', n_splits=5)
        splits = list(splitter.split(self.X_reg))
        
        self.assertEqual(len(splits), 5)
    
    def test_loo(self):
        """Leave-One-Outのテスト"""
        from core.services.ml.splitter import DataSplitter
        
        # 小さいデータでテスト
        X_small = self.X_reg.iloc[:10]
        
        splitter = DataSplitter(strategy='loo')
        splits = list(splitter.split(X_small))
        
        # LOOは n_samples 分割を作成
        self.assertEqual(len(splits), 10)
        
        # 各テストセットはサイズ1
        for train_idx, test_idx in splits:
            self.assertEqual(len(test_idx), 1)
    
    def test_leave_one_group_out(self):
        """Leave-One-Group-Outのテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='leave_one_group_out')
        splits = list(splitter.split(self.X_reg, self.y_reg, self.groups))
        
        # グループ数分の分割
        self.assertEqual(len(splits), 5)
    
    def test_repeated_kfold(self):
        """Repeated KFoldのテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='repeated_kfold', n_splits=3)
        splitter.n_repeats = 2
        
        splits = list(splitter.split(self.X_reg))
        
        # n_splits * n_repeats 分割
        self.assertEqual(len(splits), 3 * 2)
    
    def test_shuffle_split(self):
        """Shuffle Splitのテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(strategy='shuffle_split', n_splits=10, test_size=0.2)
        splits = list(splitter.split(self.X_reg))
        
        self.assertEqual(len(splits), 10)
        
        # テストサイズ確認
        for train_idx, test_idx in splits:
            self.assertAlmostEqual(len(test_idx) / len(self.X_reg), 0.2, delta=0.05)
    
    def test_train_test_split(self):
        """Train/Test分割のテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(test_size=0.3)
        X_train, X_test, y_train, y_test = splitter.train_test_split(
            self.X_reg, self.y_reg
        )
        
        self.assertEqual(len(X_train) + len(X_test), len(self.X_reg))
        self.assertAlmostEqual(len(X_test) / len(self.X_reg), 0.3, delta=0.05)
    
    def test_train_val_test_split(self):
        """Train/Val/Test 3分割のテスト"""
        from core.services.ml.splitter import DataSplitter
        
        splitter = DataSplitter(test_size=0.2)
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_val_test_split(
            self.X_reg, self.y_reg, val_size=0.1
        )
        
        total = len(X_train) + len(X_val) + len(X_test)
        self.assertEqual(total, len(self.X_reg))


class NestedCVTests(unittest.TestCase):
    """NestedCVのテスト"""
    
    def setUp(self):
        # 小さいデータセット
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        self.y = pd.Series(y)
    
    def test_nested_cv_basic(self):
        """Nested CVの基本テスト"""
        from sklearn.ensemble import RandomForestRegressor
        from core.services.ml.nested_cv import NestedCV
        
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
        }
        
        nested_cv = NestedCV(outer_cv=3, inner_cv=2)
        results = nested_cv.fit(model, self.X, self.y, param_grid)
        
        # 結果検証
        self.assertIn('outer_scores', results)
        self.assertIn('best_params', results)
        self.assertIn('mean_score', results)
        self.assertIn('std_score', results)
        
        # 外側CVの分割数分のスコア
        self.assertEqual(len(results['outer_scores']), 3)
        self.assertEqual(len(results['best_params']), 3)
    
    def test_nested_cv_summary(self):
        """Nested CV サマリーのテスト"""
        from sklearn.linear_model import Ridge
        from core.services.ml.nested_cv import NestedCV
        
        model = Ridge()
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        
        nested_cv = NestedCV(outer_cv=3, inner_cv=2)
        nested_cv.fit(model, self.X, self.y, param_grid)
        
        summary = nested_cv.get_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('Mean Score', summary)
        self.assertIn('Std Score', summary)
