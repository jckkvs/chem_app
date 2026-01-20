"""
MLPipeline統合テスト - AD/UQ統合検証

Implements: T-PIPE-001
設計思想:
- エンドツーエンドの学習・予測フロー
- AD/UQ機能の動作確認
- save/load検証
- 分岐カバレッジ・Mutation対応

参考文献:
- pytest best practices
- Django testing documentation
"""

from django.test import SimpleTestCase
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from core.services.ml.pipeline import MLPipeline
from core.services.ml.applicability_domain import ADResult


class MLPipelineBasicTests(SimpleTestCase):
    """基本的な学習・予測フロー"""
    
    def setUp(self):
        """テストデータ生成"""
        np.random.seed(42)
        self.X_train = pd.DataFrame(
            np.random.rand(100, 5), 
            columns=[f'f{i}' for i in range(5)]
        )
        self.y_train = pd.Series(np.random.rand(100))
        
        self.X_test = pd.DataFrame(
            np.random.rand(10, 5), 
            columns=[f'f{i}' for i in range(5)]
        )
    
    def test_train_basic_no_ad_uq(self):
        """基本的な学習（AD/UQ無効）"""
        pipeline = MLPipeline(
            model_type='lightgbm',
            task_type='regression',
            enable_ad=False,
            enable_uq=False,
            generate_plots=False,
        )
        
        metrics = pipeline.train(self.X_train, self.y_train)
        
        # メトリクス検証
        self.assertIn('train_r2', metrics)
        self.assertIn('train_mse', metrics)
        self.assertIn('cv_mean_score', metrics)
        
        # モデルが学習済み
        self.assertIsNotNone(pipeline.model)
        self.assertIsNone(pipeline.ad_model_)
        self.assertIsNone(pipeline.uq_model_)
    
    def test_train_with_ad_enabled(self):
        """AD有効時の学習"""
        pipeline = MLPipeline(
            model_type='lightgbm',
            enable_ad=True,
            enable_uq=False,
            generate_plots=False,
        )
        
        metrics = pipeline.train(self.X_train, self.y_train)
        
        # ADモデルが学習済み
        self.assertIsNotNone(pipeline.ad_model_)
        self.assertIsNone(pipeline.uq_model_)
        
        # AD評価を実行
        results = pipeline.get_applicability(self.X_test)
        self.assertEqual(len(results), len(self.X_test))
        
        for r in results:
            self.assertIsInstance(r, ADResult)
            self.assertIsInstance(r.is_within_domain, bool)
            self.assertGreaterEqual(r.confidence, 0)
            self.assertLessEqual(r.confidence, 1)
    
    def test_train_with_uq_enabled(self):
        """UQ有効時の学習（回帰）"""
        pipeline = MLPipeline(
            model_type='lightgbm',
            task_type='regression',
            enable_ad=False,
            enable_uq=True,
            generate_plots=False,
        )
        
        metrics = pipeline.train(self.X_train, self.y_train)
        
        # UQモデルが学習済み
        self.assertIsNone(pipeline.ad_model_)
        self.assertIsNotNone(pipeline.uq_model_)
        
        # 不確実性付き予測
        mean, lower, upper = pipeline.predict_uncertainty(self.X_test)
        
        self.assertEqual(len(mean), len(self.X_test))
        self.assertEqual(len(lower), len(self.X_test))
        self.assertEqual(len(upper), len(self.X_test))
        
        # 区間の妥当性: lower <= mean <= upper
        self.assertTrue(np.all(lower <= mean))
        self.assertTrue(np.all(mean <= upper))
    
    def test_train_with_ad_and_uq(self):
        """AD+UQ両方有効"""
        pipeline = MLPipeline(
            model_type='random_forest',
            task_type='regression',
            enable_ad=True,
            enable_uq=True,
            generate_plots=False,
        )
        
        metrics = pipeline.train(self.X_train, self.y_train)
        
        # 両方学習済み
        self.assertIsNotNone(pipeline.ad_model_)
        self.assertIsNotNone(pipeline.uq_model_)
        
        # 両方の機能が動作
        ad_results = pipeline.get_applicability(self.X_test[:1])
        self.assertEqual(len(ad_results), 1)
        
        mean, lower, upper = pipeline.predict_uncertainty(self.X_test[:1])
        self.assertEqual(len(mean), 1)


class MLPipelineClassificationTests(SimpleTestCase):
    """分類タスクのテスト"""
    
    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
        self.y_train = pd.Series(np.random.choice([0, 1], 100))
        self.X_test = pd.DataFrame(np.random.rand(10, 5), columns=[f'f{i}' for i in range(5)])
    
    def test_classification_basic(self):
        """分類タスク基本"""
        pipeline = MLPipeline(
            model_type='random_forest',
            task_type='classification',
            enable_ad=True,
            enable_uq=False,  # 分類ではUQ無効
            generate_plots=False,
        )
        
        metrics = pipeline.train(self.X_train, self.y_train)
        
        # 分類メトリクス
        self.assertIn('train_accuracy', metrics)
        self.assertIn('train_f1', metrics)
        
        # AD は動作
        self.assertIsNotNone(pipeline.ad_model_)
        # UQ は回帰専用なのでNone
        self.assertIsNone(pipeline.uq_model_)


class MLPipelineSaveLoadTests(SimpleTestCase):
    """save/load検証"""
    
    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(50, 3), columns=['f0', 'f1', 'f2'])
        self.y_train = pd.Series(np.random.rand(50))
        self.X_test = pd.DataFrame(np.random.rand(5, 3), columns=['f0', 'f1', 'f2'])
    
    def test_save_load_with_ad_uq(self):
        """AD/UQを含む保存・復元"""
        # 学習
        pipeline = MLPipeline(
            model_type='lightgbm',
            enable_ad=True,
            enable_uq=True,
            generate_plots=False,
        )
        pipeline.train(self.X_train, self.y_train)
        
        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pipeline.pkl')
            pipeline.save(path)
            
            # 読み込み
            loaded = MLPipeline.load(path)
            
            # 復元確認
            self.assertIsNotNone(loaded.model)
            self.assertIsNotNone(loaded.ad_model_)
            self.assertIsNotNone(loaded.uq_model_)
            
            # 予測が一致
            pred_orig = pipeline.predict(self.X_test)
            pred_loaded = loaded.predict(self.X_test)
            np.testing.assert_array_almost_equal(pred_orig, pred_loaded, decimal=5)
            
            # AD結果が一致
            ad_orig = pipeline.get_applicability(self.X_test)
            ad_loaded = loaded.get_applicability(self.X_test)
            for o, l in zip(ad_orig, ad_loaded):
                self.assertAlmostEqual(o.confidence, l.confidence, places=5)
            
            # UQ結果が一致
            uq_orig = pipeline.predict_uncertainty(self.X_test)
            uq_loaded = loaded.predict_uncertainty(self.X_test)
            for i in range(3):  # mean, lower, upper
                np.testing.assert_array_almost_equal(uq_orig[i], uq_loaded[i], decimal=5)


class MLPipelineErrorHandlingTests(SimpleTestCase):
    """エラーハンドリング"""
    
    def test_predict_before_train(self):
        """学習前の予測でエラー"""
        pipeline = MLPipeline()
        X = pd.DataFrame([[1, 2, 3]])
        
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.predict(X)
        self.assertIn('訓練されていません', str(ctx.exception))
    
    def test_ad_when_disabled(self):
        """AD無効時に呼び出すとエラー"""
        pipeline = MLPipeline(enable_ad=False, generate_plots=False)
        X = pd.DataFrame(np.random.rand(10, 3))
        y = pd.Series(np.random.rand(10))
        pipeline.train(X, y)
        
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.get_applicability(X)
        self.assertIn('有効化されていません', str(ctx.exception))
    
    def test_uq_when_disabled(self):
        """UQ無効時に呼び出すとエラー"""
        pipeline = MLPipeline(enable_uq=False, generate_plots=False)
        X = pd.DataFrame(np.random.rand(10, 3))
        y = pd.Series(np.random.rand(10))
        pipeline.train(X, y)
        
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.predict_uncertainty(X)
        self.assertIn('有効化されていません', str(ctx.exception))


class MLPipelineCustomParamsTests(SimpleTestCase):
    """カスタムパラメータのテスト"""
    
    def test_custom_ad_params(self):
        """ADパラメータのカスタマイズ"""
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.rand(50))
        
        pipeline = MLPipeline(
            enable_ad=True,
            enable_uq=False,
            ad_params={'k_neighbors': 3, 'threshold_percentile': 90},
            generate_plots=False,
        )
        pipeline.train(X, y)
        
        # ADモデルが使用されたパラメータを持つ
        self.assertEqual(pipeline.ad_model_.k_neighbors, 3)
        self.assertEqual(pipeline.ad_model_.threshold_percentile, 90)
    
    def test_custom_uq_params(self):
        """UQパラメータのカスタマイズ"""
        X = pd.DataFrame(np.random.rand(50, 3))
        y = pd.Series(np.random.rand(50))
        
        pipeline = MLPipeline(
            enable_ad=False,
            enable_uq=True,
            uq_params={'method': 'bootstrap', 'n_bootstrap': 10},
            generate_plots=False,
        )
        pipeline.train(X, y)
        
        # UQモデルが指定されたパラメータを持つ
        self.assertEqual(pipeline.uq_model_.method, 'bootstrap')
        self.assertEqual(pipeline.uq_model_.n_bootstrap, 10)


class MLPipelineBackwardCompatibilityTests(SimpleTestCase):
    """後方互換性テスト"""
    
    def test_config_dict(self):
        """古いconfig引数でも動作"""
        X = pd.DataFrame(np.random.rand(30, 3))
        y = pd.Series(np.random.rand(30))
        
        config = {
            'model_type': 'random_forest',
            'task_type': 'regression',
            'cv_folds': 3,
        }
        
        pipeline = MLPipeline(config=config, generate_plots=False)
        metrics = pipeline.train(X, y)
        
        # 正しく学習される
        self.assertIsNotNone(pipeline.model)
        self.assertIn('train_r2', metrics)
