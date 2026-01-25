"""
PipelineBuilderのテスト

Implements: TEST-PIPELINE-001
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class PipelineBuilderTests(unittest.TestCase):
    """PipelineBuilderのテスト"""
    
    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
        self.y = pd.Series(y)
    
    def test_build_pipeline(self):
        """Pipeline構築のテスト"""
        from core.services.ml.pipeline_builder import PipelineBuilder
        
        builder = PipelineBuilder()
        pipeline = builder.build_pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10)),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # パイプライン実行
        pipeline.fit(self.X, self.y)
        predictions = pipeline.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
    
    def test_build_feature_union(self):
        """FeatureUnion構築のテスト"""
        from core.services.ml.pipeline_builder import PipelineBuilder
        
        builder = PipelineBuilder()
        feature_union = builder.build_feature_union([
            ('pca', PCA(n_components=10)),
            ('scaler', StandardScaler())
        ])
        
        # 変換実行
        X_transformed = feature_union.fit_transform(self.X)
        
        # 2つのTransformerの出力が結合される
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertEqual(X_transformed.shape[1], 10 + 20)  # PCA(10) + Original(20)
    
    def test_make_simple_pipeline(self):
        """簡易Pipeline作成のテスト"""
        from core.services.ml.pipeline_builder import PipelineBuilder
        
        builder = PipelineBuilder()
        pipeline = builder.make_simple_pipeline(
            StandardScaler(),
            PCA(n_components=10),
            RandomForestClassifier(n_estimators=10, random_state=42)
        )
        
        pipeline.fit(self.X, self.y)
        predictions = pipeline.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
    
    def test_make_simple_union(self):
        """簡易FeatureUnion作成のテスト"""
        from core.services.ml.pipeline_builder import PipelineBuilder
        
        builder = PipelineBuilder()
        feature_union = builder.make_simple_union(
            PCA(n_components=5),
            StandardScaler()
        )
        
        X_transformed = feature_union.fit_transform(self.X)
        
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])


class CustomTransformerTests(unittest.TestCase):
    """カスタムTransformerのテスト"""
    
    def setUp(self):
        X = np.random.rand(100, 10)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(10)])
    
    def test_function_transformer(self):
        """FunctionTransformerのテスト"""
        from core.services.ml.pipeline_builder import FunctionTransformer
        
        def log_transform(X):
            return np.log1p(X)
        
        transformer = FunctionTransformer(log_transform)
        X_transformed = transformer.fit_transform(self.X)
        
        # log1p変換が適用される
        np.testing.assert_array_almost_equal(
            X_transformed,
            np.log1p(self.X.values)
        )
    
    def test_dataframe_selector(self):
        """DataFrameSelectorのテスト"""
        from core.services.ml.pipeline_builder import DataFrameSelector
        
        selector = DataFrameSelector(['f0', 'f1', 'f2'])
        X_selected = selector.fit_transform(self.X)
        
        # 3カラムが選択される
        self.assertEqual(X_selected.shape[1], 3)


class PipelineHelperTests(unittest.TestCase):
    """ヘルパー関数のテスト"""
    
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])
        self.y = pd.Series(y)
    
    def test_create_preprocessing_pipeline(self):
        """前処理パイプライン作成のテスト"""
        from core.services.ml.pipeline_builder import create_preprocessing_pipeline
        
        pipeline = create_preprocessing_pipeline(
            scaler='standard',
            feature_selector='k_best',
            n_features=10
        )
        
        self.assertIsNotNone(pipeline)
        
        # パイプライン実行
        X_transformed = pipeline.fit_transform(self.X, self.y)
        
        # 10特徴に削減される
        self.assertEqual(X_transformed.shape[1], 10)
    
    def test_create_ml_pipeline(self):
        """MLパイプライン作成のテスト"""
        from core.services.ml.pipeline_builder import create_ml_pipeline
        
        pipeline = create_ml_pipeline(
            preprocessor=StandardScaler(),
            model=RandomForestRegressor(n_estimators=10, random_state=42)
        )
        
        pipeline.fit(self.X, self.y)
        predictions = pipeline.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
    
    def test_create_feature_engineering_pipeline(self):
        """特徴量エンジニアリングパイプライン作成のテスト"""
        from core.services.ml.pipeline_builder import create_feature_engineering_pipeline
        
        pipeline = create_feature_engineering_pipeline(
            transformers=[
                ('pca', PCA(n_components=5)),
                ('scaler', StandardScaler())
            ],
            final_estimator=RandomForestRegressor(n_estimators=10, random_state=42)
        )
        
        pipeline.fit(self.X, self.y)
        predictions = pipeline.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
    
    def test_get_pipeline_steps(self):
        """Pipelineステップ取得のテスト"""
        from sklearn.pipeline import Pipeline
        from core.services.ml.pipeline_builder import get_pipeline_steps
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor())
        ])
        
        steps = get_pipeline_steps(pipeline)
        
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0][0], 'scaler')
        self.assertEqual(steps[1][0], 'model')
