"""
ChemMLPipelineのテスト

Note: 実際の化学特徴抽出器に依存するため、モックを使用
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class ChemMLPipelineBuilderTests(unittest.TestCase):
    """ChemMLPipelineBuilderのテスト"""
    
    def setUp(self):
        # テストデータ
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        self.df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        self.df['SMILES'] = ['CCO'] * 100  # ダミー
        self.df['target'] = y
    
    @patch('core.services.ml.chem_pipeline_helpers.AutoFeatureUnion')
    def test_build_qsar_pipeline_auto(self, mock_auto_union):
        """QSAR Pipeline自動構築のテスト"""
        from core.services.ml.chem_pipeline import ChemMLPipelineBuilder
        
        # モックセットアップ
        mock_union_instance = MagicMock()
        mock_auto_union.return_value.from_feature_list.return_value = mock_union_instance
        
        builder = ChemMLPipelineBuilder()
        pipeline = builder.build_qsar_pipeline(
            smiles_column='SMILES',
            features='auto',
            model='auto'
        )
        
        # Pipelineが構築される
        self.assertIsNotNone(pipeline)
        self.assertTrue(hasattr(pipeline, 'steps'))
    
    def test_parse_params(self):
        """パラメータパースのテスト"""
        from core.services.ml.chem_pipeline import ChemMLPipelineBuilder
        
        builder = ChemMLPipelineBuilder()
        params = builder._parse_params('n_bits=2048,radius=3,use_chirality=true')
        
        self.assertEqual(params['n_bits'], 2048)
        self.assertEqual(params['radius'], 3)
        self.assertEqual(params['use_chirality'], True)


class AutoFeatureUnionTests(unittest.TestCase):
    """AutoFeatureUnionのテスト"""
    
    def test_parse_params(self):
        """パラメータパースのテスト"""
        from core.services.ml.chem_pipeline_helpers import AutoFeatureUnion
        
        builder = AutoFeatureUnion()
        params = builder._parse_params('n_bits=2048,radius=3.5,use_chirality=false')
        
        self.assertEqual(params['n_bits'], 2048)
        self.assertAlmostEqual(params['radius'], 3.5)
        self.assertEqual(params['use_chirality'], False)
    
    def test_smiles_column_selector(self):
        """SmilesColumnSelectorのテスト"""
        from core.services.ml.chem_pipeline_helpers import SmilesColumnSelector
        
        df = pd.DataFrame({'SMILES': ['CCO', 'CC'], 'other': [1, 2]})
        selector = SmilesColumnSelector('SMILES')
        
        result = selector.transform(df)
        self.assertEqual(result.shape[1], 1)


class SmartColumnTransformerTests(unittest.TestCase):
    """SmartColumnTransformerのテスト"""
    
    def setUp(self):
        # 混合型データ
        self.df = pd.DataFrame({
            'SMILES': ['CCO'] * 100,
            'MW': np.random.rand(100) * 100 + 100,
            'LogP': np.random.randn(100),
            'Category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randn(100)
        })
    
    def test_detect_smiles_columns(self):
        """SMILESカラム検出のテスト"""
        from core.services.ml.chem_pipeline_helpers import SmartColumnTransformer
        
        transformer = SmartColumnTransformer()
        smiles_cols = transformer._detect_smiles_columns(self.df, exclude=[])
        
        # 'SMILES'が検出される
        self.assertIn('SMILES', smiles_cols)
    
    def test_detect_continuous_columns(self):
        """連続値カラム検出のテスト"""
        from core.services.ml.chem_pipeline_helpers import SmartColumnTransformer
        
        transformer = SmartColumnTransformer()
        continuous_cols = transformer._detect_continuous_columns(
            self.df, exclude=['SMILES', 'Category', 'target']
        )
        
        # MW, LogPが検出される
        self.assertIn('MW', continuous_cols)
        self.assertIn('LogP', continuous_cols)
    
    def test_detect_categorical_columns(self):
        """カテゴリカラム検出のテスト"""
        from core.services.ml.chem_pipeline_helpers import SmartColumnTransformer
        
        transformer = SmartColumnTransformer()
        categorical_cols = transformer._detect_categorical_columns(
            self.df, exclude=['SMILES', 'MW', 'LogP', 'target']
        )
        
        # Categoryが検出される
        self.assertIn('Category', categorical_cols)
