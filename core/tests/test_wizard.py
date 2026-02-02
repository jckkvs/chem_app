"""
ウィザードテスト
"""

import json
import pytest
from django.test import Client


class TestWizardAPI:
    """ウィザードAPIテスト"""
    
    def setup_method(self):
        self.client = Client()
    
    def test_list_available_models(self):
        """利用可能モデル一覧"""
        response = self.client.get('/api/wizard/models')
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'models' in data
        assert len(data['models']) > 0
        
        # random_forestが含まれること
        model_names = [m['name'] for m in data['models']]
        assert 'random_forest' in model_names
    
    def test_get_model_info(self):
        """モデル詳細取得"""
        response = self.client.get('/api/wizard/models/random_forest')
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['name'] == 'random_forest'
        assert 'config' in data
        assert 'description' in data['config']
    
    def test_analyze_dataset_with_csv(self):
        """CSVデータ直接指定での分析"""
        csv_data = [
            {"SMILES": "CCO", "target": 0.2},
            {"SMILES": "CC(C)O", "target": 0.5},
            {"SMILES": "CCCC", "target": 3.2},
            {"SMILES": "c1ccccc1", "target": 2.1},
        ] * 20  # 80サンプル
        
        response = self.client.post(
            '/api/wizard/analyze',
            data=json.dumps({
                'csv_data': csv_data,
                'target_column': 'target',
                'smiles_column': 'SMILES'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 必須フィールド確認
        assert 'task_type' in data
        assert 'model' in data
        assert 'features' in data
        assert 'estimated_time' in data
        assert 'estimated_accuracy' in data
        
        # タスクタイプは回帰
        assert data['task_type'] == 'regression'
        
        # モデル推奨あり
        assert data['model'] in ['random_forest', 'gradient_boosting', 'gaussian_process']


class TestMLWizard:
    """ウィザードエンジンテスト"""
    
    def test_detect_regression(self):
        """回帰タスク判定"""
        from core.services.ml.wizard import MLWizard
        import pandas as pd
        
        wizard = MLWizard()
        
        # 連続値データ
        df = pd.DataFrame({
            'SMILES': ['CCO'] * 100,
            'target': [i * 0.1 for i in range(100)]
        })
        
        task_type = wizard._detect_task_type(df['target'])
        assert task_type == 'regression'
    
    def test_detect_classification(self):
        """分類タスク判定"""
        from core.services.ml.wizard import MLWizard
        import pandas as pd
        
        wizard = MLWizard()
        
        # 2値分類データ
        df = pd.DataFrame({
            'SMILES': ['CCO'] * 100,
            'target': [0, 1] * 50
        })
        
        task_type = wizard._detect_task_type(df['target'])
        assert task_type == 'classification'
    
    def test_recommend_model_small_data(self):
        """少量データでのモデル推奨"""
        from core.services.ml.wizard import MLWizard
        
        wizard = MLWizard()
        model, reasoning = wizard._recommend_model(30, 'regression')
        
        # 少量データではGaussian Process推奨
        assert model == 'gaussian_process'
        assert '少ない' in reasoning['model']
    
    def test_recommend_model_large_data(self):
        """大量データでのモデル推奨"""
        from core.services.ml.wizard import MLWizard
        
        wizard = MLWizard()
        model, reasoning = wizard._recommend_model(10000, 'regression')
        
        # 大量データではGradient Boosting推奨
        assert model in ['gradient_boosting', 'random_forest']
