"""
学習実行APIテスト
"""

import json
import pytest
from django.test import Client


class TestTrainingAPI:
    """学習実行APIテスト"""
    
    def setup_method(self):
        self.client = Client()
    
    def test_start_training_from_wizard(self):
        """ウィザードから学習開始"""
        # デモデータセットで学習開始
        response = self.client.post(
            '/api/training/start-from-wizard',
            data=json.dumps({
                'dataset_id': 'solubility_demo',
                'wizard_recommendation': {
                    'task_type': 'regression',
                    'model': 'random_forest',
                    'features': ['morgan_fp', 'rdkit_2d']
                },
                'experiment_name': 'Test Experiment'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'success' in data
        assert data['success'] == True
        assert 'experiment_id' in data
        assert data['status'] == 'PENDING'
    
    def test_get_training_status(self):
        """学習状態取得"""
        # まず学習を開始
        start_response = self.client.post(
            '/api/training/start-from-wizard',
            data=json.dumps({
                'dataset_id': 'solubility_demo',
                'wizard_recommendation': {
                    'task_type': 'regression',
                    'model': 'random_forest',
                    'features': ['rdkit']
                },
                'experiment_name': 'Status Test'
            }),
            content_type='application/json'
        )
        
        experiment_id = start_response.json()['experiment_id']
        
        # 状態取得
        status_response = self.client.get(f'/api/training/status/{experiment_id}')
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        assert 'experiment_id' in status_data
        assert 'status' in status_data
        assert 'progress' in status_data
        assert 'current_step' in status_data
        
        # 進捗は0-100
        assert 0 <= status_data['progress'] <= 100


class TestConfigConversion:
    """設定変換ロジックテスト"""
    
    def test_build_config_from_recommendation(self):
        """推奨から設定構築"""
        from core.api_views.training_views import _build_config_from_recommendation
        
        recommendation = {
            'task_type': 'regression',
            'model': 'random_forest',
            'features': ['morgan_fp', 'rdkit_2d', 'molecular_descriptors']
        }
        
        config = _build_config_from_recommendation(recommendation)
        
        # 必須フィールド
        assert 'model_type' in config
        assert 'task_type' in config
        assert 'features' in config
        
        # タスクタイプ
        assert config['task_type'] == 'regression'
        
        # 特徴量がマッピングされている
        assert 'rdkit' in config['features']
    
    def test_model_mapping(self):
        """モデル名マッピング"""
        from core.api_views.training_views import _build_config_from_recommendation
        
        # ウィザード推奨モデル → 実装モデルのマッピング確認
        test_cases = [
            ('random_forest', 'random_forest'),
            ('gradient_boosting', 'lightgbm'),
            ('gaussian_process', 'lightgbm'),  # フォールバック
        ]
        
        for wizard_model, expected_model in test_cases:
            rec = {'model': wizard_model, 'task_type': 'regression'}
            config = _build_config_from_recommendation(rec)
            assert config['model_type'] == expected_model
