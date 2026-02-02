"""
チューニングAPI統合テスト
"""

import json
import pytest
from django.test import Client


class TestTuningAPI:
    """チューニングAPI統合テスト"""
    
    def setup_method(self):
        self.client = Client()
    
    def test_start_with_tuning_fast_preset(self):
        """高速チューニング開始（プリセット）"""
        response = self.client.post(
            '/api/tuning/start',
            data=json.dumps({
                'dataset_id': 'solubility',  # デモデータセットID
                'model_type': 'lightgbm',
                'task_type': 'regression',
                'tuning_preset': 'fast',
                'experiment_name': '溶解度予測（高速チューニング）'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] == True
        assert 'experiment_id' in data
        assert data['status'] == 'PENDING'
        assert data['tuning_method'] == 'random'
        assert data['n_iter'] == 20
    
    def test_start_with_tuning_custom_config(self):
        """カスタムチューニング設定"""
        response = self.client.post(
            '/api/tuning/start',
            data=json.dumps({
                'dataset_id': 'solubility',
                'model_type': 'random_forest',
                'tuning_config': {
                    'method': 'random',
                    'n_iter': 10,
                    'cv': 3
                },
                'experiment_name': 'カスタムチューニング'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['success'] == True
        assert data['tuning_method'] == 'random'
        assert data['n_iter'] == 10
    
    def test_start_without_dataset(self):
        """データセットなしでエラー"""
        response = self.client.post(
            '/api/tuning/start',
            data=json.dumps({
                'model_type': 'lightgbm'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
