"""
逆解析API統合テスト
"""

import json
import pytest
from django.test import Client


class TestInverseDesignAPI:
    """逆解析API統合テスト"""
    
    def setup_method(self):
        self.client = Client()
    
    def test_start_without_experiment(self):
        """実験IDなしでエラー"""
        response = self.client.post(
            '/api/inverse-design/start',
            data=json.dumps({
                'target': {'value': 2.0}
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        assert 'experiment_id' in data['error']
    
    def test_start_without_target_value(self):
        """目標値なしでエラー"""
        response = self.client.post(
            '/api/inverse-design/start',
            data=json.dumps({
                'experiment_id': 1,
                'target': {}
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
        assert 'value' in data['error']
    
    def test_inverse_design_page_loads(self):
        """逆解析ページが表示される"""
        response = self.client.get('/inverse-design')
        
        # ページが正常に表示されればOK
        assert response.status_code == 200
