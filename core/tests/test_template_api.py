"""
テンプレートAPI統合テスト
"""

import json
import pytest
from django.test import Client


class TestTemplateAPI:
    """テンプレートAPI統合テスト"""
    
    def setup_method(self):
        self.client = Client()
    
    def test_list_templates(self):
        """テンプレート一覧取得"""
        response = self.client.get('/api/templates')
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'templates' in data
        assert 'summary' in data
        assert len(data['templates']) == 6
        assert data['summary']['total'] == 6
    
    def test_list_templates_by_difficulty(self):
        """難易度フィルタ"""
        response = self.client.get('/api/templates?difficulty=beginner')
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data['templates']) >= 1
        assert all(t['difficulty'] == 'beginner' for t in data['templates'])
    
    def test_list_templates_by_task_type(self):
        """タスクタイプフィルタ"""
        response = self.client.get('/api/templates?task_type=regression')
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data['templates']) >= 1
        assert all(t['task_type'] == 'regression' for t in data['templates'])
    
    def test_get_template(self):
        """テンプレート詳細取得"""
        response = self.client.get('/api/templates/drug_discovery_basic')
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['id'] == 'drug_discovery_basic'
        assert data['name_ja'] == '創薬基礎'
        assert 'pros' in data
        assert 'cons' in data
        assert 'use_cases' in data
        assert 'model_params' in data
    
    def test_get_nonexistent_template(self):
        """存在しないテンプレート"""
        response = self.client.get('/api/templates/nonexistent')
        
        assert response.status_code == 404
    
    def test_get_template_summary(self):
        """統計取得"""
        response = self.client.get('/api/templates/summary')
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['total'] == 6
        assert 'regression' in data
        assert 'classification' in data
