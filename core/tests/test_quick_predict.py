"""
ワンクリック予測APIのテスト
"""

import json
import pytest
from django.test import Client


class TestQuickPredictAPI:
    """ワンクリック予測APIテスト"""
    
    def setup_method(self):
        """各テスト前の準備"""
        self.client = Client()
    
    def test_list_properties(self):
        """物性一覧取得"""
        response = self.client.get('/api/predict/properties')
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'properties' in data
        assert len(data['properties']) > 0
        
        # logPが含まれることを確認
        property_ids = [p['id'] for p in data['properties']]
        assert 'logP' in property_ids
    
    def test_quick_predict_valid(self):
        """正常な予測"""
        response = self.client.post(
            '/api/predict/quick',
            data=json.dumps({
                'smiles': 'CCO',
                'property': 'logP'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 必須フィールド確認
        assert 'smiles' in data
        assert 'property' in data
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'interpretation' in data
        
        # 値の範囲確認
        assert isinstance(data['prediction'], (int, float))
        assert 0.0 <= data['confidence'] <= 1.0
    
    def test_quick_predict_missing_smiles(self):
        """SMILES未入力エラー"""
        response = self.client.post(
            '/api/predict/quick',
            data=json.dumps({
                'property': 'logP'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
    
    def test_quick_predict_invalid_property(self):
        """不正な物性名エラー"""
        response = self.client.post(
            '/api/predict/quick',
            data=json.dumps({
                'smiles': 'CCO',
                'property': 'INVALID_PROPERTY'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data
    
    def test_batch_predict(self):
        """バッチ予測"""
        response = self.client.post(
            '/api/predict/quick/batch',
            data=json.dumps({
                'smiles_list': ['CCO', 'CC(C)O', 'CCCC'],
                'property': 'logP'
            }),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'results' in data
        assert 'count' in data
        assert data['count'] == 3
        assert len(data['results']) == 3
        
        # 各結果を確認
        for result in data['results']:
            assert 'smiles' in result
            assert 'prediction' in result
            assert 'confidence' in result
    
    def test_multiple_properties(self):
        """複数物性で予測"""
        properties = ['logP', 'MW', 'QED']
        
        for prop in properties:
            response = self.client.post(
                '/api/predict/quick',
                data=json.dumps({
                    'smiles': 'CCO',
                    'property': prop
                }),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['property'] == prop


class TestPretrainedModels:
    """プリトレーニングモデルテスト"""
    
    def test_list_available(self):
        """利用可能モデル一覧"""
        from core.services.ml.pretrained_models import PretrainedModels
        
        models = PretrainedModels.list_available()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'logP' in models
    
    def test_load_calculator(self):
        """計算可能プロパティ（MW、QED）"""
        from core.services.ml.pretrained_models import PretrainedModels
        
        # MWテスト
        mw_model = PretrainedModels.load('MW')
        prediction = mw_model.predict_single('CCO')
        
        # エタノールの分子量は約46
        assert 45 < prediction < 47
        assert mw_model.confidence('CCO') == 1.0
    
    def test_load_dummy_model(self):
        """ダミーモデルロード"""
        from core.services.ml.pretrained_models import PretrainedModels
        
        # logPモデル（ファイルがないのでダミー）
        model = PretrainedModels.load('logP')
        
        prediction = model.predict_single('CCO')
        assert isinstance(prediction, float)
        
        confidence = model.confidence('CCO')
        assert 0.0 <= confidence <= 1.0
