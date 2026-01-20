"""
API Endpoint Tests

新規追加したAPIエンドポイントの自動テスト
"""
import pytest
from django.test import Client
import json


class TestMoleculeEndpoints:
    """分子関連エンドポイントのテスト"""
    
    @pytest.fixture
    def client(self):
        return Client()
    
    def test_validate_valid_smiles(self, client):
        """有効なSMILESの検証"""
        response = client.post(
            '/api/molecules/validate',
            data=json.dumps({'smiles': 'CCO'}),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert data['valid'] is True
        assert data['canonical_smiles'] == 'CCO'
    
    def test_validate_invalid_smiles(self, client):
        """無効なSMILESの検証"""
        response = client.post(
            '/api/molecules/validate',
            data=json.dumps({'smiles': 'INVALID_SMILES'}),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert data['valid'] is False
        assert data['error'] is not None
    
    def test_get_properties(self, client):
        """分子物性取得"""
        response = client.get('/api/molecules/CCO/properties')
        assert response.status_code == 200
        data = response.json()
        assert 'molecular_weight' in data
        assert data['molecular_weight'] > 0
        assert 'logp' in data
        assert 'tpsa' in data
    
    def test_get_svg(self, client):
        """分子SVG取得"""
        response = client.get('/api/molecules/c1ccccc1/svg')
        assert response.status_code == 200
        assert 'svg' in response['Content-Type'].lower()


class TestHealthEndpoints:
    """ヘルスチェックエンドポイントのテスト"""
    
    @pytest.fixture
    def client(self):
        return Client()
    
    def test_health_check(self, client):
        """基本ヘルスチェック"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] in ['healthy', 'degraded']
        assert 'database' in data
    
    def test_rdkit_health(self, client):
        """RDKitヘルスチェック"""
        response = client.get('/api/health/rdkit')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert 'rdkit_version' in data


class TestDatasetEndpoints:
    """データセットエンドポイントのテスト"""
    
    @pytest.fixture
    def client(self):
        return Client()
    
    def test_list_datasets(self, client):
        """データセット一覧取得"""
        response = client.get('/api/datasets')
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestExperimentEndpoints:
    """実験エンドポイントのテスト"""
    
    @pytest.fixture
    def client(self):
        return Client()
    
    def test_list_experiments(self, client):
        """実験一覧取得"""
        response = client.get('/api/experiments')
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestSimilarityEndpoints:
    """類似度検索エンドポイントのテスト"""
    
    @pytest.fixture
    def client(self):
        return Client()
    
    def test_similarity_search(self, client):
        """類似度検索"""
        response = client.post(
            '/api/molecules/similarity',
            data=json.dumps({
                'query_smiles': 'c1ccccc1',
                'target_smiles_list': ['c1ccccc1', 'CCO', 'CC(=O)O', 'c1ccc(O)cc1'],
                'threshold': 0.5,
                'top_k': 5
            }),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert data['total_searched'] == 4
        # ベンゼン自体が最も類似している
        if data['results']:
            assert data['results'][0]['similarity'] == 1.0
    
    def test_get_fingerprint(self, client):
        """フィンガープリント取得"""
        response = client.get('/api/molecules/CCO/fingerprint')
        assert response.status_code == 200
        data = response.json()
        assert 'on_bits' in data
        assert data['n_bits'] == 2048
        assert data['bit_count'] > 0

