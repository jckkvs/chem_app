"""
ChemMLClientのテスト

カバレッジ目標: 90%以上（57 stmts）
"""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from urllib.error import HTTPError

from core.services.api_client import APIResponse, ChemMLClient


class TestAPIResponse:
    """APIResponseテスト"""
    
    def test_api_response_ok(self):
        """成功レスポンス"""
        response = APIResponse(status_code=200, data={"result": "success"})
        
        assert response.ok is True
        assert response.status_code == 200
        assert response.data == {"result": "success"}
        assert response.error is None
    
    def test_api_response_error(self):
        """エラーレスポンス"""
        response = APIResponse(
            status_code=404,
            data=None,
            error="Not Found"
        )
        
        assert response.ok is False
        assert response.error == "Not Found"
    
    def test_api_response_ok_boundary(self):
        """境界値テスト"""
        # 199はNG
        assert APIResponse(status_code=199, data={}).ok is False
        # 200-299はOK
        assert APIResponse(status_code=200, data={}).ok is True
        assert APIResponse(status_code=299, data={}).ok is True
        # 300以上はNG
        assert APIResponse(status_code=300, data={}).ok is False


class TestChemMLClientInit:
    """初期化テスト"""
    
    def test_init_defaults(self):
        """デフォルト初期化"""
        client = ChemMLClient()
        
        assert client.base_url == "http://localhost:8000"
        assert client.api_key is None
        assert client.timeout == 30
    
    def test_init_custom(self):
        """カスタム初期化"""
        client = ChemMLClient(
            base_url="https://api.example.com/",
            api_key="test_key_123",
            timeout=60
        )
        
        # 末尾のスラッシュが削除される
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test_key_123"
        assert client.timeout == 60


class TestChemMLClientRequest:
    """リクエスト処理テスト"""
    
    @patch('urllib.request.urlopen')
    def test_request_get_success(self, mock_urlopen):
        """GET成功テスト"""
        # レスポンスモック
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"data": "test"}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = ChemMLClient()
        response = client._request('GET', '/test')
        
        assert response.ok is True
        assert response.status_code == 200
        assert response.data == {"data": "test"}
    
    @patch('urllib.request.urlopen')
    def test_request_post_with_data(self, mock_urlopen):
        """POST with dataテスト"""
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.read.return_value = json.dumps({"id": 123}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = ChemMLClient()
        response = client._request(
            'POST',
            '/create',
            data={"name": "test"}
        )
        
        assert response.ok is True
        assert response.status_code == 201
        assert response.data == {"id": 123}
    
    @patch('urllib.request.urlopen')
    def test_request_with_api_key(self, mock_urlopen):
        """API Key付きリクエスト"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"success": true}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        client = ChemMLClient(api_key="secret_key")
        response = client._request('GET', '/secure')
        
        # urlopen呼び出しを検証
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert request_obj.headers.get('Authorization') == 'Bearer secret_key'
    
    @patch('urllib.request.urlopen')
    def test_request_http_error(self, mock_urlopen):
        """HTTP エラーハンドリング"""
        # HTTPError をシミュレート
        error = HTTPError(
            url='http://test.com',
            code=404,
            msg='Not Found',
            hdrs={},
            fp=None
        )
        mock_urlopen.side_effect = error
        
        client = ChemMLClient()
        response = client._request('GET', '/notfound')
        
        assert response.ok is False
        assert response.status_code == 404
        assert response.error == 'Not Found'
    
    @patch('urllib.request.urlopen')
    def test_request_generic_error(self, mock_urlopen):
        """一般的なエラーハンドリング"""
        mock_urlopen.side_effect = Exception("Network error")
        
        client = ChemMLClient()
        response = client._request('GET', '/error')
        
        assert response.ok is False
        assert response.status_code == 0
        assert "Network error" in response.error


class TestChemMLClientAPI:
    """API呼び出しテスト"""
    
    @patch.object(ChemMLClient, '_request')
    def test_predict(self, mock_request):
        """予測APIテスト"""
        mock_request.return_value = APIResponse(
            status_code=200,
            data={"predictions": [1.5, 2.3]}
        )
        
        client = ChemMLClient()
        response = client.predict(["CCO", "CC"], model_id=42)
        
        mock_request.assert_called_once_with(
            'POST',
            '/predict',
            {'smiles': ["CCO", "CC"], 'model_id': 42}
        )
        assert response.ok is True
    
    @patch.object(ChemMLClient, '_request')
    def test_predict_no_model_id(self, mock_request):
        """予測API（model_idなし）"""
        mock_request.return_value = APIResponse(200, {})
        
        client = ChemMLClient()
        client.predict(["CCO"])
        
        # model_id=Noneでも呼ばれる
        call_data = mock_request.call_args[0][2]  # data引数
        assert call_data['model_id'] is None
    
    @patch.object(ChemMLClient, '_request')
    def test_get_experiments(self, mock_request):
        """実験一覧取得テスト"""
        mock_request.return_value = APIResponse(200, {"experiments": []})
        
        client = ChemMLClient()
        client.get_experiments()
        
        mock_request.assert_called_once_with('GET', '/experiments')
    
    @patch.object(ChemMLClient, '_request')
    def test_get_experiment(self, mock_request):
        """実験詳細取得テスト"""
        mock_request.return_value = APIResponse(200, {"id": 123})
        
        client = ChemMLClient()
        client.get_experiment(123)
        
        mock_request.assert_called_once_with('GET', '/experiments/123')
    
    @patch.object(ChemMLClient, '_request')
    def test_create_experiment(self, mock_request):
        """実験作成テスト"""
        mock_request.return_value = APIResponse(201, {"id": 456})
        
        client = ChemMLClient()
        config = {"model": "rf", "params": {}}
        client.create_experiment("Test Exp", 10, config)
        
        mock_request.assert_called_once_with(
            'POST',
            '/experiments',
            {
                'name': "Test Exp",
                'dataset_id': 10,
                'config': config
            }
        )
    
    @patch.object(ChemMLClient, '_request')
    def test_get_datasets(self, mock_request):
        """データセット一覧取得テスト"""
        mock_request.return_value = APIResponse(200, {"datasets": []})
        
        client = ChemMLClient()
        client.get_datasets()
        
        mock_request.assert_called_once_with('GET', '/datasets')
    
    @patch.object(ChemMLClient, '_request')
    def test_upload_dataset(self, mock_request):
        """データセットアップロードテスト"""
        mock_request.return_value = APIResponse(201, {"id": 789})
        
        client = ChemMLClient()
        data = [{"smiles": "CCO", "logP": 1.0}]
        client.upload_dataset("My Dataset", data)
        
        mock_request.assert_called_once_with(
            'POST',
            '/datasets',
            {'name': "My Dataset", 'data': data}
        )
    
    @patch.object(ChemMLClient, '_request')
    def test_health_check_ok(self, mock_request):
        """ヘルスチェック成功"""
        mock_request.return_value = APIResponse(200, {"status": "ok"})
        
        client = ChemMLClient()
        result = client.health_check()
        
        assert result is True
    
    @patch.object(ChemMLClient, '_request')
    def test_health_check_fail(self, mock_request):
        """ヘルスチェック失敗"""
        mock_request.return_value = APIResponse(503, None, "Service Unavailable")
        
        client = ChemMLClient()
        result = client.health_check()
        
        assert result is False
    
    @patch.object(ChemMLClient, '_request')
    def test_health_check_exception(self, mock_request):
        """ヘルスチェック例外"""
        mock_request.side_effect = Exception("Connection error")
        
        client = ChemMLClient()
        result = client.health_check()
        
        assert result is False
