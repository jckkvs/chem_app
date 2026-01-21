"""
APIクライアント

Implements: F-CLIENT-001
設計思想:
- REST API呼び出し
- 認証管理
- エラーハンドリング
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """API応答"""
    status_code: int
    data: Any
    error: Optional[str] = None
    
    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class ChemMLClient:
    """
    Chemical ML Platform APIクライアント
    
    Features:
    - 予測API呼び出し
    - 実験管理
    - データセット操作
    
    Example:
        >>> client = ChemMLClient("http://localhost:8000")
        >>> result = client.predict(["CCO", "c1ccccc1"])
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> APIResponse:
        """HTTP リクエスト"""
        url = f"{self.base_url}/api{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            if data:
                json_data = json.dumps(data).encode('utf-8')
            else:
                json_data = None
            
            req = urllib.request.Request(
                url,
                data=json_data,
                headers=headers,
                method=method,
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode())
                return APIResponse(
                    status_code=response.status,
                    data=response_data,
                )
                
        except urllib.error.HTTPError as e:
            return APIResponse(
                status_code=e.code,
                data=None,
                error=str(e.reason),
            )
        except Exception as e:
            return APIResponse(
                status_code=0,
                data=None,
                error=str(e),
            )
    
    def predict(
        self,
        smiles_list: List[str],
        model_id: Optional[int] = None,
    ) -> APIResponse:
        """予測API"""
        return self._request('POST', '/predict', {
            'smiles': smiles_list,
            'model_id': model_id,
        })
    
    def get_experiments(self) -> APIResponse:
        """実験一覧取得"""
        return self._request('GET', '/experiments')
    
    def get_experiment(self, experiment_id: int) -> APIResponse:
        """実験詳細取得"""
        return self._request('GET', f'/experiments/{experiment_id}')
    
    def create_experiment(
        self,
        name: str,
        dataset_id: int,
        config: Dict[str, Any],
    ) -> APIResponse:
        """実験作成"""
        return self._request('POST', '/experiments', {
            'name': name,
            'dataset_id': dataset_id,
            'config': config,
        })
    
    def get_datasets(self) -> APIResponse:
        """データセット一覧取得"""
        return self._request('GET', '/datasets')
    
    def upload_dataset(
        self,
        name: str,
        data: List[Dict],
    ) -> APIResponse:
        """データセットアップロード"""
        return self._request('POST', '/datasets', {
            'name': name,
            'data': data,
        })
    
    def health_check(self) -> bool:
        """ヘルスチェック"""
        try:
            response = self._request('GET', '/health')
            return response.ok
        except Exception:
            return False
