"""
設定・モデル管理API

Implements: F-SETTINGS-API-001
"""

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from core.services.model_manager import ModelManager


# ModelManagerインスタンス
_model_manager = None

def get_model_manager():
    """ModelManagerシングルトン取得"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


@require_http_methods(["POST"])
@csrf_exempt
def set_proxy(request):
    """プロキシ設定API"""
    try:
        data = json.loads(request.body)
        http_proxy = data.get('http_proxy')
        https_proxy = data.get('https_proxy')
        
        manager = get_model_manager()
        manager.set_proxy(http_proxy, https_proxy)
        
        return JsonResponse({
            'success': True,
            'message': 'プロキシ設定を保存しました',
            'config': manager.get_proxy_config(),
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
        }, status=400)


@require_http_methods(["GET"])
def get_proxy(request):
    """プロキシ設定取得API"""
    manager = get_model_manager()
    return JsonResponse(manager.get_proxy_config())


@require_http_methods(["GET"])
def test_proxy(request):
    """プロキシ接続テストAPI"""
    import requests
    
    manager = get_model_manager()
    proxy_config = manager.get_proxy_config()
    
    try:
        # Google DNS に接続テスト
        proxies = {}
        if proxy_config['http_proxy']:
            proxies['http'] = proxy_config['http_proxy']
        if proxy_config['https_proxy']:
            proxies['https'] = proxy_config['https_proxy']
        
        response = requests.get(
            'https://www.google.com',
            proxies=proxies if proxies else None,
            timeout=10,
        )
        
        if response.status_code == 200:
            return JsonResponse({
                'success': True,
                'message': 'プロキシ経由で接続できました',
            })
        else:
            return JsonResponse({
                'success': False,
                'error': f'接続失敗 (Status: {response.status_code})',
            })
    
    except requests.exceptions.ProxyError:
        return JsonResponse({
            'success': False,
            'error': 'プロキシ接続エラー。設定を確認してください。',
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
        })


@require_http_methods(["GET"])
def list_models(request):
    """モデル一覧取得API"""
    manager = get_model_manager()
    models = manager.list_models()
    return JsonResponse(models, safe=False)


@require_http_methods(["POST"])
@csrf_exempt
def download_model(request, model_id):
    """モデルダウンロードAPI"""
    manager = get_model_manager()
    success, message = manager.download_model(model_id)
    
    return JsonResponse({
        'success': success,
        'message': message,
    })


@require_http_methods(["DELETE"])
@csrf_exempt
def delete_model(request, model_id):
    """モデル削除API"""
    manager = get_model_manager()
    success, message = manager.delete_model(model_id)
    
    return JsonResponse({
        'success': success,
        'message': message,
    })


@require_http_methods(["GET"])
def get_manual_instructions(request, model_id):
    """手動ダウンロード手順取得API"""
    manager = get_model_manager()
    instructions = manager.get_manual_instructions(model_id)
    
    if instructions:
        return JsonResponse(instructions)
    else:
        return JsonResponse({
            'error': 'Unknown model ID',
        }, status=404)


@require_http_methods(["GET"])
def get_cache_info(request):
    """キャッシュ情報取得API"""
    manager = get_model_manager()
    return JsonResponse(manager.get_cache_info())
