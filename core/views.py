"""
Django Views - Chemical ML Platform

テンプレートベースビュー + PWA対応
"""
import json

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

# ========== ページビュー ==========

def index(request):
    """ダッシュボード"""
    return render(request, 'index.html')


def datasets(request):
    """データセット管理"""
    return render(request, 'datasets.html')


def experiments(request):
    """実験管理"""
    return render(request, 'experiments.html')


def predict(request):
    """予測ページ"""
    return render(request, 'predict.html')


def quick_predict_page(request):
    """ワンクリック予測ページ"""
    return render(request, 'core/quick_predict.html')


def wizard_page(request):
    """AIウィザードページ"""
    return render(request, 'core/wizard.html')


def templates_page(request):
    """プリセットテンプレートページ"""
    return render(request, 'core/templates.html')


def inverse_design_page(request):
    """逆解析（インバースデザイン）ページ"""
    return render(request, 'core/inverse_design.html')


def generate_page(request):
    """分子生成AIページ"""
    return render(request, 'core/generate.html')


def timeseries_analysis(request):
    """時系列分析ページ"""
    from core.models import Dataset
    import pandas as pd
    
    if request.method == 'POST':
        try:
            # フォームデータ取得
            dataset_id = request.POST.get('dataset')
            date_column = request.POST.get('date_column')
            target_column = request.POST.get('target_column')
            model_type = request.POST.get('model_type', 'prophet')
            
            # データ読込
            dataset = Dataset.objects.get(id=dataset_id)
            df = pd.read_csv(dataset.file.path)
            
            # モデル作成と学習
            from core.services.ml.timeseries_models import ProphetWrapper, ARIMAWrapper
            
            if model_type == 'prophet':
                model = ProphetWrapper(
                    growth=request.POST.get('growth', 'linear'),
                    changepoint_prior_scale=float(request.POST.get('changepoint_prior_scale', 0.05)),
                    yearly_seasonality=request.POST.get('yearly_seasonality') == 'on',
                    weekly_seasonality=request.POST.get('weekly_seasonality') == 'on',
                )
            else:  # ARIMA/SARIMA
                p = int(request.POST.get('p', 1))
                d = int(request.POST.get('d', 1))
                q = int(request.POST.get('q', 1))
                
                if model_type == 'sarima':
                    P = int(request.POST.get('P', 1))
                    D = int(request.POST.get('D', 1))
                    Q = int(request.POST.get('Q', 1))
                    S = int(request.POST.get('S', 12))
                    model = ARIMAWrapper(order=(p, d, q), seasonal_order=(P, D, Q, S))
                else:
                    model = ARIMAWrapper(order=(p, d, q))
            
            # 学習
            X = df[[date_column]]
            y = df[target_column]
            model.fit(X, y)
            
            # 成功メッセージ
            results_html = '<div class="alert alert-success">✅ モデル学習完了！</div>'
            
            return render(request, 'core/timeseries.html', {
                'datasets': Dataset.objects.all(),
                'results': results_html
            })
        
        except Exception as e:
            results_html = f'<div class="alert alert-danger">❌ エラー: {str(e)}</div>'
            return render(request, 'core/timeseries.html', {
                'datasets': Dataset.objects.all(),
                'results': results_html
            })
    
    return render(request, 'core/timeseries.html', {
        'datasets': Dataset.objects.all()
    })


def get_dataset_columns(request, dataset_id):
    """データセットのカラム情報を取得（API）"""
    from core.models import Dataset
    import pandas as pd
    
    try:
        dataset = Dataset.objects.get(id=dataset_id)
        df = pd.read_csv(dataset.file.path)
        
        # datetime型カラム検出
        datetime_columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
            elif 'date' in col.lower() or 'time' in col.lower():
                # 試しに変換してみる
                try:
                    pd.to_datetime(df[col])
                    datetime_columns.append(col)
                except:
                    pass
        
        # 数値カラム
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        return JsonResponse({
            'datetime_columns': datetime_columns,
            'numeric_columns': numeric_columns,
            'all_columns': df.columns.tolist()
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# ========== PWA関連 ==========

@require_GET
def manifest(request):
    """PWA Web App Manifest"""
    manifest_data = {
        "name": "ChemML Platform",
        "short_name": "ChemML",
        "description": "機械学習を使った分子物性予測プラットフォーム",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f0f23",
        "theme_color": "#667eea",
        "orientation": "any",
        "icons": [
            {
                "src": "/static/icons/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/static/icons/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ],
        "categories": ["science", "productivity"],
        "lang": "ja",
        "dir": "ltr"
    }
    return JsonResponse(manifest_data, content_type='application/manifest+json')


@require_GET
def service_worker(request):
    """Service Worker for PWA"""
    sw_js = """
// ChemML Service Worker v1.0
const CACHE_NAME = 'chemml-v1';
const OFFLINE_URL = '/';

// キャッシュするリソース
const PRECACHE_URLS = [
    '/',
    '/datasets',
    '/experiments',
    '/predict',
];

// インストール
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('Caching app shell');
            return cache.addAll(PRECACHE_URLS);
        })
    );
    self.skipWaiting();
});

// アクティベート
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keyList) => {
            return Promise.all(keyList.map((key) => {
                if (key !== CACHE_NAME) {
                    console.log('Removing old cache', key);
                    return caches.delete(key);
                }
            }));
        })
    );
    self.clients.claim();
});

// フェッチ（ネットワークファースト戦略）
self.addEventListener('fetch', (event) => {
    // APIリクエストはキャッシュしない
    if (event.request.url.includes('/api/')) {
        return;
    }
    
    event.respondWith(
        fetch(event.request)
            .then((response) => {
                // レスポンスをキャッシュに保存
                if (response.status === 200) {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseClone);
                    });
                }
                return response;
            })
            .catch(() => {
                // オフライン時はキャッシュから
                return caches.match(event.request).then((response) => {
                    return response || caches.match(OFFLINE_URL);
                });
            })
    );
});
"""
    return HttpResponse(sw_js, content_type='application/javascript')

