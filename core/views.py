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

