from django.urls import path
from . import views

urlpatterns = [
    # ページビュー
    path('', views.index, name='index'),
    path('datasets', views.datasets, name='datasets'),
    path('experiments', views.experiments, name='experiments'),
    path('predict', views.predict, name='predict'),
    
    # PWA関連
    path('manifest.json', views.manifest, name='manifest'),
    path('sw.js', views.service_worker, name='service_worker'),
]

