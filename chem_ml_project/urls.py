from django.contrib import admin
from django.urls import path, include
from core.api import api, public_api

urlpatterns = [
    # path('admin/', admin.site.urls), 
    path('api/', api.urls),  # Protected API (authentication required)
    path('api/public/', public_api.urls),  # Public API (no authentication)
    path('', include('core.urls')),  # フロントエンドページ
]
