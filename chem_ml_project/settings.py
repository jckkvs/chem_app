import os
import sys
import warnings
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# ==============================================================================
# Security Settings
# ==============================================================================

# テスト環境検出（最優先）
IS_TESTING = 'test' in sys.argv or 'pytest' in sys.modules

# SECRET_KEY: Must be set via environment variable
# Generate with: python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

if IS_TESTING:
    # テスト環境: 自動生成されたSECRET_KEYを使用（CI/CD対応）
    SECRET_KEY = 'test-secret-key-for-ci-cd-and-pytest-do-not-use-in-production'
else:
    SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
    
    if not SECRET_KEY:
        # デフォルト値は開発環境のみ許可
        if os.environ.get('DJANGO_DEBUG', 'False').lower() in ('true', '1', 'yes'):
            warnings.warn(
                "DJANGO_SECRET_KEY not set. Using insecure default for development only. "
                "Set DJANGO_SECRET_KEY environment variable for production.",
                RuntimeWarning,
                stacklevel=2
            )
            SECRET_KEY = 'django-insecure-dev-only-DO-NOT-USE-IN-PRODUCTION'
        else:
            raise ValueError(
                "DJANGO_SECRET_KEY environment variable must be set in production. "
                "Generate with: python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'"
            )


# DEBUG: Environment variable (default: False for safety, True for testing)
if IS_TESTING:
    DEBUG = True  # テスト環境では常にDEBUG=True
else:
    DEBUG = os.environ.get('DJANGO_DEBUG', 'False').lower() in ('true', '1', 'yes')

# ALLOWED_HOSTS: Environment variable (default: localhost only)
ALLOWED_HOSTS_RAW = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1')
ALLOWED_HOSTS = [
    s.strip() 
    for s in ALLOWED_HOSTS_RAW.split(',')
    if s.strip()
]

# テスト環境用：Django TestCaseのtestserverを許可
if IS_TESTING:
    ALLOWED_HOSTS.append('testserver')

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 'rest_framework', # Removed as we use django-ninja and don't need DRF
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chem_ml_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'chem_ml_project.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ==============================================================================
# Security Settings for Production
# ==============================================================================

# テスト環境判定
IS_TESTING = 'test' in sys.argv or 'pytest' in sys.modules

# セキュリティ設定の有効化判定
# - 本番環境（DEBUG=False）では常に有効
# - テスト環境では環境変数で制御可能（デフォルト: 本番設定をテスト）
ENABLE_PRODUCTION_SECURITY = not DEBUG or (
    IS_TESTING and os.environ.get('TEST_PRODUCTION_SECURITY', 'True').lower() == 'true'
)

if ENABLE_PRODUCTION_SECURITY:
    # Force HTTPS (本番環境のみ、テスト時は環境変数で制御)
    if not IS_TESTING:
        SECURE_SSL_REDIRECT = os.environ.get('DJANGO_SECURE_SSL_REDIRECT', 'True').lower() == 'true'
    else:
        # テスト環境ではSSLリダイレクトを無効化（ローカルテスト用）
        SECURE_SSL_REDIRECT = False
    
    # Secure cookies
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    
    # HSTS (HTTP Strict Transport Security)
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    
    # Content Security
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = 'DENY'
else:
    # 開発環境のみセキュリティ設定を緩和
    SECURE_SSL_REDIRECT = False
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    SECURE_HSTS_SECONDS = 0

