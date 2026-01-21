# matplotlib を headless モードに設定（GUI不要でテスト実行）
import matplotlib
matplotlib.use('Agg')

import os
import pytest
import django
from django.conf import settings

def pytest_configure():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chem_ml_project.settings')
    django.setup()
