import os
import django
from django.conf import settings

def pytest_configure():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chem_ml_project.settings')
    django.setup()
