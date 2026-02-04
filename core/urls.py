from django.urls import path

from . import views

urlpatterns = [
    # ページビュー
    path('', views.index, name='index'),
    path('datasets', views.datasets, name='datasets'),
    path('experiments', views.experiments, name='experiments'),
    path('predict', views.predict, name='predict'),
    path('quick-predict', views.quick_predict_page, name='quick_predict'),
    path('wizard', views.wizard_page, name='wizard'),
    path('templates', views.templates_page, name='templates'),
    path('inverse-design', views.inverse_design_page, name='inverse_design'),
    path('generate', views.generate_page, name='generate'),
    path('timeseries', views.timeseries_analysis, name='timeseries'),
    path('proxy-settings', lambda r: __import__('django.shortcuts', fromlist=['render']).render(r, 'core/proxy_settings.html'), name='proxy_settings'),
    
    # PWA関連
    path('manifest.json', views.manifest, name='manifest'),
    path('sw.js', views.service_worker, name='service_worker'),
    
    # API: データセット情報
    path('api/datasets/<int:dataset_id>/columns/', views.get_dataset_columns, name='dataset_columns'),
    
    # API: ワンクリック予測
    path('api/predict/quick', lambda r: __import__('core.api_views.quick_predict_views', fromlist=['quick_predict']).quick_predict(r), name='quick_predict'),
    path('api/predict/quick/batch', lambda r: __import__('core.api_views.quick_predict_views', fromlist=['batch_quick_predict']).batch_quick_predict(r), name='batch_quick_predict'),
    path('api/predict/properties', lambda r: __import__('core.api_views.quick_predict_views', fromlist=['list_properties']).list_properties(r), name='list_properties'),
    
    # API: デモデータセット
    path('api/demo/datasets', lambda r: __import__('core.api_views.demo_views', fromlist=['list_demo_datasets']).list_demo_datasets(r), name='list_demo_datasets'),
    path('api/demo/datasets/<str:dataset_id>', lambda r, dataset_id: __import__('core.api_views.demo_views', fromlist=['load_demo_dataset']).load_demo_dataset(r, dataset_id), name='load_demo_dataset'),
    
    # API: ウィザード
    path('api/wizard/analyze', lambda r: __import__('core.api_views.wizard_views', fromlist=['analyze_dataset']).analyze_dataset(r), name='wizard_analyze'),
    path('api/wizard/models', lambda r: __import__('core.api_views.wizard_views', fromlist=['list_available_models']).list_available_models(r), name='wizard_list_models'),
    path('api/wizard/models/<str:model_name>', lambda r, model_name: __import__('core.api_views.wizard_views', fromlist=['get_model_info']).get_model_info(r, model_name), name='wizard_model_info'),
    
    # API: 学習実行
    path('api/training/start-from-wizard', lambda r: __import__('core.api_views.training_views', fromlist=['start_training_from_wizard']).start_training_from_wizard(r), name='start_training_from_wizard'),
    path('api/training/status/<int:experiment_id>', lambda r, experiment_id: __import__('core.api_views.training_views', fromlist=['get_training_status']).get_training_status(r, experiment_id), name='get_training_status'),
    path('api/training/results/<int:experiment_id>', lambda r, experiment_id: __import__('core.api_views.training_views', fromlist=['get_training_results']).get_training_results(r, experiment_id), name='get_training_results'),
    path('api/training/start-from-template', lambda r: __import__('core.api_views.template_views', fromlist=['start_from_template']).start_from_template(r), name='start_from_template'),
    
    # API: テンプレート
    path('api/templates', lambda r: __import__('core.api_views.template_views', fromlist=['list_templates']).list_templates(r), name='list_templates'),
    path('api/templates/summary', lambda r: __import__('core.api_views.template_views', fromlist=['get_template_summary']).get_template_summary(r), name='get_template_summary'),
    path('api/templates/<str:template_id>', lambda r, template_id: __import__('core.api_views.template_views', fromlist=['get_template']).get_template(r, template_id), name='get_template'),
    
    # API: チューニング
    path('api/tuning/start', lambda r: __import__('core.api_views.tuning_views', fromlist=['start_with_tuning']).start_with_tuning(r), name='start_with_tuning'),
    path('api/tuning/status/<int:experiment_id>', lambda r, experiment_id: __import__('core.api_views.tuning_views', fromlist=['get_tuning_status']).get_tuning_status(r, experiment_id), name='get_tuning_status'),
    path('api/tuning/results/<int:experiment_id>', lambda r, experiment_id: __import__('core.api_views.tuning_views', fromlist=['get_tuning_results']).get_tuning_results(r, experiment_id), name='get_tuning_results'),
    
    # API: 逆解析
    path('api/inverse-design/start', lambda r: __import__('core.api_views.inverse_design_views', fromlist=['start_inverse_design']).start_inverse_design(r), name='start_inverse_design'),
    path('api/inverse-design/status/<int:job_id>', lambda r, job_id: __import__('core.api_views.inverse_design_views', fromlist=['get_inverse_design_status']).get_inverse_design_status(r, job_id), name='get_inverse_design_status'),
    path('api/inverse-design/results/<int:job_id>', lambda r, job_id: __import__('core.api_views.inverse_design_views', fromlist=['get_inverse_design_results']).get_inverse_design_results(r, job_id), name='get_inverse_design_results'),
    
    # API: 分子生成
    path('api/generate/molecules', lambda r: __import__('core.api.generation_views', fromlist=['generate_molecules']).generate_molecules(r), name='generate_molecules'),
    path('api/generate/conditional', lambda r: __import__('core.api.generation_views', fromlist=['conditional_generate']).conditional_generate(r), name='conditional_generate'),
    path('api/generate/from-text', lambda r: __import__('core.api.generation_views', fromlist=['text_to_molecule']).text_to_molecule(r), name='text_to_molecule'),
]

