"""
テンプレートAPI

プリセットモデルテンプレートのAPI
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from core.services.ml.templates import TemplateManager

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def list_templates(request) -> JsonResponse:
    """
    テンプレート一覧を取得
    
    GET /api/templates
    
    Query Parameters:
    - difficulty: beginner, intermediate, advanced
    - task_type: regression, classification
    
    Response:
    {
        "templates": [
            {
                "id": "drug_discovery_basic",
                "name": "Drug Discovery Basic",
                "name_ja": "創薬基礎",
                "icon": "🧪",
                "description": "医薬品候補化合物の基本的な物性予測",
                "model_type": "random_forest",
                "task_type": "regression",
                "difficulty": "beginner",
                "estimated_time": "3-5分"
            }
        ],
        "summary": {
            "total": 6,
            "regression": 4,
            "classification": 2
        }
    }
    """
    try:
        # フィルタパラメータ
        difficulty = request.GET.get('difficulty')
        task_type = request.GET.get('task_type')
        
        # テンプレート取得
        if difficulty:
            templates = TemplateManager.list_by_difficulty(difficulty)
        elif task_type:
            templates = TemplateManager.list_by_task_type(task_type)
        else:
            templates = TemplateManager.list_all()
        
        # JSON変換
        templates_data = [
            {
                'id': t.id,
                'name': t.name,
                'name_ja': t.name_ja,
                'icon': t.icon,
                'description': t.description,
                'model_type': t.model_type,
                'task_type': t.task_type,
                'difficulty': t.difficulty,
                'estimated_time': t.estimated_time
            }
            for t in templates
        ]
        
        # 統計情報
        summary = TemplateManager.get_summary()
        
        return JsonResponse({
            'templates': templates_data,
            'summary': summary
        })
    
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_template(request, template_id: str) -> JsonResponse:
    """
    テンプレート詳細を取得
    
    GET /api/templates/{template_id}
    
    Response:
    {
        "id": "toxicity_screening",
        "name": "Toxicity Screening",
        "name_ja": "毒性스クリー닝",
        "icon": "☠️",
        "description": "化合物の毒性リスク評価",
        "model_type": "lightgbm",
        "task_type": "classification",
        "features": ["rdkit"],
        "use_cases": ["Tox21毒性予測", "AMES変異原性"],
        "recommended_for": ["毒性評価", "不均衡データ"],
        "pros": ["高精度", "不均衡データに強い"],
        "cons": ["学習時間がやや長い"],
        "model_params": {...},
        "difficulty": "intermediate",
        "estimated_time": "5-10分"
    }
    """
    try:
        template = TemplateManager.get(template_id)
        
        if template is None:
            return JsonResponse({
                'error': f'Template not found: {template_id}'
            }, status=404)
        
        # 完全な情報を返す
        data = {
            'id': template.id,
            'name': template.name,
            'name_ja': template.name_ja,
            'icon': template.icon,
            'description': template.description,
            'model_type': template.model_type,
            'task_type': template.task_type,
            'features': template.features,
            'use_cases': template.use_cases,
            'recommended_for': template.recommended_for,
            'pros': template.pros,
            'cons': template.cons,
            'model_params': template.model_params,
            'preprocessing_preset': template.preprocessing_preset,
            'difficulty': template.difficulty,
            'estimated_time': template.estimated_time
        }
        
        return JsonResponse(data)
    
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def start_from_template(request) -> JsonResponse:
    """
    テンプレートから学習を開始
    
    POST /api/training/start-from-template
    {
        "dataset_id": "toxicity_demo",
        "template_id": "toxicity_screening",
        "experiment_name": "毒性予測モデル"
    }
    
    Response:
    {
        "success": true,
        "experiment_id": 456,
        "status": "PENDING",
        "template_used": "toxicity_screening",
        "message": "学習を開始しました"
    }
    """
    try:
        data = json.loads(request.body)
        
        dataset_source = data.get('dataset_id')
        template_id = data.get('template_id')
        experiment_name = data.get('experiment_name', 'テンプレート学習')
        
        if not dataset_source or not template_id:
            return JsonResponse({
                "error": "dataset_id and template_id are required"
            }, status=400)
        
        # テンプレート存在確認
        template = TemplateManager.get(template_id)
        if template is None:
            return JsonResponse({
                "error": f"Template not found: {template_id}"
            }, status=404)
        
        # テンプレートから設定を生成
        config = TemplateManager.apply_to_config(template_id)
        
        # データセット準備（training_viewsと同じロジック）
        from core.api_views.training_views import _prepare_dataset
        dataset = _prepare_dataset(dataset_source)
        
        # Experiment作成
        from core.models import Dataset, Experiment
        
        if isinstance(dataset, dict):
            db_dataset = Dataset.objects.create(
                name=dataset['name'],
                file_path=dataset['file_path'],
                smiles_col=dataset['smiles_col'],
                target_col=dataset['target_col']
            )
        else:
            db_dataset = dataset
        
        # Experiment作成
        experiment = Experiment.objects.create(
            dataset=db_dataset,
            name=experiment_name,
            status='PENDING',
            config=config
        )
        
        # バックグラウンド学習開始
        from core.tasks import run_training_task
        run_training_task(experiment.id)
        
        logger.info(f"Training started from template: experiment_id={experiment.id}, template={template_id}")
        
        return JsonResponse({
            "success": True,
            "experiment_id": experiment.id,
            "status": "PENDING",
            "template_used": template_id,
            "message": f"「{template.name_ja}」テンプレートで学習を開始しました"
        })
    
    except Exception as e:
        logger.error(f"Failed to start training from template: {e}", exc_info=True)
        return JsonResponse({
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_template_summary(request) -> JsonResponse:
    """
    テンプレート統計を取得
    
    GET /api/templates/summary
    
    Response:
    {
        "total": 6,
        "regression": 4,
        "classification": 2,
        "beginner": 2,
        "intermediate": 4,
        "advanced": 0
    }
    """
    try:
        summary = TemplateManager.get_summary()
        return JsonResponse(summary)
    
    except Exception as e:
        logger.error(f"Failed to get summary: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)
