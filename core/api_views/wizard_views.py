"""
モデル学習ウィザードAPI
"""

import json
import logging

import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from core.services.ml.wizard import MLWizard, format_recommendation_for_ui

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def analyze_dataset(request) -> JsonResponse:
    """
    データセットを分析して最適な設定を推奨
    
    POST /api/wizard/analyze
    {
        "dataset_id": 1
    }
    または
    {
        "csv_data": [...],  // CSVの中身（JSON配列）
        "target_column": "target",
        "smiles_column": "SMILES"
    }
    
    Response:
    {
        "task_type": "regression",
        "task_type_display": "回帰問題",
        "model": "random_forest",
        "model_display": "ランダムフォレスト（万能型）",
        "features": ["morgan_fp", "rdkit_2d"],
        "estimated_time": "約2分",
        "estimated_accuracy": "R²=0.75",
        "reasoning": {...},
        "warnings": []
    }
    """
    try:
        data = json.loads(request.body)
        
        # データ取得
        if 'dataset_id' in data:
            # データベースから
            from core.models import Dataset
            dataset = Dataset.objects.get(id=data['dataset_id'])
            df = pd.read_csv(dataset.file_path)
            target_col = dataset.target_col
            smiles_col = dataset.smiles_col
        
        elif 'csv_data' in data:
            # 直接データ
            df = pd.DataFrame(data['csv_data'])
            target_col = data.get('target_column', 'target')
            smiles_col = data.get('smiles_column', 'SMILES')
        
        else:
            return JsonResponse(
                {"error": "dataset_id or csv_data is required"},
                status=400
            )
        
        # ウィザード実行
        wizard = MLWizard()
        recommendation = wizard.auto_configure(
            df=df,
            target_column=target_col,
            smiles_column=smiles_col
        )
        
        # UI用にフォーマット
        result = format_recommendation_for_ui(recommendation)
        
        # データ統計も追加
        result['data_stats'] = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'target_range': {
                'min': float(df[target_col].min()),
                'max': float(df[target_col].max()),
                'mean': float(df[target_col].mean()),
                'std': float(df[target_col].std())
            } if recommendation.task_type == 'regression' else None
        }
        
        return JsonResponse(result)
    
    except Exception as e:
        logger.error(f"Wizard analysis failed: {e}")
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


@require_http_methods(["GET"])
def get_model_info(request, model_name: str) -> JsonResponse:
    """
    モデルの詳細情報を取得
    
    GET /api/wizard/models/{model_name}
    """
    from core.services.ml.wizard import MLWizard
    
    if model_name not in MLWizard.MODEL_CONFIGS:
        return JsonResponse(
            {"error": f"Model '{model_name}' not found"},
            status=404
        )
    
    config = MLWizard.MODEL_CONFIGS[model_name]
    
    return JsonResponse({
        "name": model_name,
        "config": config
    })


@require_http_methods(["GET"])
def list_available_models(request) -> JsonResponse:
    """
    利用可能なモデル一覧
    
    GET /api/wizard/models
    """
    from core.services.ml.wizard import MLWizard
    
    models = []
    for name, config in MLWizard.MODEL_CONFIGS.items():
        models.append({
            "name": name,
            "description": config['description'],
            "speed": config['speed'],
            "accuracy": config['accuracy'],
            "interpretability": config['interpretability']
        })
    
    return JsonResponse({"models": models})
