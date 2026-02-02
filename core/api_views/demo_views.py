"""
デモデータセットAPI

サンプルデータセットのロードとプレビュー
"""

import logging
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from core.services.demo_datasets import DemoDatasets

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def list_demo_datasets(request) -> JsonResponse:
    """
    デモデータセット一覧
    
    GET /api/demo/datasets
    
    Response:
    {
        "datasets": [
            {
                "id": "solubility",
                "name": "Water Solubility",
                "name_ja": "水溶解度予測",
                "n_molecules": 1128,
                "task_type": "regression",
                "difficulty": "beginner"
            }
        ]
    }
    """
    try:
        datasets = DemoDatasets.list_all()
        
        data = {
            "datasets": [
                {
                    "id": ds.id,
                    "name": ds.name,
                    "name_ja": ds.name_ja,
                    "description": ds.description,
                    "n_molecules": ds.n_molecules,
                    "task_type": ds.task_type,
                    "difficulty": ds.difficulty,
                    "smiles_column": ds.smiles_column,
                    "target_column": ds.target_column
                }
                for ds in datasets
            ]
        }
        
        return JsonResponse(data)
    
    except Exception as e:
        logger.error(f"Failed to list demo datasets: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def load_demo_dataset(request, dataset_id: str) -> JsonResponse:
    """
    デモデータセットをロード
    
    GET /api/demo/datasets/{dataset_id}
    
    Response:
    {
        "info": {...},
        "preview": [
            {"SMILES": "CCO", "target": 0.2},
            ...
        ],
        "total_rows": 1128
    }
    """
    try:
        # データセット情報
        info = DemoDatasets.get_info(dataset_id)
        if info is None:
            return JsonResponse(
                {"error": f"Dataset '{dataset_id}' not found"},
                status=404
            )
        
        # データロード
        df = DemoDatasets.load(dataset_id)
        
        # プレビュー（最初の10行）
        preview = df.head(10).to_dict(orient='records')
        
        data = {
            "info": {
                "id": info.id,
                "name": info.name,
                "name_ja": info.name_ja,
                "description": info.description,
                "n_molecules": info.n_molecules,
                "smiles_column": info.smiles_column,
                "target_column": info.target_column,
                "task_type": info.task_type
            },
            "preview": preview,
            "total_rows": len(df),
            "columns": list(df.columns)
        }
        
        return JsonResponse(data)
    
    except Exception as e:
        logger.error(f"Failed to load demo dataset: {e}")
        return JsonResponse({"error": str(e)}, status=500)
