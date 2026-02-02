"""
ウィザード学習実行API

ウィザードの推奨設定を受けて実際にモデル学習を開始
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def start_training_from_wizard(request) -> JsonResponse:
    """
    ウィザード推奨設定からモデル学習を開始
    
    POST /api/training/start-from-wizard
    {
        "dataset_id": "solubility_demo",  // または dataset_id: 1 (DB)
        "wizard_recommendation": {
            "task_type": "regression",
            "model": "random_forest",
            "features": ["morgan_fp", "rdkit_2d"],
            ...
        },
        "experiment_name": "My First ML Model"
    }
    
    Response:
    {
        "success": true,
        "experiment_id": 123,
        "status": "PENDING",
        "message": "学習を開始しました"
    }
    """
    try:
        data = json.loads(request.body)
        
        dataset_source = data.get('dataset_id')
        recommendation = data.get('wizard_recommendation')
        experiment_name = data.get('experiment_name', 'ウィザード学習')
        
        if not dataset_source or not recommendation:
            return JsonResponse({
                "error": "dataset_id and wizard_recommendation are required"
            }, status=400)
        
        # データセット準備
        dataset = _prepare_dataset(dataset_source)
        
        # Experiment設定作成
        config = _build_config_from_recommendation(recommendation)
        
        # Experiment作成
        from core.models import Dataset, Experiment
        
        # データベースに保存されていないデモデータの場合は保存
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
        
        logger.info(f"Training started: experiment_id={experiment.id}")
        
        return JsonResponse({
            "success": True,
            "experiment_id": experiment.id,
            "status": "PENDING",
            "message": "学習を開始しました。進捗はプログレスバーで確認できます。"
        })
    
    except Exception as e:
        logger.error(f"Failed to start training: {e}", exc_info=True)
        return JsonResponse({
            "error": str(e)
        }, status=500)


def _prepare_dataset(dataset_source):
    """
    データセットを準備
    
    Args:
        dataset_source: dataset_id (int) or demo_id (str)
    
    Returns:
        Dataset object or dict
    """
    from core.models import Dataset
    from core.services.demo_datasets import DemoDatasets
    
    # 既存のDatasetから
    if isinstance(dataset_source, int):
        return Dataset.objects.get(id=dataset_source)
    
    # デモデータセットから
    demo = DemoDatasets()
    demo_info = demo.get_info(dataset_source)
    
    if not demo_info:
        raise ValueError(f"Dataset not found: {dataset_source}")
    
    # デモデータをCSVとして一時保存
    import os
    import tempfile
    import pandas as pd
    
    demo_data = demo.load(dataset_source)
    
    # 一時ファイルに保存
    temp_dir = os.path.join(tempfile.gettempdir(), 'chem_ml_demo')
    os.makedirs(temp_dir, exist_ok=True)
    
    csv_path = os.path.join(temp_dir, f"{dataset_source}.csv")
    demo_data.to_csv(csv_path, index=False)
    
    return {
        'name': demo_info['name_ja'],
        'file_path': csv_path,
        'smiles_col': demo_info['smiles_column'],
        'target_col': demo_info['target_column']
    }


def _build_config_from_recommendation(rec: dict) -> dict:
    """
    ウィザード推奨からExperiment設定を構築
    
    Args:
        rec: wizard_recommendation
    
    Returns:
        config dict for Experiment
    """
    # モデル名マッピング
    model_mapping = {
        'linear_regression': 'lightgbm',  # 現在lightgbmのみ対応
        'random_forest': 'random_forest',
        'gradient_boosting': 'lightgbm',
        'gaussian_process': 'lightgbm',  # フォールバック
        'svm': 'lightgbm'  # フォールバック
    }
    
    # 特徴量マッピング
    feature_mapping = {
        'morgan_fp': 'rdkit',  # RDKit特徴量に含まれる
        'rdkit_2d': 'rdkit',
        'molecular_descriptors': 'rdkit',
        'structural_alerts': 'rdkit'
    }
    
    model_type = model_mapping.get(rec.get('model'), 'random_forest')
    
    # 特徴量をユニークに
    features = list(set([
        feature_mapping.get(f, 'rdkit')
        for f in rec.get('features', ['rdkit'])
    ]))
    
    config = {
        'model_type': model_type,
        'task_type': rec.get('task_type', 'regression'),
        'features': features,
        'cv_folds': 5,
        'task_type_mode': 'smiles_only',
        'use_smart_engine': True,
        'target_property': _infer_target_property(rec)
    }
    
    return config


def _infer_target_property(rec: dict) -> str:
    """
    推奨からtarget_propertyを推測
    """
    # データセット名やタスクから推測
    features = rec.get('features', [])
    
    # とりあえずデフォルト
    return 'general'


@require_http_methods(["GET"])
def get_training_status(request, experiment_id: int) -> JsonResponse:
    """
    学習状態を取得
    
    GET /api/training/status/{experiment_id}
    
    Response:
    {
        "experiment_id": 123,
        "status": "RUNNING",  // PENDING, RUNNING, COMPLETED, FAILED
        "progress": 45,  // 0-100
        "current_step": "特徴量抽出中...",
        "metrics": {...}  // 完了時のみ
    }
    """
    try:
        from core.models import Experiment
        
        experiment = Experiment.objects.get(id=experiment_id)
        
        # プログレス推定（簡易版、Phase 2で改善）
        progress = 0
        current_step = "初期化中..."
        
        if experiment.status == 'PENDING':
            progress = 5
            current_step = "タスク待機中..."
        elif experiment.status == 'RUNNING':
            progress = 50
            current_step = "学習実行中..."
        elif experiment.status == 'COMPLETED':
            progress = 100
            current_step = "完了"
        elif experiment.status == 'FAILED':
            progress = 0
            current_step = "エラー"
        
        response = {
            "experiment_id": experiment.id,
            "status": experiment.status,
            "progress": progress,
            "current_step": current_step,
            "name": experiment.name
        }
        
        # 完了時はメトリクスも返す
        if experiment.status == 'COMPLETED' and experiment.metrics:
            response['metrics'] = experiment.metrics
        
        return JsonResponse(response)
    
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return JsonResponse({
            "error": str(e)
        }, status=404)


@require_http_methods(["GET"])
def get_training_results(request, experiment_id: int) -> JsonResponse:
    """
    学習結果を取得
    
    GET /api/training/results/{experiment_id}
    
    Response:
    {
        "experiment_id": 123,
        "name": "My Model",
        "status": "COMPLETED",
        "metrics": {
            "train_r2": 0.85,
            "cv_mean_score": -0.15,
            ...
        },
        "artifacts": [
            {"name": "shap_summary.png", "url": "/api/..."},
            ...
        ]
    }
    """
    try:
        from core.models import Experiment
        
        experiment = Experiment.objects.get(id=experiment_id)
        
        if experiment.status != 'COMPLETED':
            return JsonResponse({
                "error": "Experiment not completed yet"
            }, status=400)
        
        # アーティファクト一覧（簡易版）
        artifacts = []
        artifact_names = [
            'shap_summary.png',
            'learning_curve.png',
            'predicted_vs_actual.png',
            'feature_importance.png'
        ]
        
        for name in artifact_names:
            artifacts.append({
                'name': name,
                'url': f'/api/experiments/{experiment.id}/artifacts/{name}'
            })
        
        return JsonResponse({
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "metrics": experiment.metrics or {},
            "config": experiment.config,
            "artifacts": artifacts,
            "created_at": experiment.created_at.isoformat()
        })
    
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        return JsonResponse({
            "error": str(e)
        }, status=404)
