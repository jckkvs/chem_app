"""
チューニングAPI

ハイパーパラメータチューニング関連のAPI
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def start_with_tuning(request) -> JsonResponse:
    """
    ハイパーパラメータチューニング付き学習を開始
    
    POST /api/tuning/start
    {
        "dataset_id": "solubility_demo",
        "model_type": "lightgbm",
        "task_type": "regression",
        "tuning_preset": "fast",  // "fast", "balanced", "thorough"
        "experiment_name": "溶解度予測（チューニング）"
    }
    
    または
    
    {
        "dataset_id": "solubility_demo",
        "model_type": "lightgbm",
        "tuning_config": {
            "method": "random",
            "n_iter": 30,
            "cv": 5
        },
        "experiment_name": "溶解度予測（カスタムチューニング）"
    }
    
    Response:
    {
        "success": true,
        "experiment_id": 789,
        "status": "TUNING",
        "estimated_time": "5-10分",
        "tuning_method": "random",
        "n_iter": 20
    }
    """
    try:
        data = json.loads(request.body)
        
        dataset_source = data.get('dataset_id')
        model_type = data.get('model_type', 'lightgbm')
        task_type = data.get('task_type', 'regression')
        experiment_name = data.get('experiment_name', 'チューニング実験')
        
        if not dataset_source:
            return JsonResponse({
                "error": "dataset_id is required"
            }, status=400)
        
        # チューニング設定
        tuning_preset = data.get('tuning_preset', 'fast')
        tuning_config = data.get('tuning_config')
        
        if tuning_config is None:
            # プリセット使用
            from core.services.ml.hyperparameter_tuner import HyperparameterTuner
            tuner = HyperparameterTuner()
            preset_cfg = tuner.get_preset_config(tuning_preset)
            
            tuning_config = {
                'preset': tuning_preset,
                'method': preset_cfg['method'],
                'n_iter': preset_cfg.get('n_iter'),
                'cv': preset_cfg.get('cv', 5)
            }
            estimated_time = preset_cfg['estimated_time']
        else:
            estimated_time = "15-30分"  # カスタム設定の場合
        
        # データセット準備
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
        
        # 設定作成
        config = {
            'model_type': model_type,
            'task_type': task_type,
            'task_type_mode': 'smiles_only',
            'use_smart_engine': True,
            'cv_folds': tuning_config.get('cv', 5),
            'features': ['rdkit']
        }
        
        experiment = Experiment.objects.create(
            dataset=db_dataset,
            name=experiment_name,
            status='PENDING',
            config=config
        )
        
        # バックグラウンドチューニング開始
        from core.tasks import run_tuning_task
        run_tuning_task(experiment.id, tuning_config)
        
        logger.info(
            f"Tuning started: experiment_id={experiment.id}, "
            f"preset={tuning_preset}, config={tuning_config}"
        )
        
        return JsonResponse({
            "success": True,
            "experiment_id": experiment.id,
            "status": "PENDING",
            "estimated_time": estimated_time,
            "tuning_method": tuning_config['method'],
            "n_iter": tuning_config.get('n_iter', 'all'),
            "message": f"チューニングを開始しました（{estimated_time}）"
        })
    
    except Exception as e:
        logger.error(f"Failed to start tuning: {e}", exc_info=True)
        return JsonResponse({
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_tuning_status(request, experiment_id: int) -> JsonResponse:
    """
    チューニング状態を取得
    
    GET /api/tuning/status/{experiment_id}
    
    Response:
    {
        "experiment_id": 789,
        "status": "TUNING",
        "progress": {
            "current_iter": 15,
            "total_iter": 30,
            "percentage": 50,
            "best_score": 0.85,
            "current_params": {...}
        },
        "name": "溶解度予測"
    }
    """
    try:
        from core.models import Experiment
        
        experiment = Experiment.objects.get(id=experiment_id)
        
        response = {
            "experiment_id": experiment.id,
            "status": experiment.status,
            "name": experiment.name
        }
        
        # プログレス情報（簡易版）
        if experiment.status == 'TUNING':
            # TODO: 実際のプログレスを保存・取得する仕組みを追加
            response['progress'] = {
                "message": "チューニング実行中...",
                "estimated_remaining": "不明"
            }
        elif experiment.status == 'RUNNING':
            response['progress'] = {
                "message": "最適パラメータで学習中..."
            }
        elif experiment.status == 'COMPLETED':
            response['progress'] = {
                "message": "完了"
            }
            # チューニング結果があれば追加
            if experiment.metrics:
                response['tuning_info'] = {
                    'improvement': experiment.metrics.get('tuning_improvement'),
                    'trials': experiment.metrics.get('tuning_trials'),
                    'time': experiment.metrics.get('tuning_time')
                }
        
        return JsonResponse(response)
    
    except Experiment.DoesNotExist:
        return JsonResponse({
            "error": f"Experiment {experiment_id} not found"
        }, status=404)
    
    except Exception as e:
        logger.error(f"Failed to get tuning status: {e}")
        return JsonResponse({
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_tuning_results(request, experiment_id: int) -> JsonResponse:
    """
    チューニング結果を取得
    
    GET /api/tuning/results/{experiment_id}
    
    Response:
    {
        "experiment_id": 789,
        "name": "溶解度予測",
        "status": "COMPLETED",
        "best_params": {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "max_depth": 10
        },
        "best_score": 0.87,
        "improvement": 12.5,
        "total_trials": 30,
        "total_time": 1115.2,
        "metrics": {
            "train_r2": 0.89,
            "cv_mean_score": -0.12
        }
    }
    """
    try:
        from core.models import Experiment
        
        experiment = Experiment.objects.get(id=experiment_id)
        
        if experiment.status != 'COMPLETED':
            return JsonResponse({
                "error": "Tuning not completed yet",
                "status": experiment.status
            }, status=400)
        
        # 最適パラメータ
        best_params = experiment.config.get('model_params', {})
        
        # メトリクス
        metrics = experiment.metrics or {}
        
        response = {
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "best_params": best_params,
            "improvement": metrics.get('tuning_improvement'),
            "total_trials": metrics.get('tuning_trials'),
            "total_time": metrics.get('tuning_time'),
            "metrics": {
                k: v for k, v in metrics.items()
                if not k.startswith('tuning_')
            },
            "created_at": experiment.created_at.isoformat()
        }
        
        return JsonResponse(response)
    
    except Experiment.DoesNotExist:
        return JsonResponse({
            "error": f"Experiment {experiment_id} not found"
        }, status=404)
    
    except Exception as e:
        logger.error(f"Failed to get tuning results: {e}")
        return JsonResponse({
            "error": str(e)
        }, status=500)
