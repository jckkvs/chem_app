"""
逆解析API

インバースデザイン関連のAPI
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def start_inverse_design(request) -> JsonResponse:
    """
    逆解析開始
    
    POST /api/inverse-design/start
    {
        "experiment_id": 123,  // 学習済みモデルの実験ID
        "target": {
            "property": "solubility",
            "value": 2.0,
            "direction": "maximize"  // or "minimize", "target"
        },
        "method": "bayesian",  // "exhaustive", "random", "bayesian"
        "n_iterations": 100,
        "constraints": {
            "molecular_weight": {"min": 200, "max": 500},
            "logP": {"min": -2, "max": 5}
        },
        "library": null  // 全探索の場合は化合物リスト
    }
    """
    try:
        data = json.loads(request.body)
        
        experiment_id = data.get('experiment_id')
        target = data.get('target', {})
        method = data.get('method', 'bayesian')
        n_iterations = data.get('n_iterations', 100)
        constraints = data.get('constraints')
        library = data.get('library')
        
        if not experiment_id:
            return JsonResponse({
                "error": "experiment_id is required"
            }, status=400)
        
        if not target or 'value' not in target:
            return JsonResponse({
                "error": "target.value is required"
            }, status=400)
        
        # InverseDesignJob作成
        from core.models import Experiment, InverseDesignJob
        
        experiment = Experiment.objects.get(id=experiment_id)
        
        if experiment.status != 'COMPLETED':
            return JsonResponse({
                "error": "Experiment must be completed before inverse design"
            }, status=400)
        
        job = InverseDesignJob.objects.create(
            experiment=experiment,
            target_property=target.get('property', 'solubility'),
            target_value=target['value'],
            direction=target.get('direction', 'maximize'),
            method=method,
            n_iterations=n_iterations,
            constraints=constraints or {},
            status='PENDING'
        )
        
        # バックグラウンド実行
        from core.tasks import run_inverse_design_task
        run_inverse_design_task(job.id, library)
        
        logger.info(f"Inverse design started: job_id={job.id}, method={method}")
        
        return JsonResponse({
            "success": True,
            "job_id": job.id,
            "status": "PENDING",
            "estimated_time": _estimate_time(method, n_iterations),
            "message": "逆解析を開始しました"
        })
    
    except Experiment.DoesNotExist:
        return JsonResponse({
            "error": f"Experiment {experiment_id} not found"
        }, status=404)
    
    except Exception as e:
        logger.error(f"Failed to start inverse design: {e}", exc_info=True)
        return JsonResponse({
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_inverse_design_status(request, job_id: int) -> JsonResponse:
    """
    逆解析状態取得
    
    GET /api/inverse-design/status/{job_id}
    """
    try:
        from core.models import InverseDesignJob
        
        job = InverseDesignJob.objects.get(id=job_id)
        
        response = {
            "job_id": job.id,
            "status": job.status,
            "target": {
                "property": job.target_property,
                "value": job.target_value,
                "direction": job.direction
            },
            "method": job.method
        }
        
        if job.status == 'RUNNING':
            response['progress'] = {
                "message": "探索実行中..."
            }
        elif job.status == 'COMPLETED' and job.results:
            response['num_candidates'] = len(job.results.get('candidates', []))
            response['best_score'] = job.results.get('candidates', [{}])[0].get('score', 0)
        
        return JsonResponse(response)
    
    except InverseDesignJob.DoesNotExist:
        return JsonResponse({
            "error": f"Job {job_id} not found"
        }, status=404)
    
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return JsonResponse({
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_inverse_design_results(request, job_id: int) -> JsonResponse:
    """
    逆解析結果取得
    
    GET /api/inverse-design/results/{job_id}
    """
    try:
        from core.models import InverseDesignJob
        
        job = InverseDesignJob.objects.get(id=job_id)
        
        if job.status != 'COMPLETED':
            return JsonResponse({
                "error": "Job not completed yet",
                "status": job.status
            }, status=400)
        
        results = job.results or {}
        candidates = results.get('candidates', [])
        
        response = {
            "job_id": job.id,
            "status": job.status,
            "target": {
                "property": job.target_property,
                "value": job.target_value,
                "direction": job.direction
            },
            "method": job.method,
            "candidates": candidates[:100],  # トップ100
            "total_evaluated": results.get('total_evaluated', 0),
            "total_time": results.get('total_time', 0),
            "created_at": job.created_at.isoformat(),
            "completed_at": job.updated_at.isoformat()
        }
        
        return JsonResponse(response)
    
    except InverseDesignJob.DoesNotExist:
        return JsonResponse({
            "error": f"Job {job_id} not found"
        }, status=404)
    
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        return JsonResponse({
            "error": str(e)
        }, status=500)


def _estimate_time(method: str, n_iterations: int) -> str:
    """推定時間を計算"""
    if method == 'exhaustive':
        return "1-3分"
    elif method == 'random':
        if n_iterations <= 50:
            return "1-2分"
        elif n_iterations <= 100:
            return "2-5分"
        else:
            return "5-10分"
    elif method == 'bayesian':
        if n_iterations <= 50:
            return "3-5分"
        elif n_iterations <= 100:
            return "5-10分"
        else:
            return "10-15分"
    return "不明"
