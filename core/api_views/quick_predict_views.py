"""
ワンクリック予測API

複雑な設定不要で、SMILESを入力するだけで物性予測
"""

import json
import logging
from typing import Dict

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from core.services.ml.pretrained_models import PretrainedModels

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def quick_predict(request) -> JsonResponse:
    """
    ワンクリック予測
    
    POST /api/predict/quick
    {
        "smiles": "CCO",
        "property": "logP"
    }
    
    Response:
    {
        "smiles": "CCO",
        "property": "logP",
        "prediction": 0.23,
        "confidence": 0.85,
        "interpretation": "低い脂溶性",
        "unit": "",
        "model_info": {
            "name": "脂溶性（logP）",
            "accuracy": "R²=0.82"
        }
    }
    """
    try:
        data = json.loads(request.body)
        smiles = data.get('smiles')
        property_name = data.get('property', 'logP')
        
        if not smiles:
            return JsonResponse(
                {"error": "SMILES is required"},
                status=400
            )
        
        # モデルロード
        try:
            model = PretrainedModels.load(property_name)
        except ValueError as e:
            return JsonResponse(
                {"error": str(e)},
                status=400
            )
        
        # 予測
        prediction = model.predict_single(smiles)
        confidence = model.confidence(smiles)
        
        # モデル情報
        model_info_dict = PretrainedModels.AVAILABLE_MODELS[property_name]
        
        # 結果解釈
        interpretation = _interpret_result(property_name, prediction)
        
        return JsonResponse({
            "smiles": smiles,
            "property": property_name,
            "prediction": float(prediction),
            "confidence": float(confidence),
            "interpretation": interpretation,
            "unit": model_info_dict['unit'],
            "model_info": {
                "name": model_info_dict['name'],
                "description": model_info_dict['description'],
                "accuracy": model_info_dict['accuracy']
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


@require_http_methods(["GET"])
def list_properties(request) -> JsonResponse:
    """
    予測可能な物性一覧
    
    GET /api/predict/properties
    
    Response:
    {
        "properties": [
            {
                "id": "logP",
                "name": "脂溶性（logP）",
                "description": "オクタノール-水分配係数",
                "unit": "",
                "accuracy": "R²=0.82"
            },
            ...
        ]
    }
    """
    available = PretrainedModels.list_available()
    
    properties = [
        {
            "id": prop_id,
            "name": info['name'],
            "description": info['description'],
            "unit": info['unit'],
            "accuracy": info['accuracy']
        }
        for prop_id, info in available.items()
    ]
    
    return JsonResponse({"properties": properties})


@csrf_exempt
@require_http_methods(["POST"])
def batch_quick_predict(request) -> JsonResponse:
    """
    バッチ予測（複数SMILES）
    
    POST /api/predict/quick/batch
    {
        "smiles_list": ["CCO", "CC(C)O", "CCCC"],
        "property": "logP"
    }
    
    Response:
    {
        "results": [
            {
                "smiles": "CCO",
                "prediction": 0.23,
                "confidence": 0.85
            },
            ...
        ],
        "count": 3,
        "property": "logP"
    }
    """
    try:
        data = json.loads(request.body)
        smiles_list = data.get('smiles_list', [])
        property_name = data.get('property', 'logP')
        
        if not smiles_list:
            return JsonResponse(
                {"error": "smiles_list is required"},
                status=400
            )
        
        # モデルロード
        model = PretrainedModels.load(property_name)
        
        # バッチ予測
        results = []
        for smiles in smiles_list:
            try:
                prediction = model.predict_single(smiles)
                confidence = model.confidence(smiles)
                
                results.append({
                    "smiles": smiles,
                    "prediction": float(prediction),
                    "confidence": float(confidence),
                    "interpretation": _interpret_result(property_name, prediction)
                })
            except Exception as e:
                logger.warning(f"Failed to predict {smiles}: {e}")
                results.append({
                    "smiles": smiles,
                    "error": str(e)
                })
        
        return JsonResponse({
            "results": results,
            "count": len(results),
            "property": property_name
        })
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return JsonResponse(
            {"error": str(e)},
            status=500
        )


def _interpret_result(property_name: str, value: float) -> str:
    """
    予測結果を人間が理解しやすい形で解釈
    
    Args:
        property_name: 物性名
        value: 予測値
    
    Returns:
        解釈テキスト
    """
    interpretations = {
        'logP': {
            (-float('inf'), 0): "非常に親水性（水に溶けやすい）",
            (0, 2): "親水性",
            (2, 3): "中程度の脂溶性",
            (3, 5): "脂溶性（細胞膜透過性が高い）",
            (5, float('inf')): "非常に高い脂溶性（吸収されにくい可能性）"
        },
        'solubility': {
            (-float('inf'), -6): "難溶性",
            (-6, -4): "低溶解度",
            (-4, -2): "中程度の溶解度",
            (-2, 0): "高溶解度",
            (0, float('inf')): "非常に高い溶解度"
        },
        'QED': {
            (0, 0.3): "医薬品らしさが低い",
            (0.3, 0.5): "やや医薬品らしい",
            (0.5, 0.7): "医薬品らしい",
            (0.7, 1.0): "非常に医薬品らしい"
        },
        'toxicity': {
            (0, 0.3): "低毒性の可能性",
            (0.3, 0.7): "中程度の毒性リスク",
            (0.7, 1.0): "高毒性の可能性"
        }
    }
    
    if property_name not in interpretations:
        return f"値: {value:.2f}"
    
    ranges = interpretations[property_name]
    for (min_val, max_val), text in ranges.items():
        if min_val <= value < max_val:
            return text
    
    return f"値: {value:.2f}"
