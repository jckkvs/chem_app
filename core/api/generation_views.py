"""
生成AI APIエンドポイント

Implements: F-GEN-API-001
"""

import logging
from typing import List, Dict, Any

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json

from core.services.generation import MolGPTGenerator, GeneratedMoleculeValidator

logger = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["POST"])
def generate_molecules(request) -> JsonResponse:
    """
    分子生成API
    
    POST /api/generate/molecules
    {
        "n_molecules": 10,
        "temperature": 0.8,
        "max_length": 100,
        "validate": true
    }
    
    Response:
    {
        "molecules": [
            {
                "smiles": "CCO",
                "score": 1.0,
                "validation": {
                    "is_valid": true,
                    "qed_score": 0.8,
                    "lipinski_violations": 0
                }
            }
        ],
        "count": 10
    }
    """
    try:
        data = json.loads(request.body)
        
        n_molecules = data.get('n_molecules', 10)
        temperature = data.get('temperature', 1.0)
        max_length = data.get('max_length', 100)
        validate = data.get('validate', True)
        
        # 生成
        generator = MolGPTGenerator()
        molecules = generator.generate(
            n_molecules=n_molecules,
            temperature=temperature,
            max_length=max_length
        )
        
        # 検証
        validator = GeneratedMoleculeValidator() if validate else None
        
        results = []
        for mol in molecules:
            result = {
                "smiles": mol.smiles,
                "score": mol.score
            }
            
            if validator:
                validation = validator.validate(mol.smiles)
                result["validation"] = {
                    "is_valid": validation.is_valid,
                    "qed_score": validation.qed_score,
                    "lipinski_violations": validation.lipinski_violations,
                    "molecular_weight": validation.molecular_weight,
                    "logp": validation.logp
                }
            
            results.append(result)
        
        return JsonResponse({
            "molecules": results,
            "count": len(results),
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def conditional_generate(request) -> JsonResponse:
    """
    条件付き分子生成API
    
    POST /api/generate/conditional
    {
        "n_molecules": 10,
        "properties": {
            "logP": 2.5,
            "MW": 300
        },
        "temperature": 0.8
    }
    """
    try:
        data = json.loads(request.body)
        
        n_molecules = data.get('n_molecules', 10)
        properties = data.get('properties', {})
        temperature = data.get('temperature', 1.0)
        validate = data.get('validate', True)
        
        # 生成
        generator = MolGPTGenerator()
        molecules = generator.conditional_generate(
            properties=properties,
            n_molecules=n_molecules,
            temperature=temperature
        )
        
        # 検証
        validator = GeneratedMoleculeValidator() if validate else None
        
        results = []
        for mol in molecules:
            result = {
                "smiles": mol.smiles,
                "score": mol.score
            }
            
            if validator:
                validation = validator.validate(mol.smiles)
                result["validation"] = {
                    "is_valid": validation.is_valid,
                    "qed_score": validation.qed_score,
                    "lipinski_violations": validation.lipinski_violations
                }
            
            results.append(result)
        
        return JsonResponse({
            "molecules": results,
            "count": len(results),
            "target_properties": properties,
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Conditional generation failed: {e}")
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def text_to_molecule(request) -> JsonResponse:
    """
    自然言語→分子変換API
    
    POST /api/generate/from-text
    {
        "description": "水溶性の高い抗がん剤",
        "n_molecules": 5
    }
    
    Note:
        Phase 2でMolT5統合後に完全実装予定。
        現在は基本生成にフォールバック。
    """
    try:
        data = json.loads(request.body)
        
        description = data.get('description', '')
        n_molecules = data.get('n_molecules', 5)
        
        logger.warning(
            f"Text-to-molecule not fully implemented. "
            f"Using basic generation for: {description}"
        )
        
        # 基本生成にフォールバック
        generator = MolGPTGenerator()
        molecules = generator.generate(n_molecules=n_molecules)
        
        results = []
        for mol in molecules:
            results.append({
                "smiles": mol.smiles,
                "score": mol.score
            })
        
        return JsonResponse({
            "molecules": results,
            "description": description,
            "note": "Full text-to-molecule with MolT5 coming in Phase 2",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"Text-to-molecule failed: {e}")
        return JsonResponse({
            "error": str(e),
            "status": "error"
        }, status=500)
