"""
コンフィグレーションウィザード

Implements: F-WIZARD-001
設計思想:
- インタラクティブ設定
- テンプレート
- ベストプラクティス提案
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WizardStep:
    """ウィザードステップ"""
    name: str
    description: str
    options: List[str]
    default: str = ""
    selected: Optional[str] = None


@dataclass
class WizardConfig:
    """ウィザード結果設定"""
    task_type: str
    model_type: str
    feature_types: List[str]
    preprocessing: Dict[str, Any]
    training: Dict[str, Any]


class ConfigurationWizard:
    """
    コンフィグレーションウィザード
    
    Features:
    - ステップバイステップ設定
    - ベストプラクティステンプレート
    - 自動推奨
    
    Example:
        >>> wizard = ConfigurationWizard()
        >>> config = wizard.quick_setup(data_type='smiles', target='activity')
    """
    
    TEMPLATES = {
        'activity_prediction': {
            'task_type': 'regression',
            'model_type': 'lightgbm',
            'feature_types': ['rdkit', 'morgan'],
            'preprocessing': {'scaling': True, 'feature_selection': True},
            'training': {'cv_folds': 5, 'early_stopping': True},
        },
        'toxicity_classification': {
            'task_type': 'classification',
            'model_type': 'xgboost',
            'feature_types': ['rdkit', 'maccs'],
            'preprocessing': {'scaling': True, 'smote': True},
            'training': {'cv_folds': 5, 'class_weight': 'balanced'},
        },
        'property_regression': {
            'task_type': 'regression',
            'model_type': 'random_forest',
            'feature_types': ['rdkit'],
            'preprocessing': {'scaling': False, 'outlier_removal': True},
            'training': {'cv_folds': 3},
        },
    }
    
    def __init__(self):
        self.steps: List[WizardStep] = []
        self.current_step: int = 0
    
    def quick_setup(
        self,
        data_type: str = 'smiles',
        target: str = 'activity',
        data_size: int = 1000,
    ) -> WizardConfig:
        """クイックセットアップ"""
        # テンプレート選択
        if 'tox' in target.lower():
            template = self.TEMPLATES['toxicity_classification']
        elif data_size < 500:
            template = self.TEMPLATES['property_regression']
        else:
            template = self.TEMPLATES['activity_prediction']
        
        # データサイズに応じた調整
        if data_size < 200:
            template['training']['cv_folds'] = 3
            template['model_type'] = 'random_forest'
        elif data_size > 5000:
            template['training']['cv_folds'] = 10
        
        return WizardConfig(**template)
    
    def get_recommendation(
        self,
        n_samples: int,
        n_features: int,
        task_type: str,
    ) -> Dict[str, Any]:
        """推奨設定を取得"""
        recommendations = {
            'model_type': 'lightgbm',
            'preprocessing': {},
            'training': {},
        }
        
        # サンプル数に応じた推奨
        if n_samples < 100:
            recommendations['model_type'] = 'random_forest'
            recommendations['training']['cv_folds'] = 3
            recommendations['note'] = 'データ数が少ないため、シンプルなモデルを推奨'
        elif n_samples < 500:
            recommendations['training']['cv_folds'] = 5
        else:
            recommendations['training']['cv_folds'] = 10
            recommendations['training']['early_stopping'] = True
        
        # 特徴量数に応じた推奨
        if n_features > 1000:
            recommendations['preprocessing']['feature_selection'] = True
            recommendations['preprocessing']['n_features'] = 200
        
        return recommendations
    
    def generate_config_file(self, config: WizardConfig) -> str:
        """設定ファイル生成"""
        import json
        
        config_dict = {
            'task_type': config.task_type,
            'model': {
                'type': config.model_type,
            },
            'features': config.feature_types,
            'preprocessing': config.preprocessing,
            'training': config.training,
        }
        
        return json.dumps(config_dict, indent=2)
    
    def validate_config(self, config: WizardConfig) -> List[str]:
        """設定を検証"""
        warnings = []
        
        if config.task_type == 'classification' and config.model_type not in ['xgboost', 'lightgbm', 'random_forest']:
            warnings.append(f"分類タスクに {config.model_type} は推奨されません")
        
        if 'xtb' in config.feature_types and len(config.feature_types) > 3:
            warnings.append("XTB特徴量は計算コストが高いです")
        
        return warnings
