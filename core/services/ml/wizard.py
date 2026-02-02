"""
ML学習ウィザード

データを分析し、最適なモデル設定を自動推奨
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WizardRecommendation:
    """ウィザード推奨結果"""
    task_type: str  # 'regression' or 'classification'
    recommended_model: str
    recommended_features: List[str]
    estimated_time: str
    estimated_accuracy: str
    reasoning: Dict[str, str]
    warnings: List[str]


class MLWizard:
    """
    機械学習ウィザード
    
    初心者でも簡単に使えるよう、データから自動で最適な設定を推奨
    
    Features:
    - タスクタイプ自動判定（回帰/分類）
    - データサイズに基づくモデル推奨
    - 特徴量組み合わせ推奨
    - 予測精度・時間見積もり
    
    Example:
        >>> wizard = MLWizard()
        >>> rec = wizard.auto_configure(df, 'target')
        >>> print(rec.recommended_model)  # 'random_forest'
    """
    
    # モデル設定テンプレート
    MODEL_CONFIGS = {
        'linear_regression': {
            'min_samples': 20,
            'max_samples': 1000,
            'speed': 'very_fast',
            'accuracy': 'low_to_medium',
            'interpretability': 'high',
            'description': 'シンプルな線形モデル。高速で解釈しやすい'
        },
        'random_forest': {
            'min_samples': 50,
            'max_samples': 10000,
            'speed': 'medium',
            'accuracy': 'high',
            'interpretability': 'medium',
            'description': '万能型。ほとんどの問題で高精度'
        },
        'gradient_boosting': {
            'min_samples': 100,
            'max_samples': 50000,
            'speed': 'slow',
            'accuracy': 'very_high',
            'interpretability': 'low',
            'description': '最高精度。時間はかかるが精度重視'
        },
        'gaussian_process': {
            'min_samples': 10,
            'max_samples': 500,
            'speed': 'slow',
            'accuracy': 'high',
            'interpretability': 'medium',
            'description': '少量データに最適。不確実性も推定'
        },
        'svm': {
            'min_samples': 50,
            'max_samples': 5000,
            'speed': 'medium',
            'accuracy': 'high',
            'interpretability': 'low',
            'description': '中規模データに適する'
        }
    }
    
    def auto_configure(
        self,
        df: pd.DataFrame,
        target_column: str,
        smiles_column: str = 'SMILES'
    ) -> WizardRecommendation:
        """
        データから自動で最適設定を推奨
        
        Args:
            df: データフレーム
            target_column: ターゲット列名
            smiles_column: SMILES列名
        
        Returns:
            推奨設定
        """
        warnings_list = []
        
        # 1. タスクタイプ判定
        task_type = self._detect_task_type(df[target_column])
        
        # 2. データサイズ確認
        n_samples = len(df)
        if n_samples < 20:
            warnings_list.append(
                f"データが少なすぎます（{n_samples}件）。最低50件を推奨"
            )
        
        # 3. モデル推奨
        model, reasoning = self._recommend_model(n_samples, task_type)
        
        # 4. 特徴量推奨
        features = self._recommend_features(n_samples, task_type)
        
        # 5. 予測精度見積もり
        est_accuracy = self._estimate_accuracy(n_samples, task_type, model)
        
        # 6. 処理時間見積もり
        est_time = self._estimate_time(n_samples, model, features)
        
        return WizardRecommendation(
            task_type=task_type,
            recommended_model=model,
            recommended_features=features,
            estimated_time=est_time,
            estimated_accuracy=est_accuracy,
            reasoning=reasoning,
            warnings=warnings_list
        )
    
    def _detect_task_type(self, target_series: pd.Series) -> str:
        """タスクタイプ自動判定"""
        unique_values = target_series.nunique()
        total_values = len(target_series)
        
        # 一意値が少ない場合は分類
        if unique_values <= 2:
            return 'classification'
        elif unique_values < 10 and unique_values / total_values < 0.05:
            return 'classification'
        else:
            return 'regression'
    
    def _recommend_model(
        self,
        n_samples: int,
        task_type: str
    ) -> Tuple[str, Dict[str, str]]:
        """データサイズとタスクに基づくモデル推奨"""
        reasoning = {}
        
        if n_samples < 50:
            model = 'gaussian_process'
            reasoning['model'] = (
                f"サンプル数が少ない（{n_samples}件）ため、"
                "少量データに適したGaussian Processを推奨"
            )
        
        elif n_samples < 500:
            model = 'random_forest'
            reasoning['model'] = (
                f"サンプル数が中規模（{n_samples}件）で、"
                "Random Forestがバランス良く高精度"
            )
        
        elif n_samples < 5000:
            if task_type == 'classification':
                model = 'gradient_boosting'
                reasoning['model'] = (
                    "十分なデータ量があり、分類タスクなので"
                    "Gradient Boostingで最高精度を狙えます"
                )
            else:
                model = 'random_forest'
                reasoning['model'] = (
                    "回帰タスクで安定した予測が期待できる"
                    "Random Forestを推奨"
                )
        
        else:
            model = 'gradient_boosting'
            reasoning['model'] = (
                f"大規模データ（{n_samples}件）なので、"
                "Gradient Boostingで最高精度を目指します"
            )
        
        return model, reasoning
    
    def _recommend_features(
        self,
        n_samples: int,
        task_type: str
    ) -> List[str]:
        """特徴量組み合わせ推奨"""
        features = []
        
        # 基本: Morgan指紋は必須
        features.append('morgan_fp')
        
        # RDKit 2D記述子（計算が高速）
        features.append('rdkit_2d')
        
        # データ量が多い場合は高度な特徴量も
        if n_samples >= 500:
            features.append('molecular_descriptors')
        
        # 分類タスクの場合はアラート情報も有用
        if task_type == 'classification':
            features.append('structural_alerts')
        
        return features
    
    def _estimate_accuracy(
        self,
        n_samples: int,
        task_type: str,
        model: str
    ) -> str:
        """予測精度見積もり"""
        base_scores = {
            'linear_regression': 0.60,
            'random_forest': 0.75,
            'gradient_boosting': 0.85,
            'gaussian_process': 0.70,
            'svm': 0.72
        }
        
        base_score = base_scores.get(model, 0.70)
        
        # データサイズによる補正
        if n_samples < 50:
            base_score -= 0.15
        elif n_samples < 200:
            base_score -= 0.05
        elif n_samples > 1000:
            base_score += 0.05
        
        # タスクタイプによる補正
        if task_type == 'classification':
            metric = 'AUC'
            base_score = min(0.95, base_score + 0.05)
        else:
            metric = 'R²'
        
        return f"{metric}={base_score:.2f}"
    
    def _estimate_time(
        self,
        n_samples: int,
        model: str,
        features: List[str]
    ) -> str:
        """処理時間見積もり"""
        # 基本時間（秒）
        base_times = {
            'linear_regression': 5,
            'random_forest': 30,
            'gradient_boosting': 120,
            'gaussian_process': 60,
            'svm': 45
        }
        
        base_time = base_times.get(model, 30)
        
        # サンプル数による補正
        time_sec = base_time * (n_samples / 100)
        
        # 特徴量数による補正
        time_sec *= (1 + len(features) * 0.2)
        
        # 時間を分単位に変換
        if time_sec < 60:
            return f"約{int(time_sec)}秒"
        elif time_sec < 600:
            return f"約{int(time_sec/60)}分"
        else:
            return f"約{int(time_sec/60)}分（{int(time_sec/3600)}時間）"


def format_recommendation_for_ui(rec: WizardRecommendation) -> Dict:
    """UI表示用に推奨結果をフォーマット"""
    return {
        "task_type": rec.task_type,
        "task_type_display": "分類問題" if rec.task_type == 'classification' else "回帰問題",
        "model": rec.recommended_model,
        "model_display": _get_model_display_name(rec.recommended_model),
        "features": rec.recommended_features,
        "features_display": [_get_feature_display_name(f) for f in rec.recommended_features],
        "estimated_time": rec.estimated_time,
        "estimated_accuracy": rec.estimated_accuracy,
        "reasoning": rec.reasoning,
        "warnings": rec.warnings
    }


def _get_model_display_name(model: str) -> str:
    """モデル表示名"""
    names = {
        'linear_regression': '線形回帰（高速・シンプル）',
        'random_forest': 'ランダムフォレスト（万能型）',
        'gradient_boosting': '勾配ブースティング（最高精度）',
        'gaussian_process': 'ガウス過程（少量データ向け）',
        'svm': 'サポートベクターマシン（中規模データ向け）'
    }
    return names.get(model, model)


def _get_feature_display_name(feature: str) -> str:
    """特徴量表示名"""
    names = {
        'morgan_fp': 'Morgan指紋（分子の部分構造）',
        'rdkit_2d': 'RDKit 2D記述子（基本物性）',
        'molecular_descriptors': '分子記述子（詳細な物性）',
        'structural_alerts': '構造アラート（毒性など）'
    }
    return names.get(feature, feature)
