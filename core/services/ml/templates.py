"""
プリセットモデルテンプレート

用途別の最適なML設定を提供
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelTemplate:
    """
    モデルテンプレート
    
    用途に応じた推奨設定を定義
    """
    id: str
    name: str
    name_ja: str
    description: str
    icon: str  # 絵文字
    
    # 推奨設定
    model_type: str  # 'random_forest', 'gradient_boosting', 'lightgbm'
    task_type: str  # 'regression', 'classification'
    features: List[str]  # ['rdkit', 'morgan_fp']
    
    # 詳細情報
    use_cases: List[str] = field(default_factory=list)  # 適用例
    recommended_for: List[str] = field(default_factory=list)  # 推奨データタイプ
    pros: List[str] = field(default_factory=list)  # 長所
    cons: List[str] = field(default_factory=list)  # 短所
    
    # パラメータ
    model_params: Dict[str, Any] = field(default_factory=dict)
    preprocessing_preset: str = 'tree_optimized'
    
    # メタデータ
    difficulty: str = 'beginner'  # 'beginner', 'intermediate', 'advanced'
    estimated_time: str = '5-10分'


class TemplateManager:
    """
    テンプレート管理
    
    用途別のプリセットテンプレートを提供
    
    Example:
        >>> manager = TemplateManager()
        >>> templates = manager.list_all()
        >>> template = manager.get('drug_discovery_basic')
        >>> config = manager.apply_to_config('toxicity_screening')
    """
    
    TEMPLATES: Dict[str, ModelTemplate] = {
        'drug_discovery_basic': ModelTemplate(
            id='drug_discovery_basic',
            name='Drug Discovery Basic',
            name_ja='創薬基礎',
            description='医薬品候補化合物の基本的な物性予測',
            icon='🧪',
            model_type='random_forest',
            task_type='regression',
            features=['rdkit'],
            use_cases=[
                'logP予測',
                '溶解度予測',
                '分子量計算',
                'QED（医薬品らしさ）予測'
            ],
            recommended_for=[
                '初学者向け',
                '安定した結果が必要',
                '解釈性重視'
            ],
            pros=[
                '結果が安定',
                '解釈しやすい',
                '過学習しにくい',
                '学習が速い'
            ],
            cons=[
                '非常に複雑なパターンには不向き',
                '深層学習ほどの精度は出ない'
            ],
            model_params={
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            difficulty='beginner',
            estimated_time='3-5分'
        ),
        
        'toxicity_screening': ModelTemplate(
            id='toxicity_screening',
            name='Toxicity Screening',
            name_ja='毒性スクリーニング',
            description='化合物の毒性リスク評価',
            icon='☠️',
            model_type='lightgbm',
            task_type='classification',
            features=['rdkit'],
            use_cases=[
                'Tox21毒性予測',
                'AMES変異原性',
                'hERG阻害予測',
                '肝毒性評価'
            ],
            recommended_for=[
                '毒性評価',
                '不均衡データ',
                '高精度が必要'
            ],
            pros=[
                '高精度',
                '不均衡データに強い',
                '特徴量重要度で原因特定可能',
                'クラス重み付け対応'
            ],
            cons=[
                '学習時間がやや長い',
                'パラメータ調整が必要な場合あり'
            ],
            model_params={
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 8,
                'random_state': 42,
                'class_weight': 'balanced'  # 不均衡データ対応
            },
            difficulty='intermediate',
            estimated_time='5-10分'
        ),
        
        'solubility_prediction': ModelTemplate(
            id='solubility_prediction',
            name='Solubility Prediction',
            name_ja='溶解度予測',
            description='水溶解度・脂溶性予測',
            icon='💧',
            model_type='lightgbm',
            task_type='regression',
            features=['rdkit'],
            use_cases=[
                '水溶解度予測',
                'logP（脂溶性）',
                'logD',
                'logS'
            ],
            recommended_for=[
                '溶解度予測',
                '非線形性の高いデータ',
                '高精度が必要'
            ],
            pros=[
                '非線形関係を捕捉',
                '高精度',
                '分子サイズ・極性を考慮',
                '外れ値に強い'
            ],
            cons=[
                '学習時間が長い',
                'ハイパーパラメータ調整推奨'
            ],
            model_params={
                'n_estimators': 300,
                'learning_rate': 0.03,
                'max_depth': 12,
                'num_leaves': 31,
                'random_state': 42
            },
            difficulty='intermediate',
            estimated_time='7-12分'
        ),
        
        'adme_prediction': ModelTemplate(
            id='adme_prediction',
            name='ADME Prediction',
            name_ja='ADME予測',
            description='薬物動態パラメータ予測（吸収・分布・代謝・排泄）',
            icon='💊',
            model_type='random_forest',
            task_type='regression',
            features=['rdkit'],
            use_cases=[
                'Caco-2透過性',
                '血漿タンパク結合率',
                'クリアランス',
                '生物学的利用能'
            ],
            recommended_for=[
                'ADME予測',
                '複数パラメータ',
                '生理学的解釈重視'
            ],
            pros=[
                '複数のパラメータに適用可能',
                '安定した予測',
                '生理学的意味を持つ特徴量重要度',
                'ロバスト'
            ],
            cons=[
                'ごく高精度は期待しにくい',
                'ディープラーニングには劣る場合も'
            ],
            model_params={
                'n_estimators': 150,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42
            },
            difficulty='intermediate',
            estimated_time='5-8分'
        ),
        
        'activity_prediction': ModelTemplate(
            id='activity_prediction',
            name='Activity Prediction',
            name_ja='活性予測',
            description='タンパク質結合活性・IC50予測',
            icon='🎯',
            model_type='lightgbm',
            task_type='regression',
            features=['rdkit'],
            use_cases=[
                'IC50予測',
                'Ki値予測',
                '阻害率予測',
                'EC50予測'
            ],
            recommended_for=[
                '活性予測',
                '構造-活性相関',
                'ターゲット特異的'
            ],
            pros=[
                '構造-活性相関の捕捉',
                '複雑なパターン認識',
                '高精度',
                'ターゲット特異性を学習'
            ],
            cons=[
                'データ量が必要（>100サンプル推奨）',
                '学習時間がやや長い'
            ],
            model_params={
                'n_estimators': 250,
                'learning_rate': 0.04,
                'max_depth': 10,
                'num_leaves': 31,
                'random_state': 42
            },
            difficulty='intermediate',
            estimated_time='6-10分'
        ),
        
        'fast_screening': ModelTemplate(
            id='fast_screening',
            name='Fast Screening',
            name_ja='高速スクリーニング',
            description='大規模化合物ライブラリの初期スクリーニング',
            icon='⚡',
            model_type='random_forest',  # 軽量版
            task_type='regression',
            features=['rdkit'],  # 最小限
            use_cases=[
                '数万化合物の粗スクリーニング',
                '初期フィルタリング',
                'バーチャルスクリーニング',
                '高速予測'
            ],
            recommended_for=[
                '大規模データ',
                '速度重視',
                '初期スクリーニング'
            ],
            pros=[
                '高速処理',
                'メモリ効率が良い',
                'スケーラブル',
                '数百万化合物対応'
            ],
            cons=[
                '精度は中程度',
                '複雑なパターンは捉えにくい',
                '詳細予測には不向き'
            ],
            model_params={
                'n_estimators': 50,  # 少なめ
                'max_depth': 8,  # 浅め
                'n_jobs': -1,
                'random_state': 42
            },
            difficulty='beginner',
            estimated_time='1-3分'
        ),
    }
    
    @classmethod
    def list_all(cls) -> List[ModelTemplate]:
        """すべてのテンプレートをリスト"""
        return list(cls.TEMPLATES.values())
    
    @classmethod
    def get(cls, template_id: str) -> Optional[ModelTemplate]:
        """テンプレート取得"""
        return cls.TEMPLATES.get(template_id)
    
    @classmethod
    def list_by_difficulty(cls, difficulty: str) -> List[ModelTemplate]:
        """難易度別にフィルタ"""
        return [
            t for t in cls.TEMPLATES.values()
            if t.difficulty == difficulty
        ]
    
    @classmethod
    def list_by_task_type(cls, task_type: str) -> List[ModelTemplate]:
        """タスクタイプ別にフィルタ"""
        return [
            t for t in cls.TEMPLATES.values()
            if t.task_type == task_type
        ]
    
    @classmethod
    def apply_to_config(cls, template_id: str) -> Dict[str, Any]:
        """
        テンプレートをExperiment設定に変換
        
        Args:
            template_id: テンプレートID
        
        Returns:
            Experiment config dict
        
        Raises:
            ValueError: 存在しないテンプレートID
        """
        template = cls.get(template_id)
        if template is None:
            available = ', '.join(cls.TEMPLATES.keys())
            raise ValueError(
                f"Template '{template_id}' not found. "
                f"Available: {available}"
            )
        
        config = {
            'model_type': template.model_type,
            'task_type': template.task_type,
            'features': template.features,
            'cv_folds': 5,
            'task_type_mode': 'smiles_only',
            'use_smart_engine': True,
            'target_property': cls._infer_target_property(template),
            'model_params': template.model_params,
            'preprocessing_preset': template.preprocessing_preset
        }
        
        return config
    
    @classmethod
    def _infer_target_property(cls, template: ModelTemplate) -> str:
        """テンプレートからtarget_propertyを推測"""
        # テンプレートIDから推測
        property_map = {
            'solubility_prediction': 'solubility',
            'toxicity_screening': 'toxicity',
            'drug_discovery_basic': 'general',
            'adme_prediction': 'adme',
            'activity_prediction': 'activity',
            'fast_screening': 'general'
        }
        return property_map.get(template.id, 'general')
    
    @classmethod
    def get_summary(cls) -> Dict[str, int]:
        """テンプレート統計"""
        all_templates = cls.list_all()
        return {
            'total': len(all_templates),
            'regression': len([t for t in all_templates if t.task_type == 'regression']),
            'classification': len([t for t in all_templates if t.task_type == 'classification']),
            'beginner': len([t for t in all_templates if t.difficulty == 'beginner']),
            'intermediate': len([t for t in all_templates if t.difficulty == 'intermediate']),
            'advanced': len([t for t in all_templates if t.difficulty == 'advanced']),
        }
