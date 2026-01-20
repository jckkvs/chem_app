"""
説明生成（NLP for Chemistry）

Implements: F-EXPLAIN-GEN-001
設計思想:
- 予測結果の自然言語説明
- 特徴量重要度の解釈
- レポート自動生成
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """説明"""
    summary: str
    details: List[str]
    confidence: float
    key_factors: List[str]


class ExplanationGenerator:
    """
    説明生成エンジン
    
    Features:
    - 予測結果の自然言語化
    - 特徴量重要度の解釈
    - ドメイン知識の組み込み
    
    Example:
        >>> gen = ExplanationGenerator()
        >>> explanation = gen.generate(prediction, feature_contributions)
    """
    
    # 化学ドメイン用語
    FEATURE_DESCRIPTIONS = {
        'MolWt': '分子量',
        'LogP': '脂溶性（LogP）',
        'TPSA': '極性表面積',
        'NumHDonors': '水素結合ドナー数',
        'NumHAcceptors': '水素結合アクセプター数',
        'NumRotatableBonds': '回転可能結合数',
        'RingCount': '環数',
        'AromaticProportion': '芳香族比率',
    }
    
    PROPERTY_TEMPLATES = {
        'solubility': {
            'high': '高い溶解性が予測されます。極性官能基が溶解性に寄与しています。',
            'low': '低い溶解性が予測されます。疎水性が高く、水への溶解が困難です。',
        },
        'activity': {
            'high': '高い活性が予測されます。ターゲットとの相互作用が期待されます。',
            'low': '低い活性が予測されます。構造最適化が必要かもしれません。',
        },
    }
    
    def generate(
        self,
        prediction: float,
        feature_contributions: Dict[str, float],
        property_type: str = 'activity',
        threshold: float = 0.5,
    ) -> Explanation:
        """説明を生成"""
        # 寄与度でソート
        sorted_contribs = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        
        # 上位因子
        top_positive = [(k, v) for k, v in sorted_contribs if v > 0][:3]
        top_negative = [(k, v) for k, v in sorted_contribs if v < 0][:3]
        
        # サマリー生成
        level = 'high' if prediction > threshold else 'low'
        summary = self.PROPERTY_TEMPLATES.get(property_type, {}).get(
            level, f"予測値: {prediction:.3f}"
        )
        
        # 詳細生成
        details = []
        
        if top_positive:
            pos_factors = [self._describe_feature(k, v) for k, v in top_positive]
            details.append(f"正の寄与: {', '.join(pos_factors)}")
        
        if top_negative:
            neg_factors = [self._describe_feature(k, v) for k, v in top_negative]
            details.append(f"負の寄与: {', '.join(neg_factors)}")
        
        key_factors = [k for k, v in sorted_contribs[:5]]
        
        return Explanation(
            summary=summary,
            details=details,
            confidence=min(1.0, abs(prediction - threshold) * 2),
            key_factors=key_factors,
        )
    
    def _describe_feature(self, feature: str, contribution: float) -> str:
        """特徴量を説明"""
        name = self.FEATURE_DESCRIPTIONS.get(feature, feature)
        direction = '増加' if contribution > 0 else '減少'
        return f"{name}が{direction}に寄与"
    
    def generate_report(
        self,
        smiles: str,
        predictions: Dict[str, float],
        contributions: Dict[str, Dict[str, float]],
    ) -> str:
        """レポート生成"""
        lines = [
            f"# 分子解析レポート",
            f"",
            f"**SMILES**: `{smiles}`",
            f"",
            f"## 予測結果",
        ]
        
        for prop, pred in predictions.items():
            lines.append(f"- **{prop}**: {pred:.4f}")
            
            if prop in contributions:
                explanation = self.generate(pred, contributions[prop], prop)
                lines.append(f"  - {explanation.summary}")
        
        return "\n".join(lines)
