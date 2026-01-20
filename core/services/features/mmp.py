"""
マッチド分子ペア分析（MMP）

Implements: F-MMP-001
設計思想:
- 構造変換抽出
- 活性変化予測
- SAR分析
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class MolecularPair:
    """分子ペア"""
    smiles1: str
    smiles2: str
    transformation: str
    property_change: float
    core: str


class MatchedMolecularPairAnalyzer:
    """
    マッチド分子ペア分析
    
    Features:
    - 構造変換検出
    - 活性変化分析
    - SAR知識抽出
    
    Example:
        >>> mmp = MatchedMolecularPairAnalyzer()
        >>> pairs = mmp.find_pairs(smiles_list, activities)
    """
    
    def find_pairs(
        self,
        smiles_list: List[str],
        activities: Optional[List[float]] = None,
        min_similarity: float = 0.7,
    ) -> List[MolecularPair]:
        """ペアを検出"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
            
            mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols if m]
            
            pairs = []
            
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    
                    if sim >= min_similarity and sim < 1.0:
                        prop_change = 0.0
                        if activities and i < len(activities) and j < len(activities):
                            prop_change = activities[j] - activities[i]
                        
                        transformation = self._detect_transformation(
                            smiles_list[i], smiles_list[j]
                        )
                        
                        pairs.append(MolecularPair(
                            smiles1=smiles_list[i],
                            smiles2=smiles_list[j],
                            transformation=transformation,
                            property_change=prop_change,
                            core="",
                        ))
            
            return pairs
            
        except Exception as e:
            logger.error(f"MMP analysis failed: {e}")
            return []
    
    def _detect_transformation(self, smiles1: str, smiles2: str) -> str:
        """構造変換を検出"""
        # 簡易的な差分検出
        set1 = set(smiles1)
        set2 = set(smiles2)
        
        added = set2 - set1
        removed = set1 - set2
        
        if added and removed:
            return f"-{''.join(removed)}+{''.join(added)}"
        elif added:
            return f"+{''.join(added)}"
        elif removed:
            return f"-{''.join(removed)}"
        return "similar"
    
    def summarize_transformations(
        self,
        pairs: List[MolecularPair],
    ) -> Dict[str, float]:
        """変換ごとの効果を集計"""
        transform_effects: Dict[str, List[float]] = {}
        
        for pair in pairs:
            if pair.transformation not in transform_effects:
                transform_effects[pair.transformation] = []
            transform_effects[pair.transformation].append(pair.property_change)
        
        return {
            t: sum(effects) / len(effects)
            for t, effects in transform_effects.items()
        }
    
    def suggest_modifications(
        self,
        smiles: str,
        pairs: List[MolecularPair],
        target_improvement: float = 0,
    ) -> List[str]:
        """改善提案"""
        beneficial = [
            p for p in pairs
            if p.property_change > target_improvement
        ]
        
        suggestions = []
        for pair in beneficial[:5]:
            suggestions.append(
                f"変換 {pair.transformation}: +{pair.property_change:.2f}"
            )
        
        return suggestions
