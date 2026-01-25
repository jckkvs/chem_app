"""
合成可能性評価（SAScore/SCScore inspired）

Implements: F-SYNTH-001
設計思想:
- 合成可能性スコア
- 複雑度評価
- 合成経路示唆
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass  
class SynthesizabilityResult:
    """合成可能性結果"""
    smiles: str
    sa_score: float  # 1-10 (低いほど合成しやすい)
    complexity_score: float
    is_synthesizable: bool
    difficulty: str  # 'easy', 'moderate', 'hard', 'very_hard'
    structural_alerts: List[str]


class SynthesizabilityAssessor:
    """
    合成可能性評価（SA Score inspired）
    
    Features:
    - SAスコア計算
    - 構造複雑度評価
    - 合成難易度分類
    
    Example:
        >>> assessor = SynthesizabilityAssessor()
        >>> result = assessor.assess("CCO")
    """
    
    # 難しい部分構造
    DIFFICULT_FRAGMENTS = [
        ('[C@@H]', 'キラル中心'),
        ('[C@H]', 'キラル中心'),
        ('C#C', 'アルキン'),
        ('C=C=C', 'アレン'),
        ('[N+]([O-])=O', 'ニトロ基'),
        ('c1ccc2c(c1)ccc1ccccc12', '縮合芳香環'),
    ]
    
    def assess(self, smiles: str) -> Optional[SynthesizabilityResult]:
        """合成可能性を評価"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 複雑度因子
            n_atoms = mol.GetNumHeavyAtoms()
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            
            # SAスコア計算（簡易版）
            complexity = (
                n_atoms * 0.1 +
                n_rings * 0.5 +
                n_stereo * 1.0 +
                n_rotatable * 0.1
            )
            
            # スコアを1-10に正規化
            sa_score = 1 + min(9, complexity)
            
            # 難しい構造のチェック
            alerts = []
            for pattern, name in self.DIFFICULT_FRAGMENTS:
                patt_mol = Chem.MolFromSmarts(pattern)
                if patt_mol and mol.HasSubstructMatch(patt_mol):
                    alerts.append(name)
                    sa_score += 0.5
            
            sa_score = min(10, sa_score)
            
            # 難易度分類
            if sa_score <= 3:
                difficulty = 'easy'
            elif sa_score <= 5:
                difficulty = 'moderate'
            elif sa_score <= 7:
                difficulty = 'hard'
            else:
                difficulty = 'very_hard'
            
            return SynthesizabilityResult(
                smiles=smiles,
                sa_score=sa_score,
                complexity_score=complexity,
                is_synthesizable=sa_score < 6,
                difficulty=difficulty,
                structural_alerts=alerts,
            )
            
        except Exception as e:
            logger.error(f"Synthesizability assessment failed: {e}")
            return None
    
    def rank_by_synthesizability(
        self,
        smiles_list: List[str],
    ) -> List[SynthesizabilityResult]:
        """合成容易さでランキング"""
        results = []
        
        for smiles in smiles_list:
            result = self.assess(smiles)
            if result:
                results.append(result)
        
        return sorted(results, key=lambda x: x.sa_score)
