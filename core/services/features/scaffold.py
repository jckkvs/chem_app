"""
スキャフォールド分析

Implements: F-SCAFFOLD-001
設計思想:
- 骨格抽出
- スキャフォールドホッピング
- 多様性分析
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ScaffoldInfo:
    """スキャフォールド情報"""
    smiles: str
    scaffold: str
    generic_scaffold: str
    ring_count: int


@dataclass
class ScaffoldAnalysis:
    """スキャフォールド分析結果"""
    scaffolds: Dict[str, List[str]]  # scaffold -> [smiles]
    scaffold_counts: Dict[str, int]
    diversity_score: float
    most_common: List[tuple]


class ScaffoldAnalyzer:
    """
    スキャフォールド分析
    
    Features:
    - Murckoスキャフォールド抽出
    - スキャフォールド多様性
    - クラスタリング
    
    Example:
        >>> analyzer = ScaffoldAnalyzer()
        >>> analysis = analyzer.analyze(smiles_list)
    """
    
    def get_scaffold(
        self,
        smiles: str,
        generic: bool = False,
    ) -> Optional[ScaffoldInfo]:
        """スキャフォールドを抽出"""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Murckoスキャフォールド
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
            
            # ジェネリックスキャフォールド（側鎖なし）
            generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            generic_smiles = Chem.MolToSmiles(generic_mol)
            
            from rdkit.Chem import rdMolDescriptors
            ring_count = rdMolDescriptors.CalcNumRings(mol)
            
            return ScaffoldInfo(
                smiles=smiles,
                scaffold=scaffold_smiles,
                generic_scaffold=generic_smiles,
                ring_count=ring_count,
            )
            
        except Exception as e:
            logger.warning(f"Scaffold extraction failed: {e}")
            return None
    
    def analyze(
        self,
        smiles_list: List[str],
        use_generic: bool = True,
    ) -> ScaffoldAnalysis:
        """スキャフォールド分析"""
        scaffolds: Dict[str, List[str]] = {}
        
        for smiles in smiles_list:
            info = self.get_scaffold(smiles, generic=use_generic)
            if info:
                scaffold = info.generic_scaffold if use_generic else info.scaffold
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = []
                scaffolds[scaffold].append(smiles)
        
        # カウント
        scaffold_counts = {s: len(mols) for s, mols in scaffolds.items()}
        
        # 多様性スコア（ユニークスキャフォールド比率）
        n_unique = len(scaffolds)
        n_total = len(smiles_list)
        diversity_score = n_unique / n_total if n_total > 0 else 0
        
        # 最も多いスキャフォールド
        most_common = Counter(scaffold_counts).most_common(10)
        
        return ScaffoldAnalysis(
            scaffolds=scaffolds,
            scaffold_counts=scaffold_counts,
            diversity_score=diversity_score,
            most_common=most_common,
        )
    
    def scaffold_hopping(
        self,
        smiles: str,
        database_smiles: List[str],
        same_ring_count: bool = True,
    ) -> List[str]:
        """スキャフォールドホッピング候補を探索"""
        source_info = self.get_scaffold(smiles)
        if source_info is None:
            return []
        
        candidates = []
        
        for db_smiles in database_smiles:
            db_info = self.get_scaffold(db_smiles)
            if db_info is None:
                continue
            
            # 異なるスキャフォールドで同じ環数
            if db_info.scaffold != source_info.scaffold:
                if not same_ring_count or db_info.ring_count == source_info.ring_count:
                    candidates.append(db_smiles)
        
        return candidates
