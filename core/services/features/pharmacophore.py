"""
ファーマコフォア生成（Schrödinger Phase inspired）

Implements: F-PHARMACOPHORE-001
設計思想:
- 3D薬理活性特徴抽出
- ファーマコフォア記述子
- 類似度スクリーニング
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PharmacophoreFeature:
    """ファーマコフォア特徴"""
    feature_type: str  # 'donor', 'acceptor', 'hydrophobic', 'aromatic', 'positive', 'negative'
    position: tuple  # (x, y, z)
    atom_indices: List[int]
    
    def distance_to(self, other: 'PharmacophoreFeature') -> float:
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other.position)))


@dataclass
class Pharmacophore:
    """ファーマコフォアモデル"""
    smiles: str
    features: List[PharmacophoreFeature] = field(default_factory=list)
    
    def get_feature_counts(self) -> Dict[str, int]:
        counts = {}
        for f in self.features:
            counts[f.feature_type] = counts.get(f.feature_type, 0) + 1
        return counts
    
    def get_distance_matrix(self) -> np.ndarray:
        n = len(self.features)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.features[i].distance_to(self.features[j])
                matrix[i, j] = d
                matrix[j, i] = d
        return matrix


class PharmacophoreGenerator:
    """
    ファーマコフォア生成（Phase/MOE inspired）
    
    Features:
    - H-bond donor/acceptor検出
    - 疎水性/芳香族中心
    - 正/負電荷中心
    - ファーマコフォアフィンガープリント
    
    Example:
        >>> gen = PharmacophoreGenerator()
        >>> pharm = gen.generate("c1ccccc1O")
    """
    
    # SMARTSパターン
    PATTERNS = {
        'donor': '[#7,#8,#16][H]',
        'acceptor': '[#7,#8,#16;!H0]',
        'hydrophobic': '[C,c,F,Cl,Br,I,S;!$(C=[O,N,S])]',
        'aromatic': 'a',
        'positive': '[+1,+2,+3]',
        'negative': '[-1,-2,-3]',
    }
    
    def generate(self, smiles: str) -> Optional[Pharmacophore]:
        """ファーマコフォアを生成"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 3D座標生成
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            
            features = []
            
            for ftype, smarts in self.PATTERNS.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    continue
                
                matches = mol.GetSubstructMatches(pattern)
                
                for match in matches:
                    # 中心座標を計算
                    positions = [conf.GetAtomPosition(idx) for idx in match]
                    center = (
                        np.mean([p.x for p in positions]),
                        np.mean([p.y for p in positions]),
                        np.mean([p.z for p in positions]),
                    )
                    
                    features.append(PharmacophoreFeature(
                        feature_type=ftype,
                        position=center,
                        atom_indices=list(match),
                    ))
            
            return Pharmacophore(smiles=smiles, features=features)
            
        except Exception as e:
            logger.error(f"Pharmacophore generation failed: {e}")
            return None
    
    def get_fingerprint(self, pharm: Pharmacophore, n_bits: int = 256) -> np.ndarray:
        """ファーマコフォアフィンガープリント"""
        fp = np.zeros(n_bits, dtype=np.int8)
        
        counts = pharm.get_feature_counts()
        distances = pharm.get_distance_matrix()
        
        # 特徴カウントをエンコード
        type_indices = {
            'donor': 0, 'acceptor': 1, 'hydrophobic': 2,
            'aromatic': 3, 'positive': 4, 'negative': 5
        }
        
        for ftype, count in counts.items():
            if ftype in type_indices:
                idx = type_indices[ftype]
                fp[idx * 10:(idx + 1) * 10] = min(count, 10)
        
        # 距離をエンコード
        if len(distances) > 0:
            flat_distances = distances[np.triu_indices_from(distances, k=1)]
            for i, d in enumerate(flat_distances[:50]):
                fp[60 + i] = int(d / 2) % 10
        
        return fp
    
    def similarity(self, pharm1: Pharmacophore, pharm2: Pharmacophore) -> float:
        """ファーマコフォア類似度"""
        fp1 = self.get_fingerprint(pharm1)
        fp2 = self.get_fingerprint(pharm2)
        
        # コサイン類似度
        dot = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
