"""
形状類似性計算（OpenEye ROCS inspired）

Implements: F-SHAPE-001
設計思想:
- 3D形状オーバーラップ
- 形状フィンガープリント
- ファーマコフォア形状マッチング
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShapeResult:
    """形状比較結果"""
    smiles1: str
    smiles2: str
    shape_similarity: float  # 0-1
    volume_overlap: float
    centroid_distance: float


class ShapeSimilarity:
    """
    形状類似性計算（ROCS/USR inspired）
    
    Features:
    - USR (Ultrafast Shape Recognition)
    - 3D形状記述子
    - 形状ベース検索
    
    Example:
        >>> shape = ShapeSimilarity()
        >>> result = shape.compare("CCO", "CCCO")
    """
    
    def get_usr_descriptor(self, smiles: str) -> Optional[np.ndarray]:
        """USR記述子を計算"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors3D
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            
            # 原子座標取得
            coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])
            
            if len(coords) == 0:
                return None
            
            # USR記述子（4基準点からの距離統計）
            centroid = coords.mean(axis=0)
            
            # 重心からの距離
            d_ctd = np.linalg.norm(coords - centroid, axis=1)
            
            # 最遠点
            farthest_idx = np.argmax(d_ctd)
            farthest = coords[farthest_idx]
            d_fct = np.linalg.norm(coords - farthest, axis=1)
            
            # 最遠点から最遠
            farthest2_idx = np.argmax(d_fct)
            farthest2 = coords[farthest2_idx]
            d_ftf = np.linalg.norm(coords - farthest2, axis=1)
            
            # 統計量（mean, std, skew）
            descriptor = []
            for d in [d_ctd, d_fct, d_ftf]:
                descriptor.extend([
                    d.mean(),
                    d.std(),
                    ((d - d.mean()) ** 3).mean() / (d.std() ** 3 + 1e-9),  # skewness
                ])
            
            return np.array(descriptor)
            
        except Exception as e:
            logger.warning(f"USR calculation failed: {e}")
            return None
    
    def compare(self, smiles1: str, smiles2: str) -> Optional[ShapeResult]:
        """2分子の形状類似性を計算"""
        usr1 = self.get_usr_descriptor(smiles1)
        usr2 = self.get_usr_descriptor(smiles2)
        
        if usr1 is None or usr2 is None:
            return None
        
        # USR類似度（Manhattan距離の逆数）
        distance = np.abs(usr1 - usr2).sum()
        similarity = 1.0 / (1.0 + distance / 12)  # 正規化
        
        return ShapeResult(
            smiles1=smiles1,
            smiles2=smiles2,
            shape_similarity=float(similarity),
            volume_overlap=float(similarity),  # 簡易近似
            centroid_distance=float(distance),
        )
    
    def search_similar(
        self,
        query_smiles: str,
        database_smiles: List[str],
        top_k: int = 5,
    ) -> List[ShapeResult]:
        """形状類似分子を検索"""
        query_usr = self.get_usr_descriptor(query_smiles)
        if query_usr is None:
            return []
        
        results = []
        for smi in database_smiles:
            result = self.compare(query_smiles, smi)
            if result:
                results.append(result)
        
        results.sort(key=lambda x: x.shape_similarity, reverse=True)
        return results[:top_k]
