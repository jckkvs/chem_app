"""
フィンガープリント計算エンジン

Implements: F-FP-001
設計思想:
- 複数FPタイプ対応
- 類似性計算
- ベクトル化出力
"""

from __future__ import annotations

import logging
from typing import List, Optional, Literal, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FingerprintCalculator:
    """
    分子フィンガープリント計算機
    
    Features:
    - Morgan (ECFP), RDKit, AtomPair, MACCS
    - Tanimoto類似度
    - NumPy配列出力
    
    Example:
        >>> calc = FingerprintCalculator(fp_type='morgan')
        >>> fps = calc.calculate_batch(["CCO", "c1ccccc1"])
    """
    
    def __init__(
        self,
        fp_type: Literal['morgan', 'rdkit', 'maccs', 'atompair', 'topological'] = 'morgan',
        radius: int = 2,
        n_bits: int = 2048,
    ):
        """
        Args:
            fp_type: フィンガープリントタイプ
            radius: Morgan FP半径
            n_bits: ビット数
        """
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
    
    def calculate(self, smiles: str) -> Optional[np.ndarray]:
        """単一SMILESのFPを計算"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
            from rdkit import DataStructs
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if self.fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.n_bits
                )
            elif self.fp_type == 'rdkit':
                fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            elif self.fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif self.fp_type == 'atompair':
                fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
                    mol, nBits=self.n_bits
                )
            elif self.fp_type == 'topological':
                fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                    mol, nBits=self.n_bits
                )
            else:
                return None
            
            # NumPy配列に変換
            arr = np.zeros((len(fp),), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
            
        except Exception as e:
            logger.warning(f"FP calculation failed: {e}")
            return None
    
    def calculate_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """バッチ計算"""
        fps = [self.calculate(smi) for smi in smiles_list]
        
        # 有効なFPの長さを取得
        valid_fps = [fp for fp in fps if fp is not None]
        if not valid_fps:
            return pd.DataFrame()
        
        n_bits = len(valid_fps[0])
        
        # 無効なものはゼロベクトル
        result = []
        for fp in fps:
            if fp is not None:
                result.append(fp)
            else:
                result.append(np.zeros(n_bits, dtype=np.int8))
        
        columns = [f'FP_{i}' for i in range(n_bits)]
        return pd.DataFrame(result, columns=columns)
    
    def tanimoto_similarity(self, smiles1: str, smiles2: str) -> float:
        """Tanimoto類似度"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
            
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, self.radius, nBits=self.n_bits)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, self.radius, nBits=self.n_bits)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
            
        except Exception:
            return 0.0
    
    def similarity_matrix(self, smiles_list: List[str]) -> np.ndarray:
        """類似度行列を計算"""
        n = len(smiles_list)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    sim = self.tanimoto_similarity(smiles_list[i], smiles_list[j])
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        
        return matrix
    
    def find_similar(
        self,
        query_smiles: str,
        database_smiles: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """類似分子検索"""
        similarities = [
            (smi, self.tanimoto_similarity(query_smiles, smi))
            for smi in database_smiles
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
