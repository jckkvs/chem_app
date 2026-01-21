"""
分子グラフ特徴量（SchNet/DimeNet inspired）

Implements: F-GRAPH-001
設計思想:
- 分子グラフ構造
- ノード/エッジ特徴量
- グラフ畳み込み準備
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MolecularGraph:
    """分子グラフ"""
    smiles: str
    node_features: np.ndarray  # (n_atoms, n_node_features)
    edge_index: np.ndarray  # (2, n_edges)
    edge_features: np.ndarray  # (n_edges, n_edge_features)
    n_atoms: int
    n_bonds: int
    global_features: Dict[str, float] = field(default_factory=dict)


class MolecularGraphBuilder:
    """
    分子グラフ構築（SchNet/DimeNet inspired）
    
    Features:
    - 原子特徴量（元素、電荷、混成など）
    - 結合特徴量（結合次数、共役など）
    - 3D座標（オプション）
    
    Example:
        >>> builder = MolecularGraphBuilder()
        >>> graph = builder.build("CCO")
    """
    
    # 元素の特徴量次元
    ATOM_FEATURES = ['atomic_num', 'formal_charge', 'hybridization', 'aromatic', 'ring']
    BOND_FEATURES = ['bond_type', 'conjugated', 'ring']
    
    def build(self, smiles: str, include_3d: bool = False) -> Optional[MolecularGraph]:
        """分子グラフを構築"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if include_3d:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            
            # ノード特徴量
            node_features = self._get_node_features(mol)
            
            # エッジ（結合）
            edge_index, edge_features = self._get_edge_features(mol)
            
            return MolecularGraph(
                smiles=smiles,
                node_features=node_features,
                edge_index=edge_index,
                edge_features=edge_features,
                n_atoms=mol.GetNumAtoms(),
                n_bonds=mol.GetNumBonds(),
            )
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            return None
    
    def _get_node_features(self, mol) -> np.ndarray:
        """ノード（原子）特徴量"""
        from rdkit import Chem
        
        features = []
        
        for atom in mol.GetAtoms():
            atom_feat = [
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                atom.GetTotalNumHs(),
                atom.GetDegree(),
            ]
            features.append(atom_feat)
        
        return np.array(features, dtype=np.float32)
    
    def _get_edge_features(self, mol) -> Tuple[np.ndarray, np.ndarray]:
        """エッジ（結合）特徴量"""
        from rdkit import Chem
        
        edge_index = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 無向グラフ（双方向）
            edge_index.append([i, j])
            edge_index.append([j, i])
            
            bond_feat = [
                float(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                float(bond.GetBondTypeAsDouble()),
            ]
            
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)  # 双方向
        
        if edge_index:
            return (
                np.array(edge_index, dtype=np.int64).T,
                np.array(edge_features, dtype=np.float32),
            )
        else:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    
    def batch_build(self, smiles_list: List[str]) -> List[MolecularGraph]:
        """バッチ構築"""
        return [self.build(smi) for smi in smiles_list if self.build(smi) is not None]
    
    def to_adjacency_matrix(self, graph: MolecularGraph) -> np.ndarray:
        """隣接行列に変換"""
        adj = np.zeros((graph.n_atoms, graph.n_atoms))
        
        for k in range(graph.edge_index.shape[1]):
            i, j = graph.edge_index[:, k]
            adj[i, j] = 1
        
        return adj
