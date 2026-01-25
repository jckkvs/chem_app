"""
等変グラフニューラルネットワーク (Equivariant GNN) 埋め込み

Implements: F-EGNN-001
設計思想:
- 3D構造を考慮した分子表現学習
- E(3)等変性（回転・並進・反転に対して不変/共変）
- 量子化学特性、力場、テンソル物性の予測に必須

サポートモデル:
- SchNet: 連続フィルター畳み込み
- PaiNN: 偏極可能原子相互作用ネットワーク
- DimeNet: 方向性メッセージパッシング

参考文献:
- SchNet: Schütt et al., J. Chem. Phys. 2018
- PaiNN: Schütt et al., ICML 2021
- DimeNet: Klicpera et al., ICLR 2020

依存関係（オプショナル）:
- schnetpack>=2.0
- pytorch_geometric>=2.0
- ase>=3.22
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default weights directory
DEFAULT_WEIGHTS_DIR = Path.home() / ".chem_ml" / "weights"

# オプショナル依存の遅延インポート
_SCHNETPACK_AVAILABLE: Optional[bool] = None
_PYG_AVAILABLE: Optional[bool] = None
_ASE_AVAILABLE: Optional[bool] = None


def _check_schnetpack() -> bool:
    global _SCHNETPACK_AVAILABLE
    if _SCHNETPACK_AVAILABLE is None:
        try:
            import schnetpack
            _SCHNETPACK_AVAILABLE = True
            logger.info(f"schnetpack v{schnetpack.__version__} available")
        except ImportError:
            _SCHNETPACK_AVAILABLE = False
    return _SCHNETPACK_AVAILABLE


def _check_pyg() -> bool:
    global _PYG_AVAILABLE
    if _PYG_AVAILABLE is None:
        try:
            import torch_geometric
            _PYG_AVAILABLE = True
            logger.info(f"torch_geometric v{torch_geometric.__version__} available")
        except ImportError:
            _PYG_AVAILABLE = False
    return _PYG_AVAILABLE


def _check_ase() -> bool:
    global _ASE_AVAILABLE
    if _ASE_AVAILABLE is None:
        try:
            import ase
            _ASE_AVAILABLE = True
        except ImportError:
            _ASE_AVAILABLE = False
    return _ASE_AVAILABLE


@dataclass
class MolecularStructure:
    """3D分子構造データ"""
    smiles: str
    atomic_numbers: np.ndarray  # (n_atoms,)
    positions: np.ndarray       # (n_atoms, 3)
    
    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)
    
    @classmethod
    def from_smiles(cls, smiles: str, optimize: bool = True) -> Optional['MolecularStructure']:
        """SMILESから3D構造を生成（RDKit使用）"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            mol = Chem.AddHs(mol)
            
            # 3D座標生成
            if optimize:
                # ETKDG法で埋め込み
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                if result == -1:
                    # 失敗した場合は別の方法
                    result = AllChem.EmbedMolecule(mol, randomSeed=42)
                
                if result != -1:
                    # 力場最適化
                    try:
                        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    except Exception:
                        pass
            else:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            
            return cls(
                smiles=smiles,
                atomic_numbers=atomic_numbers,
                positions=positions,
            )
            
        except Exception as e:
            logger.debug(f"Failed to generate 3D structure: {e}")
            return None


class BaseEquivariantEmbedding(ABC):
    """等変GNN埋め込みの基底クラス"""
    
    @abstractmethod
    def get_embeddings(
        self, 
        structures: List[MolecularStructure]
    ) -> np.ndarray:
        """分子構造から埋め込みベクトルを取得"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """モデルが利用可能か"""
        pass


class SchNetEmbedding(BaseEquivariantEmbedding):
    """
    SchNet による3D分子埋め込み
    
    連続フィルター畳み込みを使用し、原子間距離に基づく
    回転・並進不変な表現を学習。
    
    Reference: Schütt et al., "SchNet: A continuous-filter 
    convolutional neural network for modeling quantum interactions"
    
    Usage:
        embedder = SchNetEmbedding()
        embeddings = embedder.get_embeddings(structures)
    """
    
    def __init__(
        self,
        model_name: str = "qm9_energy",
        cutoff: float = 5.0,
        n_atom_basis: int = 128,
        device: str = "cpu",
        weights_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: 事前学習モデル名 or パス
            cutoff: カットオフ距離 (Å)
            n_atom_basis: 原子埋め込み次元
            device: 計算デバイス
        """
        self.model_name = model_name
        self.cutoff = cutoff
        self.cutoff = cutoff
        self.n_atom_basis = n_atom_basis
        self.device = device
        self.weights_path = weights_path
        
        # Resolve default weights path for custom models
        if self.weights_path is None:
            default_path = DEFAULT_WEIGHTS_DIR / f"schnet_{model_name}.pt"
            if default_path.exists():
                self.weights_path = str(default_path)
        
        self._model = None
        self._loaded = False
    
    def is_available(self) -> bool:
        return _check_schnetpack()
    
    def _load_model(self):
        """モデルをロード"""
        if self._loaded:
            return
        
        if not self.is_available():
            raise ImportError("schnetpack is not installed")
        
        import schnetpack as spk
        import torch

        # 事前学習モデルのロード
        # schnetpackのモデルハブから取得、またはカスタムモデル
        try:
            # QM9事前学習モデル
            self._model = spk.representation.SchNet(
                n_atom_basis=self.n_atom_basis,
                n_filters=self.n_atom_basis,
                n_interactions=6,
                cutoff=self.cutoff,
            )
            self._model.to(self.device)
            
            # Load weights if available
            if self.weights_path and Path(self.weights_path).exists():
                try:
                    state_dict = torch.load(self.weights_path, map_location=self.device)
                    self._model.load_state_dict(state_dict)
                    logger.info(f"Loaded SchNet weights from {self.weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load SchNet weights: {e}")
            
            self._model.eval()
            self._loaded = True
            logger.info("SchNet model initialized")
        except Exception as e:
            logger.error(f"Failed to load SchNet model: {e}")
            raise
    
    def _structure_to_input(
        self, 
        structure: MolecularStructure
    ) -> Dict[str, Any]:
        """構造をSchNet入力形式に変換"""
        import torch
        
        return {
            'positions': torch.tensor(
                structure.positions, 
                dtype=torch.float32, 
                device=self.device
            ),
            'atomic_numbers': torch.tensor(
                structure.atomic_numbers, 
                dtype=torch.long, 
                device=self.device
            ),
            'n_atoms': torch.tensor([structure.n_atoms], device=self.device),
        }
    
    def get_embeddings(
        self, 
        structures: List[MolecularStructure]
    ) -> np.ndarray:
        """分子構造から埋め込みを取得"""
        self._load_model()
        
        import torch
        
        embeddings = []
        
        with torch.no_grad():
            for struct in structures:
                try:
                    inputs = self._structure_to_input(struct)
                    
                    # SchNetは原子ごとの表現を出力
                    # プーリングで分子レベルの埋め込みに変換
                    atom_embeddings = self._model(inputs)
                    
                    # 平均プーリング
                    mol_embedding = atom_embeddings.mean(dim=0).cpu().numpy()
                    embeddings.append(mol_embedding)
                    
                except Exception as e:
                    logger.warning(f"Embedding failed for structure: {e}")
                    embeddings.append(np.zeros(self.n_atom_basis))
        
        return np.array(embeddings)


class PaiNNEmbedding(BaseEquivariantEmbedding):
    """
    PaiNN (Polarizable Atom Interaction Neural Network) 埋め込み
    
    等変メッセージパッシングで、スカラーとベクトル特徴を同時に学習。
    テンソル物性（分極率等）の予測に優れる。
    
    Reference: Schütt et al., "Equivariant message passing for 
    the prediction of tensorial properties and molecular spectra"
    """
    
    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 3,
        cutoff: float = 5.0,
        device: str = "cpu",
        weights_path: Optional[str] = None,
    ):
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff = cutoff
        self.device = device
        self.weights_path = weights_path
        
        if self.weights_path is None:
            default_path = DEFAULT_WEIGHTS_DIR / "painn.pt"
            if default_path.exists():
                self.weights_path = str(default_path)
        
        self._model = None
        self._loaded = False
    
    def is_available(self) -> bool:
        return _check_schnetpack()
    
    def _load_model(self):
        if self._loaded:
            return
        
        if not self.is_available():
            raise ImportError("schnetpack is not installed")
        
        import schnetpack as spk
        import torch
        
        try:
            self._model = spk.representation.PaiNN(
                n_atom_basis=self.n_atom_basis,
                n_interactions=self.n_interactions,
                radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=self.cutoff),
                cutoff_fn=spk.nn.CosineCutoff(self.cutoff),
            )
            self._model.to(self.device)
            
            # Load weights if available
            if self.weights_path and Path(self.weights_path).exists():
                try:
                    state_dict = torch.load(self.weights_path, map_location=self.device)
                    self._model.load_state_dict(state_dict)
                    logger.info(f"Loaded PaiNN weights from {self.weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load PaiNN weights: {e}")
                    
            self._model.eval()
            self._loaded = True
            logger.info("PaiNN model initialized")
        except Exception as e:
            logger.error(f"Failed to load PaiNN model: {e}")
            raise
    
    def get_embeddings(
        self, 
        structures: List[MolecularStructure]
    ) -> np.ndarray:
        """PaiNN埋め込みを取得"""
        self._load_model()
        
        import torch
        
        embeddings = []
        
        with torch.no_grad():
            for struct in structures:
                try:
                    # PaiNN入力形式に変換
                    inputs = {
                        'positions': torch.tensor(
                            struct.positions, 
                            dtype=torch.float32,
                            device=self.device
                        ).unsqueeze(0),
                        'atomic_numbers': torch.tensor(
                            struct.atomic_numbers,
                            dtype=torch.long,
                            device=self.device
                        ).unsqueeze(0),
                    }
                    
                    # 埋め込み取得
                    result = self._model(inputs)
                    
                    # スカラー特徴のプーリング
                    mol_embedding = result['scalar_representation'].mean(dim=1).squeeze().cpu().numpy()
                    embeddings.append(mol_embedding)
                    
                except Exception as e:
                    logger.warning(f"PaiNN embedding failed: {e}")
                    embeddings.append(np.zeros(self.n_atom_basis))
        
        return np.array(embeddings)


class EquivariantEmbeddingEngine:
    """
    等変GNN埋め込みの統合エンジン
    
    複数のモデルを統一インターフェースで使用可能。
    SMILESから3D構造生成→埋め込み取得のパイプライン。
    
    Usage:
        engine = EquivariantEmbeddingEngine()
        df = engine.get_embeddings_df(smiles_list, model='schnet')
    """
    
    AVAILABLE_MODELS = {
        'schnet': SchNetEmbedding,
        'painn': PaiNNEmbedding,
    }
    
    def __init__(self, optimize_3d: bool = True):
        """
        Args:
            optimize_3d: 3D構造を力場最適化するか
        """
        self.optimize_3d = optimize_3d
        self._models: Dict[str, BaseEquivariantEmbedding] = {}
    
    def _get_model(self, model_name: str) -> BaseEquivariantEmbedding:
        """モデルを取得（遅延初期化）"""
        if model_name not in self._models:
            if model_name not in self.AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_class = self.AVAILABLE_MODELS[model_name]
            self._models[model_name] = model_class()
        
        return self._models[model_name]
    
    def is_model_available(self, model_name: str) -> bool:
        """モデルが利用可能か"""
        if model_name not in self.AVAILABLE_MODELS:
            return False
        
        model = self._get_model(model_name)
        return model.is_available()
    
    def smiles_to_structures(
        self, 
        smiles_list: List[str]
    ) -> List[Optional[MolecularStructure]]:
        """SMILESリストから3D構造を生成"""
        structures = []
        
        for smi in smiles_list:
            struct = MolecularStructure.from_smiles(smi, optimize=self.optimize_3d)
            structures.append(struct)
        
        return structures
    
    def get_embeddings(
        self,
        smiles_list: List[str],
        model_name: str = 'schnet',
    ) -> np.ndarray:
        """SMILESリストから埋め込みを取得"""
        model = self._get_model(model_name)
        
        if not model.is_available():
            logger.warning(f"Model {model_name} is not available, returning zeros")
            return np.zeros((len(smiles_list), 128))
        
        # 3D構造生成
        structures = self.smiles_to_structures(smiles_list)
        
        # 有効な構造のみ処理
        valid_structures = []
        valid_indices = []
        
        for i, struct in enumerate(structures):
            if struct is not None:
                valid_structures.append(struct)
                valid_indices.append(i)
        
        if not valid_structures:
            logger.warning("No valid structures generated")
            return np.zeros((len(smiles_list), 128))
        
        # 埋め込み取得
        valid_embeddings = model.get_embeddings(valid_structures)
        
        # 結果を元のインデックスに配置
        n_dim = valid_embeddings.shape[1]
        all_embeddings = np.zeros((len(smiles_list), n_dim))
        
        for i, idx in enumerate(valid_indices):
            all_embeddings[idx] = valid_embeddings[i]
        
        return all_embeddings
    
    def get_embeddings_df(
        self,
        smiles_list: List[str],
        model_name: str = 'schnet',
    ) -> pd.DataFrame:
        """埋め込みをDataFrameとして取得"""
        embeddings = self.get_embeddings(smiles_list, model_name)
        
        columns = [f"{model_name}_dim_{i}" for i in range(embeddings.shape[1])]
        return pd.DataFrame(embeddings, columns=columns)
    
    @staticmethod
    def list_available_models() -> Dict[str, bool]:
        """利用可能なモデルの一覧"""
        return {
            'schnet': _check_schnetpack(),
            'painn': _check_schnetpack(),
        }


def get_equivariant_embeddings(
    smiles_list: List[str],
    model: str = 'schnet',
) -> pd.DataFrame:
    """
    便利関数: 等変GNN埋め込みを取得
    
    Args:
        smiles_list: SMILESリスト
        model: モデル名 ('schnet', 'painn')
        
    Returns:
        pd.DataFrame: 埋め込みベクトル
    """
    engine = EquivariantEmbeddingEngine()
    return engine.get_embeddings_df(smiles_list, model_name=model)
