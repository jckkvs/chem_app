"""
データセット分析器 - 入力分子の特性を自動分析し最適記述子を推奨

Implements: F-ANALYZER-001
設計思想:
- データセットの分子構造特性を事前分析
- 支配的な骨格・官能基を特定
- 分子特性に基づく記述子カテゴリ推奨
- 継続的に改善されるべきヒューリスティクス

参考文献:
- Molecular Scaffolds in Drug Discovery (Hu et al., 2016)
- Chemical Space Analysis (Lipkus et al., 2008)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# RDKitの遅延インポート
_RDKIT_AVAILABLE: Optional[bool] = None


def _check_rdkit() -> bool:
    """RDKitが利用可能か"""
    global _RDKIT_AVAILABLE
    if _RDKIT_AVAILABLE is None:
        try:
            from rdkit import Chem
            _RDKIT_AVAILABLE = True
        except ImportError:
            _RDKIT_AVAILABLE = False
    return _RDKIT_AVAILABLE


@dataclass
class FunctionalGroupProfile:
    """官能基分布プロファイル"""
    aromatic_ring_ratio: float = 0.0
    aliphatic_ring_ratio: float = 0.0
    heteroatom_ratio: float = 0.0
    amine_ratio: float = 0.0
    carboxylic_acid_ratio: float = 0.0
    ester_ratio: float = 0.0
    amide_ratio: float = 0.0
    hydroxyl_ratio: float = 0.0
    carbonyl_ratio: float = 0.0
    halogen_ratio: float = 0.0
    ether_ratio: float = 0.0
    sulfur_containing_ratio: float = 0.0
    
    def dominant_groups(self, threshold: float = 0.3) -> List[str]:
        """閾値以上の割合を持つ官能基"""
        dominant = []
        for name, value in [
            ('aromatic', self.aromatic_ring_ratio),
            ('aliphatic_ring', self.aliphatic_ring_ratio),
            ('amine', self.amine_ratio),
            ('carboxylic_acid', self.carboxylic_acid_ratio),
            ('ester', self.ester_ratio),
            ('amide', self.amide_ratio),
            ('hydroxyl', self.hydroxyl_ratio),
            ('carbonyl', self.carbonyl_ratio),
            ('halogen', self.halogen_ratio),
            ('ether', self.ether_ratio),
            ('sulfur', self.sulfur_containing_ratio),
        ]:
            if value >= threshold:
                dominant.append(name)
        return dominant


@dataclass
class MolecularProfile:
    """分子特性プロファイル"""
    mw_mean: float = 0.0
    mw_std: float = 0.0
    mw_min: float = 0.0
    mw_max: float = 0.0
    
    heavy_atom_mean: float = 0.0
    rotatable_bond_mean: float = 0.0
    ring_count_mean: float = 0.0
    aromatic_ring_mean: float = 0.0
    
    fraction_csp3_mean: float = 0.0
    
    @property
    def is_small_molecule(self) -> bool:
        """低分子か（MW < 500）"""
        return self.mw_mean < 500
    
    @property
    def is_polymer_like(self) -> bool:
        """ポリマー様か（MW > 1000 または繰り返し単位）"""
        return self.mw_mean > 1000
    
    @property
    def is_rigid(self) -> bool:
        """剛直分子が多いか"""
        # 回転可能結合が少なく、芳香環が多い
        return self.rotatable_bond_mean < 3 and self.aromatic_ring_mean > 1
    
    @property
    def is_flexible(self) -> bool:
        """柔軟分子が多いか"""
        return self.rotatable_bond_mean > 6 and self.fraction_csp3_mean > 0.5


@dataclass 
class DatasetProfile:
    """データセット全体のプロファイル"""
    n_molecules: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    
    molecular: MolecularProfile = field(default_factory=MolecularProfile)
    functional_groups: FunctionalGroupProfile = field(default_factory=FunctionalGroupProfile)
    
    # 推奨
    recommended_preset: str = "general"
    recommended_descriptor_categories: List[str] = field(default_factory=list)
    recommended_pretrained_models: List[str] = field(default_factory=list)
    
    analysis_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'n_molecules': self.n_molecules,
            'n_valid': self.n_valid,
            'n_invalid': self.n_invalid,
            'molecular': {
                'mw_mean': self.molecular.mw_mean,
                'mw_std': self.molecular.mw_std,
                'heavy_atom_mean': self.molecular.heavy_atom_mean,
                'rotatable_bond_mean': self.molecular.rotatable_bond_mean,
                'aromatic_ring_mean': self.molecular.aromatic_ring_mean,
                'fraction_csp3_mean': self.molecular.fraction_csp3_mean,
                'is_rigid': self.molecular.is_rigid,
                'is_flexible': self.molecular.is_flexible,
            },
            'functional_groups': {
                'dominant': self.functional_groups.dominant_groups(),
            },
            'recommendations': {
                'preset': self.recommended_preset,
                'descriptor_categories': self.recommended_descriptor_categories,
                'pretrained_models': self.recommended_pretrained_models,
            },
            'notes': self.analysis_notes,
        }


class DatasetAnalyzer:
    """
    データセットの分子構造特性を分析し、最適な記述子を推奨
    
    Usage:
        analyzer = DatasetAnalyzer()
        profile = analyzer.analyze(smiles_list)
        print(profile.recommended_preset)
        print(profile.analysis_notes)
    """
    
    # SMARTSパターンによる官能基検出
    FUNCTIONAL_GROUP_SMARTS = {
        'aromatic': '[a]',
        'amine_primary': '[NX3;H2;!$(NC=O)]',
        'amine_secondary': '[NX3;H1;!$(NC=O)]',
        'amine_tertiary': '[NX3;H0;!$(NC=O)]',
        'carboxylic_acid': '[CX3](=O)[OX2H1]',
        'ester': '[CX3](=O)[OX2][#6]',
        'amide': '[NX3][CX3](=[OX1])[#6]',
        'hydroxyl': '[OX2H]',
        'carbonyl': '[CX3]=[OX1]',
        'halogen': '[F,Cl,Br,I]',
        'ether': '[OX2]([#6])[#6]',
        'sulfur': '[#16]',
        'nitro': '[N+](=O)[O-]',
        'nitrile': '[CX2]#[NX1]',
    }
    
    def __init__(self, sample_size: int = 500):
        """
        Args:
            sample_size: 大規模データセットの場合のサンプリング数
        """
        self.sample_size = sample_size
        self._smarts_cache: Dict[str, any] = {}
    
    def analyze(self, smiles_list: List[str]) -> DatasetProfile:
        """
        SMILESリストを分析してプロファイルを生成
        
        Args:
            smiles_list: SMILESの文字列リスト
            
        Returns:
            DatasetProfile: 分析結果
        """
        if not _check_rdkit():
            logger.warning("RDKitが利用不可、デフォルトプロファイルを返します")
            profile = DatasetProfile(n_molecules=len(smiles_list))
            profile.analysis_notes.append("RDKitが利用不可のため分析スキップ")
            return profile
        
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski

        # サンプリング
        if len(smiles_list) > self.sample_size:
            indices = np.random.choice(len(smiles_list), self.sample_size, replace=False)
            sampled = [smiles_list[i] for i in indices]
            logger.info(f"大規模データセット: {len(smiles_list)} → {self.sample_size}サンプル")
        else:
            sampled = smiles_list
        
        # 分子オブジェクト生成
        mols = []
        invalid = 0
        for smi in sampled:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    mols.append(mol)
                else:
                    invalid += 1
            except Exception:
                invalid += 1
        
        profile = DatasetProfile(
            n_molecules=len(smiles_list),
            n_valid=len(mols),
            n_invalid=invalid,
        )
        
        if not mols:
            profile.analysis_notes.append("有効な分子がありません")
            return profile
        
        # 分子特性を計算
        profile.molecular = self._compute_molecular_profile(mols)
        
        # 官能基プロファイル
        profile.functional_groups = self._compute_functional_groups(mols)
        
        # 推奨を生成
        self._generate_recommendations(profile)
        
        return profile
    
    def _compute_molecular_profile(self, mols: List) -> MolecularProfile:
        """分子特性を計算"""
        from rdkit.Chem import Descriptors, Lipinski
        
        mws = []
        heavy_atoms = []
        rotatable_bonds = []
        ring_counts = []
        aromatic_rings = []
        fraction_csp3s = []
        
        for mol in mols:
            try:
                mws.append(Descriptors.MolWt(mol))
                heavy_atoms.append(Descriptors.HeavyAtomCount(mol))
                rotatable_bonds.append(Descriptors.NumRotatableBonds(mol))
                ring_counts.append(Descriptors.RingCount(mol))
                aromatic_rings.append(Descriptors.NumAromaticRings(mol))
                fraction_csp3s.append(Descriptors.FractionCSP3(mol))
            except Exception:
                continue
        
        return MolecularProfile(
            mw_mean=np.mean(mws) if mws else 0,
            mw_std=np.std(mws) if mws else 0,
            mw_min=np.min(mws) if mws else 0,
            mw_max=np.max(mws) if mws else 0,
            heavy_atom_mean=np.mean(heavy_atoms) if heavy_atoms else 0,
            rotatable_bond_mean=np.mean(rotatable_bonds) if rotatable_bonds else 0,
            ring_count_mean=np.mean(ring_counts) if ring_counts else 0,
            aromatic_ring_mean=np.mean(aromatic_rings) if aromatic_rings else 0,
            fraction_csp3_mean=np.mean(fraction_csp3s) if fraction_csp3s else 0,
        )
    
    def _compute_functional_groups(self, mols: List) -> FunctionalGroupProfile:
        """官能基分布を計算"""
        from rdkit import Chem
        
        n_mols = len(mols)
        counts = {key: 0 for key in self.FUNCTIONAL_GROUP_SMARTS}
        
        # SMARTSパターンをコンパイル（キャッシュ）
        for name, smarts in self.FUNCTIONAL_GROUP_SMARTS.items():
            if name not in self._smarts_cache:
                self._smarts_cache[name] = Chem.MolFromSmarts(smarts)
        
        for mol in mols:
            for name, pattern in self._smarts_cache.items():
                if pattern and mol.HasSubstructMatch(pattern):
                    counts[name] += 1
        
        # 割合に変換
        def ratio(key):
            return counts.get(key, 0) / n_mols if n_mols > 0 else 0
        
        return FunctionalGroupProfile(
            aromatic_ring_ratio=ratio('aromatic'),
            amine_ratio=(ratio('amine_primary') + ratio('amine_secondary') + ratio('amine_tertiary')),
            carboxylic_acid_ratio=ratio('carboxylic_acid'),
            ester_ratio=ratio('ester'),
            amide_ratio=ratio('amide'),
            hydroxyl_ratio=ratio('hydroxyl'),
            carbonyl_ratio=ratio('carbonyl'),
            halogen_ratio=ratio('halogen'),
            ether_ratio=ratio('ether'),
            sulfur_containing_ratio=ratio('sulfur'),
        )
    
    def _generate_recommendations(self, profile: DatasetProfile) -> None:
        """分析結果から推奨を生成"""
        notes = profile.analysis_notes
        rec_categories = []
        rec_models = []
        
        mol = profile.molecular
        fg = profile.functional_groups
        
        # 分子サイズに基づく推奨
        if mol.is_polymer_like:
            notes.append("高分子量サンプルが多い → PolyBERT推奨")
            rec_models.append('polybert')
            rec_categories.extend(['glass_transition', 'tensile_strength', 'gas_permeability'])
        elif mol.is_small_molecule:
            notes.append("低分子が中心 → 標準記述子で十分")
        
        # 剛直性に基づく推奨
        if mol.is_rigid:
            notes.append("剛直分子が多い → 芳香族・形状記述子推奨")
            rec_categories.extend(['aromatic_descriptors', 'shape_descriptors'])
            profile.recommended_preset = 'elastic_modulus'
        elif mol.is_flexible:
            notes.append("柔軟分子が多い → 回転可能結合・トポロジー記述子推奨")
            rec_categories.extend(['topology_descriptors'])
        
        # 官能基に基づく推奨
        dominant = fg.dominant_groups(threshold=0.3)
        
        if 'aromatic' in dominant:
            notes.append("芳香環が支配的 → π電子系記述子推奨")
            rec_categories.append('electronic_descriptors')
            if mol.aromatic_ring_mean > 2:
                notes.append("共役系が長い可能性 → XTB HOMO-LUMO推奨")
                rec_categories.append('xtb_electronic')
        
        if 'amine' in dominant:
            notes.append("アミン系が多い → 塩基性・pKa関連記述子推奨")
            rec_categories.append('basicity_descriptors')
            rec_models.append('unipka')
        
        if 'carboxylic_acid' in dominant:
            notes.append("カルボン酸が多い → 酸性・pKa関連記述子推奨")
            rec_categories.append('acidity_descriptors')
            rec_models.append('unipka')
        
        if 'hydroxyl' in dominant:
            notes.append("水酸基が多い → 水素結合記述子推奨")
            rec_categories.append('hydrogen_bond_descriptors')
        
        if 'halogen' in dominant:
            notes.append("ハロゲンが多い → 極性・電子求引性記述子推奨")
        
        # デフォルト
        if not rec_categories:
            rec_categories = ['general']
        
        rec_models.append('unimol')  # 常に推奨
        
        profile.recommended_descriptor_categories = list(set(rec_categories))
        profile.recommended_pretrained_models = list(set(rec_models))
        
        if not profile.recommended_preset or profile.recommended_preset == 'general':
            # カテゴリからプリセットを推測
            if 'glass_transition' in rec_categories:
                profile.recommended_preset = 'glass_transition'
            elif 'electronic_descriptors' in rec_categories:
                profile.recommended_preset = 'optical_gap'


def analyze_dataset(smiles_list: List[str]) -> DatasetProfile:
    """便利関数: データセットを分析"""
    return DatasetAnalyzer().analyze(smiles_list)
