"""
分子骨格・化学空間分析

Implements: F-SCAFFOLD-001
設計思想:
- 分子骨格（Murcko Scaffold）の多様性分析
- 化学空間カバレッジの可視化
- データセットの代表性評価
- トレーニング/テスト分割の最適化

参考文献:
- Murcko Scaffold: Bemis & Murcko, J. Med. Chem. 1996
- Scaffold Diversity: Schuffenhauer et al., J. Med. Chem. 2007
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# RDKitの遅延インポート
_RDKIT_AVAILABLE: Optional[bool] = None


def _check_rdkit() -> bool:
    global _RDKIT_AVAILABLE
    if _RDKIT_AVAILABLE is None:
        try:
            from rdkit import Chem
            _RDKIT_AVAILABLE = True
        except ImportError:
            _RDKIT_AVAILABLE = False
    return _RDKIT_AVAILABLE


@dataclass
class ScaffoldProfile:
    """骨格分析結果"""
    n_molecules: int = 0
    n_unique_scaffolds: int = 0
    scaffold_diversity: float = 0.0  # unique scaffolds / molecules
    
    # 最頻出骨格
    top_scaffolds: List[Tuple[str, int]] = field(default_factory=list)
    
    # 骨格カテゴリ分布
    scaffold_categories: Dict[str, int] = field(default_factory=dict)
    
    # 分析ノート
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'n_molecules': self.n_molecules,
            'n_unique_scaffolds': self.n_unique_scaffolds,
            'scaffold_diversity': self.scaffold_diversity,
            'top_scaffolds': self.top_scaffolds[:10],
            'scaffold_categories': self.scaffold_categories,
            'notes': self.notes,
        }


class ScaffoldAnalyzer:
    """
    分子骨格の多様性分析
    
    Murcko骨格を抽出し、データセットの構造的多様性を評価。
    
    Usage:
        analyzer = ScaffoldAnalyzer()
        profile = analyzer.analyze(smiles_list)
        print(profile.scaffold_diversity)
    """
    
    # 骨格カテゴリ（SMARTSパターン）
    SCAFFOLD_CATEGORIES = {
        'benzene': 'c1ccccc1',
        'pyridine': 'n1ccccc1',
        'pyrimidine': 'n1ccncc1',
        'naphthalene': 'c1ccc2ccccc2c1',
        'indole': 'c1ccc2[nH]ccc2c1',
        'cyclohexane': 'C1CCCCC1',
        'piperidine': 'N1CCCCC1',
        'morpholine': 'O1CCNCC1',
    }
    
    def __init__(self, include_sidechains: bool = False):
        """
        Args:
            include_sidechains: 側鎖を含めるか（Falseで純粋なMurcko骨格）
        """
        self.include_sidechains = include_sidechains
        self._smarts_cache: Dict[str, any] = {}
    
    def analyze(self, smiles_list: List[str]) -> ScaffoldProfile:
        """SMILESリストの骨格分析"""
        if not _check_rdkit():
            logger.warning("RDKitが利用不可")
            return ScaffoldProfile(n_molecules=len(smiles_list))
        
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        scaffolds = []
        categories = Counter()
        
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                
                # Murcko骨格抽出
                if self.include_sidechains:
                    scaffold = MurckoScaffold.MakeScaffoldGeneric(
                        MurckoScaffold.GetScaffoldForMol(mol)
                    )
                else:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                
                scaffold_smi = Chem.MolToSmiles(scaffold)
                scaffolds.append(scaffold_smi)
                
                # カテゴリ判定
                for cat_name, cat_smarts in self.SCAFFOLD_CATEGORIES.items():
                    if cat_smarts not in self._smarts_cache:
                        self._smarts_cache[cat_smarts] = Chem.MolFromSmarts(cat_smarts)
                    
                    pattern = self._smarts_cache[cat_smarts]
                    if pattern and mol.HasSubstructMatch(pattern):
                        categories[cat_name] += 1
                        break
                else:
                    categories['other'] += 1
                    
            except Exception as e:
                logger.debug(f"Scaffold extraction failed: {e}")
                continue
        
        # 統計計算
        scaffold_counts = Counter(scaffolds)
        n_unique = len(scaffold_counts)
        diversity = n_unique / len(scaffolds) if scaffolds else 0
        
        top_scaffolds = scaffold_counts.most_common(10)
        
        # プロファイル生成
        profile = ScaffoldProfile(
            n_molecules=len(smiles_list),
            n_unique_scaffolds=n_unique,
            scaffold_diversity=diversity,
            top_scaffolds=top_scaffolds,
            scaffold_categories=dict(categories),
        )
        
        # 分析ノート
        if diversity < 0.1:
            profile.notes.append("骨格多様性が非常に低い（<10%）→ 過学習リスク")
        elif diversity < 0.3:
            profile.notes.append("骨格多様性が低め（<30%）→ 外挿に注意")
        elif diversity > 0.7:
            profile.notes.append("骨格多様性が高い（>70%）→ 汎化性能期待")
        
        if top_scaffolds and top_scaffolds[0][1] > len(smiles_list) * 0.3:
            dominant = top_scaffolds[0][0]
            profile.notes.append(f"支配的骨格あり: {dominant[:50]}...")
        
        return profile


class ChemicalSpaceAnalyzer:
    """
    化学空間カバレッジ分析
    
    フィンガープリントベースで化学空間を可視化・評価。
    """
    
    def __init__(
        self,
        fp_type: str = 'morgan',
        fp_radius: int = 2,
        fp_bits: int = 1024,
    ):
        self.fp_type = fp_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
    
    def compute_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """SMILESリストからフィンガープリント行列を計算"""
        if not _check_rdkit():
            return np.zeros((len(smiles_list), self.fp_bits))
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        fps = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.fp_radius, nBits=self.fp_bits
                    )
                    fps.append(list(fp))
                else:
                    fps.append([0] * self.fp_bits)
            except Exception:
                fps.append([0] * self.fp_bits)
        
        return np.array(fps)
    
    def compute_pairwise_similarity(
        self, 
        smiles_list: List[str],
        sample_size: int = 500,
    ) -> Tuple[float, float]:
        """
        ペアワイズ類似度統計を計算
        
        Returns:
            (mean_similarity, std_similarity)
        """
        fps = self.compute_fingerprints(smiles_list)
        
        # サンプリング
        if len(fps) > sample_size:
            indices = np.random.choice(len(fps), sample_size, replace=False)
            fps = fps[indices]
        
        # Tanimoto類似度計算
        n = len(fps)
        similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Tanimoto
                intersection = np.sum(fps[i] & fps[j])
                union = np.sum(fps[i] | fps[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        if not similarities:
            return 0.0, 0.0
        
        return float(np.mean(similarities)), float(np.std(similarities))
    
    def analyze_coverage(
        self,
        train_smiles: List[str],
        test_smiles: List[str],
    ) -> Dict[str, float]:
        """
        トレーニングとテストの化学空間カバレッジを分析
        
        Returns:
            - train_test_similarity: 平均類似度
            - test_coverage: テストがトレーニング空間でカバーされる割合
        """
        train_fps = self.compute_fingerprints(train_smiles)
        test_fps = self.compute_fingerprints(test_smiles)
        
        # 各テスト分子の最近傍トレーニング分子との類似度
        max_similarities = []
        
        for test_fp in test_fps:
            max_sim = 0
            for train_fp in train_fps:
                intersection = np.sum(test_fp & train_fp)
                union = np.sum(test_fp | train_fp)
                if union > 0:
                    sim = intersection / union
                    max_sim = max(max_sim, sim)
            max_similarities.append(max_sim)
        
        max_similarities = np.array(max_similarities)
        
        return {
            'train_test_mean_nn_similarity': float(np.mean(max_similarities)),
            'test_coverage_50': float(np.mean(max_similarities > 0.5)),
            'test_coverage_70': float(np.mean(max_similarities > 0.7)),
            'test_outliers': float(np.mean(max_similarities < 0.3)),
        }


class ScaffoldSplitter:
    """
    骨格ベースのデータ分割
    
    同じ骨格の分子が同じ分割に入るようにすることで、
    より厳格な汎化性能評価が可能。
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split(
        self, 
        smiles_list: List[str],
    ) -> Tuple[List[int], List[int]]:
        """
        骨格ベースでトレーニング/テストインデックスを返す
        
        Returns:
            (train_indices, test_indices)
        """
        if not _check_rdkit():
            # フォールバック: ランダム分割
            n = len(smiles_list)
            indices = np.random.RandomState(self.random_state).permutation(n)
            split_idx = int(n * (1 - self.test_size))
            return list(indices[:split_idx]), list(indices[split_idx:])
        
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # 骨格→インデックスのマッピング
        scaffold_to_indices: Dict[str, List[int]] = {}
        
        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smi = Chem.MolToSmiles(scaffold)
                else:
                    scaffold_smi = ""
            except Exception:
                scaffold_smi = ""
            
            if scaffold_smi not in scaffold_to_indices:
                scaffold_to_indices[scaffold_smi] = []
            scaffold_to_indices[scaffold_smi].append(i)
        
        # 骨格をシャッフルして分割
        np.random.seed(self.random_state)
        scaffolds = list(scaffold_to_indices.keys())
        np.random.shuffle(scaffolds)
        
        n_total = len(smiles_list)
        n_test_target = int(n_total * self.test_size)
        
        train_indices = []
        test_indices = []
        
        for scaffold in scaffolds:
            indices = scaffold_to_indices[scaffold]
            if len(test_indices) < n_test_target:
                test_indices.extend(indices)
            else:
                train_indices.extend(indices)
        
        return train_indices, test_indices


def analyze_scaffolds(smiles_list: List[str]) -> ScaffoldProfile:
    """便利関数: 骨格分析を実行"""
    return ScaffoldAnalyzer().analyze(smiles_list)


def scaffold_split(
    smiles_list: List[str],
    test_size: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """便利関数: 骨格ベース分割"""
    return ScaffoldSplitter(test_size=test_size).split(smiles_list)
