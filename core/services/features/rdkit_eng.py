"""
RDKit記述子抽出エンジン - 最適記述子選択対応版

Implements: F-002
設計思想:
- 全記述子投入ではなく、物性タイプに応じた選択的抽出
- DescriptorSelectorとの統合
- エラー時の適切なログとフォールバック

参考文献:
- RDKit: Open-Source Cheminformatics Software
- Molecular Descriptors for Chemoinformatics (Todeschini & Consonni, 2009)
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Set, Dict, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# RDKit警告を抑制（ログに記録）
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')


# 記述子カテゴリ別の定義
DESCRIPTOR_CATEGORIES: Dict[str, List[str]] = {
    'constitutional': [
        'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
        'NumRadicalElectrons', 'HeavyAtomCount', 'NumHeteroatoms',
    ],
    'topological': [
        'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'Kappa1', 'Kappa2', 'Kappa3', 'HallKierAlpha',
    ],
    'electronic': [
        'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 
        'MinAbsPartialCharge',
    ],
    'lipophilicity': [
        'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
        'NumHDonors', 'NumHAcceptors',
    ],
    'structural': [
        'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
        'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumAromaticCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedCarbocycles',
        'NumAliphaticHeterocycles', 'NumAliphaticCarbocycles',
        'RingCount', 'FractionCSP3', 'NumSpiroAtoms', 'NumBridgeheadAtoms',
    ],
    'druglikeness': [
        'qed',  # Quantitative Estimate of Drug-likeness
        'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'MolLogP',
    ],
}


def get_all_rdkit_descriptors() -> List[str]:
    """利用可能な全RDKit記述子名を取得"""
    return [name for name, _ in Descriptors._descList]


def get_descriptors_by_category(category: str) -> List[str]:
    """カテゴリ別の記述子リストを取得"""
    return DESCRIPTOR_CATEGORIES.get(category, [])


class RDKitFeatureExtractor(BaseFeatureExtractor):
    """
    RDKit分子記述子抽出器 - 選択的記述子抽出対応
    
    Features:
    - カテゴリ別記述子選択
    - 特定記述子リストの指定
    - 全記述子モード（推奨しない）
    - エラー時の適切なハンドリング
    
    Example (推奨 - カテゴリ指定):
        >>> extractor = RDKitFeatureExtractor(categories=['lipophilicity', 'structural'])
        >>> features = extractor.transform(smiles_list)
    
    Example (特定記述子):
        >>> extractor = RDKitFeatureExtractor(descriptors=['MolWt', 'MolLogP', 'TPSA'])
        >>> features = extractor.transform(smiles_list)
    """
    
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        descriptors: Optional[List[str]] = None,
        use_all: bool = False,
        include_smiles: bool = True,
        error_value: float = np.nan,
        **kwargs
    ):
        """
        Args:
            categories: 記述子カテゴリリスト ('constitutional', 'topological', etc.)
            descriptors: 特定の記述子名リスト（categoriesより優先）
            use_all: 全記述子を使用（非推奨、計算コスト高）
            include_smiles: 出力にSMILESカラムを含めるか
            error_value: 計算エラー時のデフォルト値
        """
        super().__init__(**kwargs)
        self.categories = categories
        self.descriptors = descriptors
        self.use_all = use_all
        self.include_smiles = include_smiles
        self.error_value = error_value
        
        # 計算対象の記述子を決定
        self._target_descriptors: List[str] = self._resolve_descriptors()
        
        # 記述子関数のキャッシュ
        self._descriptor_funcs: Dict[str, Any] = {}
        self._build_descriptor_cache()
    
    def _resolve_descriptors(self) -> List[str]:
        """使用する記述子リストを決定"""
        if self.descriptors:
            # 特定記述子が指定された場合
            return self.descriptors
        
        if self.use_all:
            # 全記述子（非推奨）
            logger.warning("全RDKit記述子を使用します（200+個、計算コスト高）")
            return get_all_rdkit_descriptors()
        
        if self.categories:
            # カテゴリから記述子を収集
            result: Set[str] = set()
            for cat in self.categories:
                if cat in DESCRIPTOR_CATEGORIES:
                    result.update(DESCRIPTOR_CATEGORIES[cat])
                else:
                    logger.warning(f"不明なカテゴリ: {cat}")
            return list(result)
        
        # デフォルト: 汎用セット（バランス型）
        return [
            'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
            'HeavyAtomCount', 'RingCount', 'LabuteASA',
            'Chi0', 'Chi1', 'Kappa1', 'Kappa2',
            'BertzCT', 'HallKierAlpha',
        ]
    
    def _build_descriptor_cache(self) -> None:
        """記述子関数のキャッシュを構築"""
        desc_dict = dict(Descriptors._descList)
        
        for name in self._target_descriptors:
            if name in desc_dict:
                self._descriptor_funcs[name] = desc_dict[name]
            elif name == 'qed':
                # QEDは別モジュール
                try:
                    from rdkit.Chem.QED import qed
                    self._descriptor_funcs['qed'] = qed
                except ImportError:
                    logger.warning("QED記述子が利用できません")
            else:
                logger.warning(f"不明な記述子: {name}")
    
    def _calculate_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """単一分子の記述子を計算"""
        result = {}
        
        for name, func in self._descriptor_funcs.items():
            try:
                value = func(mol)
                # NaNや無限大をチェック
                if value is None or not np.isfinite(value):
                    result[name] = self.error_value
                else:
                    result[name] = float(value)
            except Exception as e:
                logger.debug(f"記述子計算エラー ({name}): {e}")
                result[name] = self.error_value
        
        return result
    
    def transform(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        SMILESリストから記述子を抽出
        
        Args:
            smiles_list: SMILESのリスト
            
        Returns:
            pd.DataFrame: 記述子DataFrame
        """
        features = []
        valid_count = 0
        error_count = 0
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            
            if mol is not None:
                row = self._calculate_descriptors(mol)
                valid_count += 1
            else:
                # 無効なSMILES
                row = {name: self.error_value for name in self._descriptor_funcs.keys()}
                error_count += 1
                logger.debug(f"無効なSMILES: {smi}")
            
            features.append(row)
        
        if error_count > 0:
            logger.warning(f"{error_count}/{len(smiles_list)} SMILESの解析に失敗")
        
        df = pd.DataFrame(features)
        
        if self.include_smiles:
            df.insert(0, 'SMILES', smiles_list)
        
        return df
    
    @property
    def descriptor_names(self) -> List[str]:
        """使用する記述子名のリスト"""
        return list(self._descriptor_funcs.keys())
    
    @property
    def n_descriptors(self) -> int:
        """記述子の数"""
        return len(self._descriptor_funcs)
    
    @staticmethod
    def list_categories() -> Dict[str, List[str]]:
        """利用可能なカテゴリとその記述子を取得"""
        return DESCRIPTOR_CATEGORIES.copy()
    
    @staticmethod
    def list_all_descriptors() -> List[str]:
        """全RDKit記述子名を取得"""
        return get_all_rdkit_descriptors()
