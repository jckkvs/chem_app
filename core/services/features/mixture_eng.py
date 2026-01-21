"""
混合物記述子計算エンジン

Implements: F-MIXTURE-001
設計思想:
- 複数SMILESと割合から加重平均記述子を計算
- 混合物として物性予測に使用
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MixtureComponent:
    """混合物成分"""
    smiles: str
    ratio: float  # 0.0 - 1.0 または 0-100%
    name: Optional[str] = None
    
    def __post_init__(self):
        # パーセント表記を0-1に変換
        if self.ratio > 1.0:
            self.ratio = self.ratio / 100.0


class MixtureFeatureExtractor:
    """
    混合物特徴量抽出器
    
    Features:
    - 各成分のRDKit記述子を計算
    - 割合で加重平均
    - 追加の混合特性（成分数、最大/最小比率等）
    
    Example:
        >>> extractor = MixtureFeatureExtractor()
        >>> components = [
        ...     MixtureComponent("C=C", 0.6),   # エチレン60%
        ...     MixtureComponent("CCCC", 0.4),  # ブタン40%
        ... ]
        >>> features = extractor.transform([components])
    """
    
    def __init__(
        self,
        base_extractor=None,
        include_mixture_features: bool = True,
    ):
        """
        Args:
            base_extractor: ベースの特徴量抽出器（RDKitFeatureExtractor等）
            include_mixture_features: 混合物固有特徴を追加するか
        """
        if base_extractor is None:
            from .rdkit_eng import RDKitFeatureExtractor
            base_extractor = RDKitFeatureExtractor(
                categories=['lipophilicity', 'structural', 'topological']
            )
        
        self.base_extractor = base_extractor
        self.include_mixture_features = include_mixture_features
    
    def transform(
        self,
        mixtures: List[List[MixtureComponent]],
    ) -> pd.DataFrame:
        """
        混合物リストから特徴量を抽出
        
        Args:
            mixtures: 各混合物は成分リスト
            
        Returns:
            pd.DataFrame: 加重平均記述子
        """
        all_features = []
        
        for mixture in mixtures:
            features = self._transform_single_mixture(mixture)
            all_features.append(features)
        
        return pd.DataFrame(all_features)
    
    def _transform_single_mixture(
        self,
        components: List[MixtureComponent],
    ) -> Dict[str, float]:
        """単一混合物の特徴量を計算"""
        if not components:
            return {}
        
        # 割合の正規化
        total_ratio = sum(c.ratio for c in components)
        if total_ratio == 0:
            total_ratio = 1.0
        
        normalized = [
            MixtureComponent(c.smiles, c.ratio / total_ratio, c.name)
            for c in components
        ]
        
        # 各成分の記述子を取得
        smiles_list = [c.smiles for c in normalized]
        ratios = np.array([c.ratio for c in normalized])
        
        base_df = self.base_extractor.transform(smiles_list)
        
        # SMILESカラムを除外
        numeric_cols = base_df.select_dtypes(include=[np.number]).columns
        base_array = base_df[numeric_cols].values
        
        # 加重平均
        weighted_avg = (base_array * ratios.reshape(-1, 1)).sum(axis=0)
        
        result = {col: float(weighted_avg[i]) for i, col in enumerate(numeric_cols)}
        
        # 混合物固有特徴
        if self.include_mixture_features:
            result['n_components'] = len(components)
            result['max_ratio'] = float(max(ratios))
            result['min_ratio'] = float(min(ratios))
            result['ratio_std'] = float(np.std(ratios))
            
            # 各成分の記述子の標準偏差（混合の不均一性）
            if len(components) > 1:
                for i, col in enumerate(numeric_cols):
                    result[f'{col}_std'] = float(np.std(base_array[:, i]))
        
        return result
    
    @staticmethod
    def parse_mixture_string(mixture_str: str) -> List[MixtureComponent]:
        """
        文字列から混合物成分をパース
        
        フォーマット例:
        - "C=C:60,CCCC:40"
        - "ethylene 60% butane 40%"
        - "C=C|0.6|CCC|0.4"
        
        Args:
            mixture_str: 混合物文字列
            
        Returns:
            成分リスト
        """
        components = []
        
        # カンマまたはパイプで分割
        if '|' in mixture_str:
            parts = mixture_str.split('|')
            for i in range(0, len(parts) - 1, 2):
                smiles = parts[i].strip()
                try:
                    ratio = float(parts[i + 1].strip().replace('%', ''))
                    components.append(MixtureComponent(smiles, ratio))
                except (ValueError, IndexError):
                    pass
        
        elif ':' in mixture_str:
            for part in mixture_str.split(','):
                if ':' in part:
                    smiles, ratio_str = part.split(':')
                    try:
                        ratio = float(ratio_str.strip().replace('%', ''))
                        components.append(MixtureComponent(smiles.strip(), ratio))
                    except ValueError:
                        pass
        
        else:
            # 単一SMILES
            smiles = mixture_str.strip()
            if smiles:
                components.append(MixtureComponent(smiles, 100.0))
        
        return components
    
    @staticmethod
    def format_mixture(components: List[MixtureComponent]) -> str:
        """混合物を文字列にフォーマット"""
        parts = [f"{c.smiles}:{c.ratio*100:.0f}%" for c in components]
        return ", ".join(parts)


def create_mixture_from_ratio_dict(
    ratio_dict: Dict[str, float]
) -> List[MixtureComponent]:
    """
    SMILES→割合の辞書から混合物を作成
    
    Args:
        ratio_dict: {"C=C": 60, "CCCC": 40}
        
    Returns:
        成分リスト
    """
    return [
        MixtureComponent(smiles=smi, ratio=ratio)
        for smi, ratio in ratio_dict.items()
    ]
