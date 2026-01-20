"""
統合タスクハンドラ - 4タスクタイプ対応

Implements: F-TASK-UNIFIED-001
設計思想:
- ① SMILES単独 → 物性予測
- ② 表データ（SMILESなし） → 特性予測
- ③ 混合物（SMILES + 割合） → 物性予測
- ④ SMILES + 表データ → 物性予測

それぞれのタスクタイプを統一インターフェースで扱う
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """タスクタイプ"""
    SMILES_ONLY = "smiles_only"           # ① SMILES単独
    TABULAR_ONLY = "tabular_only"         # ② 表データのみ
    MIXTURE = "mixture"                    # ③ 混合物
    SMILES_WITH_TABULAR = "smiles_tabular"  # ④ SMILES + 表データ


class UnifiedTaskHandler:
    """
    統合タスクハンドラ
    
    Features:
    - 4つのタスクタイプを統一インターフェースで処理
    - データ形式の自動検出
    - 適切な特徴量抽出器の選択
    
    Example:
        >>> handler = UnifiedTaskHandler()
        >>> task_type = handler.detect_task_type(df)
        >>> features = handler.extract_features(df, task_type)
    """
    
    def __init__(
        self,
        smiles_extractors: Optional[List[Any]] = None,
        mixture_extractor: Optional[Any] = None,
    ):
        """
        Args:
            smiles_extractors: SMILES用特徴量抽出器リスト
            mixture_extractor: 混合物用特徴量抽出器
        """
        self.smiles_extractors = smiles_extractors or []
        self.mixture_extractor = mixture_extractor
        
        # デフォルト抽出器を設定
        if not self.smiles_extractors:
            from .features.rdkit_eng import RDKitFeatureExtractor
            self.smiles_extractors = [RDKitFeatureExtractor()]
        
        if self.mixture_extractor is None:
            from .features.mixture_eng import MixtureFeatureExtractor
            self.mixture_extractor = MixtureFeatureExtractor()
    
    def detect_task_type(
        self,
        df: pd.DataFrame,
        smiles_col: Optional[str] = None,
        ratio_col: Optional[str] = None,
    ) -> TaskType:
        """
        データからタスクタイプを自動検出
        
        Args:
            df: 入力DataFrame
            smiles_col: SMILESカラム名
            ratio_col: 割合カラム名
            
        Returns:
            TaskType
        """
        # SMILESカラムの検出
        smiles_candidates = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'mol']
        if smiles_col:
            smiles_candidates = [smiles_col] + smiles_candidates
        
        has_smiles = any(c in df.columns for c in smiles_candidates)
        
        # 割合カラムの検出
        ratio_candidates = ['ratio', 'Ratio', 'fraction', 'percent', '%', 'weight', 'wt']
        if ratio_col:
            ratio_candidates = [ratio_col] + ratio_candidates
        
        has_ratio = any(c in df.columns for c in ratio_candidates)
        
        # 他の数値カラムの存在
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_ratio_numeric = [c for c in numeric_cols if c.lower() not in [r.lower() for r in ratio_candidates]]
        has_tabular = len(non_ratio_numeric) > 0
        
        # タスクタイプ判定
        if has_smiles and has_ratio:
            return TaskType.MIXTURE
        elif has_smiles and has_tabular:
            return TaskType.SMILES_WITH_TABULAR
        elif has_smiles:
            return TaskType.SMILES_ONLY
        else:
            return TaskType.TABULAR_ONLY
    
    def extract_features(
        self,
        df: pd.DataFrame,
        task_type: TaskType,
        target_col: str = 'target',
        smiles_col: str = 'SMILES',
        ratio_col: str = 'ratio',
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        タスクタイプに応じた特徴量抽出
        
        Args:
            df: 入力DataFrame
            task_type: タスクタイプ
            target_col: ターゲットカラム
            smiles_col: SMILESカラム
            ratio_col: 割合カラム
            
        Returns:
            (特徴量DataFrame, ターゲットSeries)
        """
        # ターゲット抽出
        if target_col in df.columns:
            y = df[target_col]
        else:
            y = pd.Series([np.nan] * len(df))
        
        if task_type == TaskType.SMILES_ONLY:
            return self._extract_smiles_only(df, smiles_col), y
        
        elif task_type == TaskType.TABULAR_ONLY:
            return self._extract_tabular_only(df, target_col), y
        
        elif task_type == TaskType.MIXTURE:
            return self._extract_mixture(df, smiles_col, ratio_col), y
        
        elif task_type == TaskType.SMILES_WITH_TABULAR:
            return self._extract_smiles_with_tabular(df, smiles_col, target_col), y
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _extract_smiles_only(
        self,
        df: pd.DataFrame,
        smiles_col: str,
    ) -> pd.DataFrame:
        """① SMILES単独での特徴量抽出"""
        # SMILESカラムを探す
        col = self._find_column(df, [smiles_col, 'smiles', 'SMILES', 'Smiles'])
        if col is None:
            raise ValueError("SMILES column not found")
        
        smiles_list = df[col].tolist()
        
        # 全抽出器で特徴量を取得
        features_list = []
        for extractor in self.smiles_extractors:
            feat_df = extractor.transform(smiles_list)
            feat_df = feat_df.drop(columns=['SMILES'], errors='ignore')
            features_list.append(feat_df)
        
        return pd.concat(features_list, axis=1)
    
    def _extract_tabular_only(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """② 表データのみの特徴量抽出"""
        # ターゲット以外の数値カラムを使用
        exclude = [target_col]
        numeric_df = df.select_dtypes(include=[np.number])
        
        feature_cols = [c for c in numeric_df.columns if c not in exclude]
        return numeric_df[feature_cols].copy()
    
    def _extract_mixture(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        ratio_col: str,
    ) -> pd.DataFrame:
        """③ 混合物の特徴量抽出"""
        from .features.mixture_eng import MixtureComponent
        
        # グループ化キーを探す（同じ混合物の成分をグループ化）
        group_col = self._find_column(df, ['mixture_id', 'group', 'sample_id', 'id'])
        
        if group_col:
            # グループごとに混合物を構成
            mixtures = []
            groups = df.groupby(group_col)
            
            for _, group_df in groups:
                components = []
                for _, row in group_df.iterrows():
                    smi = row.get(smiles_col, row.get('SMILES', ''))
                    ratio = row.get(ratio_col, row.get('ratio', 100))
                    if smi:
                        components.append(MixtureComponent(smi, ratio))
                mixtures.append(components)
        else:
            # 各行が1つの混合物（カンマ区切り等）
            mixtures = []
            for _, row in df.iterrows():
                smi_val = row.get(smiles_col, row.get('SMILES', ''))
                ratio_val = row.get(ratio_col, 100)
                
                # パース
                if isinstance(smi_val, str) and ',' in smi_val:
                    smiles_list = [s.strip() for s in smi_val.split(',')]
                    if isinstance(ratio_val, str):
                        ratios = [float(r.strip().replace('%', '')) for r in ratio_val.split(',')]
                    else:
                        ratios = [100 / len(smiles_list)] * len(smiles_list)
                    
                    components = [
                        MixtureComponent(s, r) for s, r in zip(smiles_list, ratios)
                    ]
                else:
                    components = [MixtureComponent(smi_val, ratio_val)]
                
                mixtures.append(components)
        
        return self.mixture_extractor.transform(mixtures)
    
    def _extract_smiles_with_tabular(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        """④ SMILES + 表データの特徴量抽出"""
        # SMILES特徴
        smiles_features = self._extract_smiles_only(df, smiles_col)
        
        # 表データ特徴
        exclude = [smiles_col, target_col, 'SMILES', 'smiles', 'Smiles']
        tabular_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        tabular_features = df[tabular_cols].copy() if tabular_cols else pd.DataFrame()
        
        # 結合
        return pd.concat([smiles_features.reset_index(drop=True), 
                         tabular_features.reset_index(drop=True)], axis=1)
    
    def _find_column(
        self,
        df: pd.DataFrame,
        candidates: List[str],
    ) -> Optional[str]:
        """カラム検索"""
        for c in candidates:
            if c in df.columns:
                return c
        return None


def create_handler_for_task(task_type: TaskType, **kwargs) -> UnifiedTaskHandler:
    """タスクタイプに最適化されたハンドラを作成"""
    if task_type == TaskType.SMILES_ONLY:
        from .features.rdkit_eng import RDKitFeatureExtractor
        return UnifiedTaskHandler(
            smiles_extractors=[RDKitFeatureExtractor(**kwargs)]
        )
    
    elif task_type == TaskType.MIXTURE:
        from .features.mixture_eng import MixtureFeatureExtractor
        return UnifiedTaskHandler(
            mixture_extractor=MixtureFeatureExtractor(**kwargs)
        )
    
    return UnifiedTaskHandler(**kwargs)
