"""
データローダー

Implements: F-DATALOADER-001
設計思想:
- 複数フォーマット対応
- SMILESバリデーション
- 自動カラム検出
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChemDataLoader:
    """
    化学データローダー
    
    Features:
    - CSV/Excel/SDF読み込み
    - SMILESバリデーション
    - 自動カラム検出
    - データ分割
    
    Example:
        >>> loader = ChemDataLoader()
        >>> df = loader.load("data.csv")
        >>> X, y, smiles = loader.prepare(df, target="solubility")
    """
    
    def __init__(self):
        pass
    
    def load(
        self,
        filepath: str,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """ファイル読み込み"""
        path = Path(filepath)
        
        if path.suffix.lower() == '.csv':
            return pd.read_csv(filepath)
        
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(filepath, sheet_name=sheet_name)
        
        elif path.suffix.lower() == '.sdf':
            return self._load_sdf(filepath)
        
        elif path.suffix.lower() == '.json':
            return pd.read_json(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def _load_sdf(self, filepath: str) -> pd.DataFrame:
        """SDFファイル読み込み"""
        try:
            from rdkit import Chem
            from rdkit.Chem import PandasTools
            
            df = PandasTools.LoadSDF(filepath)
            
            # SMILESカラム追加
            if 'SMILES' not in df.columns and 'ROMol' in df.columns:
                df['SMILES'] = df['ROMol'].apply(
                    lambda m: Chem.MolToSmiles(m) if m else None
                )
            
            return df
            
        except Exception as e:
            logger.error(f"SDF loading failed: {e}")
            raise
    
    def prepare(
        self,
        df: pd.DataFrame,
        target: str,
        smiles_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        データ準備
        
        Returns:
            (X, y, smiles_list)
        """
        # SMILESカラム検出
        if smiles_col is None:
            smiles_col = self._detect_smiles_column(df)
        
        smiles_list = df[smiles_col].tolist() if smiles_col else []
        
        # ターゲット
        if target not in df.columns:
            raise ValueError(f"Target column not found: {target}")
        y = df[target]
        
        # 特徴量
        if feature_cols is None:
            exclude = [target, smiles_col] if smiles_col else [target]
            feature_cols = [c for c in df.columns if c not in exclude]
            feature_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64]]
        
        X = df[feature_cols]
        
        return X, y, smiles_list
    
    def _detect_smiles_column(self, df: pd.DataFrame) -> Optional[str]:
        """SMILESカラムを自動検出"""
        candidates = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'mol', 'structure']
        
        for c in candidates:
            if c in df.columns:
                return c
        
        # 文字列カラムでSMILESっぽいものを探す
        for c in df.columns:
            if df[c].dtype == object:
                sample = df[c].dropna().iloc[:5].tolist()
                if all(self._looks_like_smiles(s) for s in sample):
                    return c
        
        return None
    
    def _looks_like_smiles(self, s: str) -> bool:
        """SMILESっぽいか判定"""
        if not isinstance(s, str):
            return False
        
        # 基本的なSMILES文字
        smiles_chars = set('CNOPSFIBrcnopsfibl[]()=#@+-0123456789')
        return all(c in smiles_chars for c in s)
    
    def validate_smiles(self, smiles_list: List[str]) -> Dict[str, Any]:
        """SMILESをバリデート"""
        try:
            from rdkit import Chem
            
            valid = []
            invalid = []
            
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    valid.append(smi)
                else:
                    invalid.append(smi)
            
            return {
                'total': len(smiles_list),
                'valid': len(valid),
                'invalid_count': len(invalid),
                'invalid_smiles': invalid[:10],
                'validity_rate': len(valid) / len(smiles_list) if smiles_list else 0,
            }
            
        except Exception:
            return {'error': 'RDKit not available'}
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """データ分割"""
        from sklearn.model_selection import train_test_split
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )
