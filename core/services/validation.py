"""
入力バリデーション

Implements: F-VALIDATE-001
設計思想:
- SMILES検証
- データフレーム検証
- 設定検証
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """バリデーション結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class SMILESValidator:
    """
    SMILESバリデーター
    
    Example:
        >>> validator = SMILESValidator()
        >>> result = validator.validate("CCO")
    """
    
    def validate(self, smiles: str) -> ValidationResult:
        """単一SMILESを検証"""
        errors = []
        warnings = []
        
        try:
            from rdkit import Chem
            
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                errors.append(f"Invalid SMILES: {smiles}")
            else:
                # 警告チェック
                n_atoms = mol.GetNumAtoms()
                if n_atoms > 200:
                    warnings.append(f"Large molecule: {n_atoms} atoms")
                
                if n_atoms < 2:
                    warnings.append(f"Very small molecule: {n_atoms} atoms")
                
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats={'smiles': smiles},
        )
    
    def validate_batch(self, smiles_list: List[str]) -> ValidationResult:
        """バッチ検証"""
        all_errors = []
        all_warnings = []
        valid_count = 0
        
        for i, smi in enumerate(smiles_list):
            result = self.validate(smi)
            if result.is_valid:
                valid_count += 1
            else:
                all_errors.extend([f"[{i}] {e}" for e in result.errors])
            all_warnings.extend(result.warnings)
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            stats={
                'total': len(smiles_list),
                'valid': valid_count,
                'invalid': len(smiles_list) - valid_count,
            },
        )


class DataFrameValidator:
    """
    DataFrameバリデーター
    """
    
    def validate(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> ValidationResult:
        """DataFrameを検証"""
        errors = []
        warnings = []
        
        # 空チェック
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings, {})
        
        # 必須カラムチェック
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                errors.append(f"Missing columns: {missing}")
        
        # 数値カラムチェック
        if numeric_cols:
            for col in numeric_cols:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' is not numeric")
        
        # 欠損値警告
        missing_pct = df.isnull().sum().sum() / df.size * 100
        if missing_pct > 10:
            warnings.append(f"High missing rate: {missing_pct:.1f}%")
        
        # 重複警告
        dup_pct = df.duplicated().sum() / len(df) * 100
        if dup_pct > 5:
            warnings.append(f"Duplicates: {dup_pct:.1f}%")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats={
                'rows': len(df),
                'cols': len(df.columns),
                'missing_pct': missing_pct,
            },
        )


class ConfigValidator:
    """
    設定バリデーター
    """
    
    VALID_MODEL_TYPES = ['lightgbm', 'xgboost', 'random_forest']
    VALID_FEATURE_TYPES = ['rdkit', 'xtb', 'uma', 'morgan']
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """設定を検証"""
        errors = []
        warnings = []
        
        # モデルタイプ
        model_type = config.get('model', {}).get('model_type')
        if model_type and model_type not in self.VALID_MODEL_TYPES:
            errors.append(f"Invalid model_type: {model_type}")
        
        # 数値パラメータ
        n_estimators = config.get('model', {}).get('n_estimators', 100)
        if n_estimators < 1:
            errors.append("n_estimators must be >= 1")
        elif n_estimators > 10000:
            warnings.append("n_estimators is very high")
        
        test_size = config.get('training', {}).get('test_size', 0.2)
        if not 0 < test_size < 1:
            errors.append("test_size must be between 0 and 1")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=config,
        )
