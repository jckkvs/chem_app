"""
入力検証ユーティリティ

Implements: F-VALID-001
設計思想:
- API入力の検証
- 型チェックと範囲チェック
- 分かりやすいエラーメッセージ

機能:
- SMILES検証
- 数値範囲検証
- データフレーム構造検証
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any, Union, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """検証エラー"""
    field: str
    message: str
    value: Any = None


class ValidationResult:
    """検証結果"""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
    
    def add_error(
        self,
        field: str,
        message: str,
        value: Any = None,
    ) -> None:
        self.errors.append(ValidationError(field, message, value))
    
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            error_messages = [
                f"{e.field}: {e.message}" for e in self.errors
            ]
            raise ValueError("\n".join(error_messages))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'valid': self.is_valid,
            'errors': [
                {'field': e.field, 'message': e.message}
                for e in self.errors
            ],
            'warnings': self.warnings,
        }


class InputValidator:
    """
    入力検証器
    
    Usage:
        validator = InputValidator()
        result = validator.validate_smiles_input(smiles_list)
        result.raise_if_invalid()
    """
    
    def validate_smiles(
        self,
        smiles: str,
        allow_empty: bool = False,
    ) -> ValidationResult:
        """単一SMILESを検証"""
        result = ValidationResult()
        
        if not smiles:
            if not allow_empty:
                result.add_error('smiles', 'SMILES is required')
            return result
        
        if not isinstance(smiles, str):
            result.add_error('smiles', 'SMILES must be a string', smiles)
            return result
        
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.add_error('smiles', 'Invalid SMILES structure', smiles)
        except ImportError:
            result.add_warning('RDKit not available, skipping validation')
        
        return result
    
    def validate_smiles_list(
        self,
        smiles_list: List[str],
        min_count: int = 1,
        max_count: int = None,
    ) -> ValidationResult:
        """SMILESリストを検証"""
        result = ValidationResult()
        
        if not isinstance(smiles_list, (list, tuple)):
            result.add_error('smiles_list', 'Must be a list')
            return result
        
        if len(smiles_list) < min_count:
            result.add_error(
                'smiles_list',
                f'At least {min_count} SMILES required',
                len(smiles_list)
            )
        
        if max_count and len(smiles_list) > max_count:
            result.add_error(
                'smiles_list',
                f'Maximum {max_count} SMILES allowed',
                len(smiles_list)
            )
        
        # 個別検証（最初の10件のみ詳細エラー）
        invalid_count = 0
        try:
            from rdkit import Chem
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None:
                    invalid_count += 1
                    if invalid_count <= 10:
                        result.add_error(
                            f'smiles_list[{i}]',
                            'Invalid SMILES',
                            smi
                        )
            
            if invalid_count > 10:
                result.add_warning(
                    f'{invalid_count - 10} more invalid SMILES not shown'
                )
        except ImportError:
            result.add_warning('RDKit not available')
        
        return result
    
    def validate_numeric(
        self,
        value: Any,
        field_name: str,
        min_value: float = None,
        max_value: float = None,
        allow_none: bool = False,
    ) -> ValidationResult:
        """数値を検証"""
        result = ValidationResult()
        
        if value is None:
            if not allow_none:
                result.add_error(field_name, 'Value is required')
            return result
        
        try:
            num = float(value)
        except (TypeError, ValueError):
            result.add_error(field_name, 'Must be a number', value)
            return result
        
        if min_value is not None and num < min_value:
            result.add_error(
                field_name,
                f'Must be >= {min_value}',
                num
            )
        
        if max_value is not None and num > max_value:
            result.add_error(
                field_name,
                f'Must be <= {max_value}',
                num
            )
        
        return result
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None,
        min_rows: int = 1,
    ) -> ValidationResult:
        """DataFrameを検証"""
        result = ValidationResult()
        
        if not isinstance(df, pd.DataFrame):
            result.add_error('dataframe', 'Must be a pandas DataFrame')
            return result
        
        if len(df) < min_rows:
            result.add_error(
                'dataframe',
                f'At least {min_rows} rows required',
                len(df)
            )
        
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                result.add_error(
                    'columns',
                    f'Missing columns: {missing}'
                )
        
        return result
    
    def validate_model_params(
        self,
        params: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
    ) -> ValidationResult:
        """モデルパラメータを検証"""
        result = ValidationResult()
        
        for param_name, rules in schema.items():
            value = params.get(param_name)
            
            required = rules.get('required', False)
            if required and value is None:
                result.add_error(param_name, 'Parameter is required')
                continue
            
            if value is None:
                continue
            
            # 型チェック
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                result.add_error(
                    param_name,
                    f'Must be of type {expected_type.__name__}',
                    type(value).__name__
                )
            
            # 範囲チェック
            if 'min' in rules and value < rules['min']:
                result.add_error(param_name, f"Must be >= {rules['min']}", value)
            if 'max' in rules and value > rules['max']:
                result.add_error(param_name, f"Must be <= {rules['max']}", value)
            
            # 選択肢チェック
            if 'choices' in rules and value not in rules['choices']:
                result.add_error(
                    param_name,
                    f"Must be one of {rules['choices']}",
                    value
                )
        
        return result


def validate_smiles(smiles: str) -> bool:
    """便利関数: SMILES検証"""
    validator = InputValidator()
    return validator.validate_smiles(smiles).is_valid


def validate_smiles_list(smiles_list: List[str]) -> ValidationResult:
    """便利関数: SMILESリスト検証"""
    validator = InputValidator()
    return validator.validate_smiles_list(smiles_list)


# パラメータスキーマ例
MODEL_PARAM_SCHEMA = {
    'n_estimators': {
        'type': int,
        'min': 1,
        'max': 10000,
        'required': False,
    },
    'max_depth': {
        'type': int,
        'min': 1,
        'max': 100,
        'required': False,
    },
    'learning_rate': {
        'type': float,
        'min': 0.001,
        'max': 1.0,
        'required': False,
    },
    'model_type': {
        'choices': ['rf', 'gb', 'lgbm', 'xgb'],
        'required': True,
    },
}
