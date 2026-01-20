"""
データインポート/エクスポート

Implements: F-IO-001
設計思想:
- 多様なフォーマット対応
- 検証とエラーハンドリング
- ストリーミング処理（大規模データ）

対応フォーマット:
- CSV/TSV
- SDF (Structure Data Format)
- JSON/JSONL
- Excel (オプション)
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Generator, Iterator
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """インポート結果"""
    total: int
    successful: int
    failed: int
    data: pd.DataFrame
    errors: List[Dict[str, Any]]
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0


class DataImporter:
    """
    データインポーター
    
    Usage:
        importer = DataImporter()
        result = importer.import_csv("data.csv", smiles_column="SMILES")
    """
    
    def import_csv(
        self,
        filepath: str,
        smiles_column: str = "smiles",
        name_column: str = None,
        target_column: str = None,
        validate_smiles: bool = True,
        **kwargs,
    ) -> ImportResult:
        """CSVをインポート"""
        df = pd.read_csv(filepath, **kwargs)
        return self._process_dataframe(
            df, smiles_column, name_column, target_column, validate_smiles
        )
    
    def import_tsv(
        self,
        filepath: str,
        smiles_column: str = "smiles",
        **kwargs,
    ) -> ImportResult:
        """TSVをインポート"""
        kwargs['sep'] = '\t'
        return self.import_csv(filepath, smiles_column, **kwargs)
    
    def import_excel(
        self,
        filepath: str,
        smiles_column: str = "smiles",
        sheet_name: str = None,
        **kwargs,
    ) -> ImportResult:
        """Excelをインポート"""
        if sheet_name:
            kwargs['sheet_name'] = sheet_name
        df = pd.read_excel(filepath, **kwargs)
        return self._process_dataframe(df, smiles_column)
    
    def import_sdf(
        self,
        filepath: str,
        property_names: List[str] = None,
    ) -> ImportResult:
        """SDFをインポート"""
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit is required for SDF import")
        
        data = []
        errors = []
        
        suppl = Chem.SDMolSupplier(filepath)
        
        for i, mol in enumerate(suppl):
            if mol is None:
                errors.append({'index': i, 'error': 'Invalid molecule'})
                continue
            
            row = {
                'smiles': Chem.MolToSmiles(mol),
                'name': mol.GetProp('_Name') if mol.HasProp('_Name') else None,
            }
            
            if property_names:
                for prop in property_names:
                    if mol.HasProp(prop):
                        row[prop] = mol.GetProp(prop)
            else:
                # 全プロパティを取得
                for prop in mol.GetPropsAsDict():
                    if not prop.startswith('_'):
                        row[prop] = mol.GetProp(prop)
            
            data.append(row)
        
        return ImportResult(
            total=len(suppl),
            successful=len(data),
            failed=len(errors),
            data=pd.DataFrame(data),
            errors=errors,
        )
    
    def import_json(
        self,
        filepath: str,
        smiles_key: str = "smiles",
    ) -> ImportResult:
        """JSONをインポート"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'molecules' in data:
            df = pd.DataFrame(data['molecules'])
        else:
            df = pd.DataFrame([data])
        
        return self._process_dataframe(df, smiles_key)
    
    def import_jsonl(
        self,
        filepath: str,
        smiles_key: str = "smiles",
    ) -> ImportResult:
        """JSONLをインポート（行ごとのJSON）"""
        data = []
        errors = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line.strip())
                    data.append(row)
                except json.JSONDecodeError as e:
                    errors.append({'line': i, 'error': str(e)})
        
        df = pd.DataFrame(data)
        result = self._process_dataframe(df, smiles_key)
        result.errors.extend(errors)
        
        return result
    
    def _process_dataframe(
        self,
        df: pd.DataFrame,
        smiles_column: str,
        name_column: str = None,
        target_column: str = None,
        validate_smiles: bool = True,
    ) -> ImportResult:
        """DataFrameを処理"""
        errors = []
        
        # カラム正規化
        df.columns = df.columns.str.lower().str.strip()
        smiles_column = smiles_column.lower()
        
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column not found: {smiles_column}")
        
        # SMILES検証
        if validate_smiles:
            try:
                from rdkit import Chem
                
                valid_mask = df[smiles_column].apply(
                    lambda x: Chem.MolFromSmiles(str(x)) is not None
                )
                
                invalid_indices = df[~valid_mask].index.tolist()
                for idx in invalid_indices:
                    errors.append({
                        'index': idx,
                        'smiles': df.loc[idx, smiles_column],
                        'error': 'Invalid SMILES'
                    })
                
                df = df[valid_mask].reset_index(drop=True)
            except ImportError:
                logger.warning("RDKit not available, skipping validation")
        
        return ImportResult(
            total=len(df) + len(errors),
            successful=len(df),
            failed=len(errors),
            data=df,
            errors=errors,
        )


class DataExporter:
    """
    データエクスポーター
    
    Usage:
        exporter = DataExporter()
        exporter.export_csv(df, "output.csv")
    """
    
    def export_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        **kwargs,
    ) -> None:
        """CSVにエクスポート"""
        kwargs.setdefault('index', False)
        df.to_csv(filepath, **kwargs)
        logger.info(f"Exported {len(df)} rows to {filepath}")
    
    def export_tsv(
        self,
        df: pd.DataFrame,
        filepath: str,
        **kwargs,
    ) -> None:
        """TSVにエクスポート"""
        kwargs['sep'] = '\t'
        self.export_csv(df, filepath, **kwargs)
    
    def export_sdf(
        self,
        df: pd.DataFrame,
        filepath: str,
        smiles_column: str = "smiles",
        name_column: str = None,
        property_columns: List[str] = None,
    ) -> None:
        """SDFにエクスポート"""
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit is required for SDF export")
        
        writer = Chem.SDWriter(filepath)
        
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(str(row[smiles_column]))
            if mol is None:
                continue
            
            # 名前
            if name_column and name_column in row:
                mol.SetProp('_Name', str(row[name_column]))
            
            # プロパティ
            columns = property_columns or [
                c for c in df.columns
                if c not in [smiles_column, name_column]
            ]
            
            for col in columns:
                if pd.notna(row[col]):
                    mol.SetProp(col, str(row[col]))
            
            writer.write(mol)
        
        writer.close()
        logger.info(f"Exported {len(df)} molecules to {filepath}")
    
    def export_json(
        self,
        df: pd.DataFrame,
        filepath: str,
        orient: str = "records",
    ) -> None:
        """JSONにエクスポート"""
        df.to_json(filepath, orient=orient, force_ascii=False, indent=2)
        logger.info(f"Exported {len(df)} rows to {filepath}")
    
    def export_jsonl(
        self,
        df: pd.DataFrame,
        filepath: str,
    ) -> None:
        """JSONLにエクスポート"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(df)} rows to {filepath}")


def import_data(
    filepath: str,
    smiles_column: str = "smiles",
    format: str = None,
) -> ImportResult:
    """
    便利関数: データインポート
    
    フォーマットは拡張子から自動判定。
    """
    path = Path(filepath)
    fmt = format or path.suffix.lower()
    
    importer = DataImporter()
    
    if fmt in ['.csv']:
        return importer.import_csv(filepath, smiles_column)
    elif fmt in ['.tsv', '.txt']:
        return importer.import_tsv(filepath, smiles_column)
    elif fmt in ['.sdf', '.mol']:
        return importer.import_sdf(filepath)
    elif fmt in ['.json']:
        return importer.import_json(filepath, smiles_column)
    elif fmt in ['.jsonl', '.ndjson']:
        return importer.import_jsonl(filepath, smiles_column)
    elif fmt in ['.xlsx', '.xls']:
        return importer.import_excel(filepath, smiles_column)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def export_data(
    df: pd.DataFrame,
    filepath: str,
    format: str = None,
) -> None:
    """
    便利関数: データエクスポート
    """
    path = Path(filepath)
    fmt = format or path.suffix.lower()
    
    exporter = DataExporter()
    
    if fmt in ['.csv']:
        exporter.export_csv(df, filepath)
    elif fmt in ['.tsv', '.txt']:
        exporter.export_tsv(df, filepath)
    elif fmt in ['.sdf', '.mol']:
        exporter.export_sdf(df, filepath)
    elif fmt in ['.json']:
        exporter.export_json(df, filepath)
    elif fmt in ['.jsonl', '.ndjson']:
        exporter.export_jsonl(df, filepath)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
