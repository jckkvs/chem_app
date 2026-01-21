"""
分子データベース管理

Implements: F-DB-001
設計思想:
- 分子データの一元管理
- プロジェクト・実験の階層構造
- メタデータとタグ付け
- 検索・フィルタリング

機能:
- 分子登録・検索
- 実験結果の保存・取得
- プロジェクト管理
- データエクスポート（CSV, SDF, JSON）
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Molecule:
    """分子エンティティ"""
    smiles: str
    name: Optional[str] = None
    inchi: Optional[str] = None
    inchi_key: Optional[str] = None
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    @property
    def hash_id(self) -> str:
        """SMILESからユニークID生成"""
        return hashlib.md5(self.smiles.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_smiles(cls, smiles: str, name: str = None) -> 'Molecule':
        """SMILESから分子を作成"""
        mol = cls(smiles=smiles, name=name)
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, inchi
            
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol:
                mol.molecular_formula = Chem.rdMolDescriptors.CalcMolFormula(rdmol)
                mol.molecular_weight = Descriptors.MolWt(rdmol)
                mol.inchi = inchi.MolToInchi(rdmol)
                mol.inchi_key = inchi.MolToInchiKey(rdmol)
        except Exception as e:
            logger.debug(f"RDKit processing failed: {e}")
        
        return mol


@dataclass
class Experiment:
    """実験エンティティ"""
    id: str
    name: str
    project_id: str
    model_type: str
    target_property: str
    molecules: List[str] = field(default_factory=list)  # SMILES list
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: str = "created"  # created, running, completed, failed
    
    def __post_init__(self):
        now = datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now


@dataclass
class Project:
    """プロジェクトエンティティ"""
    id: str
    name: str
    description: str = ""
    experiments: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class MoleculeDatabase:
    """
    分子データベース管理
    
    SQLiteベースの軽量データベース。
    分子、実験、プロジェクトを統合管理。
    
    Usage:
        db = MoleculeDatabase("my_project.db")
        
        # 分子登録
        db.add_molecule("CCO", name="Ethanol", tags=["solvent"])
        
        # 検索
        molecules = db.search_molecules(tags=["solvent"])
        
        # エクスポート
        db.export_csv("molecules.csv")
    """
    
    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """
        Args:
            db_path: データベースファイルパス（":memory:"でインメモリ）
        """
        self.db_path = Path(db_path) if db_path != ":memory:" else db_path
        self.conn = sqlite3.connect(str(db_path))
        self._init_tables()
    
    def _init_tables(self):
        """テーブル初期化"""
        cursor = self.conn.cursor()
        
        # 分子テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS molecules (
                id TEXT PRIMARY KEY,
                smiles TEXT UNIQUE NOT NULL,
                name TEXT,
                inchi TEXT,
                inchi_key TEXT,
                molecular_formula TEXT,
                molecular_weight REAL,
                properties TEXT,
                tags TEXT,
                created_at TEXT
            )
        ''')
        
        # プロジェクトテーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                created_at TEXT
            )
        ''')
        
        # 実験テーブル
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project_id TEXT,
                model_type TEXT,
                target_property TEXT,
                molecules TEXT,
                results TEXT,
                metrics TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')
        
        # プロパティ値テーブル（実験結果）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS property_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id TEXT,
                experiment_id TEXT,
                property_name TEXT,
                value REAL,
                uncertainty REAL,
                created_at TEXT,
                FOREIGN KEY (molecule_id) REFERENCES molecules(id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        ''')
        
        self.conn.commit()
    
    # ========== 分子操作 ==========
    
    def add_molecule(
        self,
        smiles: str,
        name: str = None,
        properties: Dict[str, Any] = None,
        tags: List[str] = None,
    ) -> Molecule:
        """分子を追加"""
        mol = Molecule.from_smiles(smiles, name)
        mol.properties = properties or {}
        mol.tags = tags or []
        
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO molecules 
                (id, smiles, name, inchi, inchi_key, molecular_formula, 
                 molecular_weight, properties, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                mol.hash_id, mol.smiles, mol.name, mol.inchi, mol.inchi_key,
                mol.molecular_formula, mol.molecular_weight,
                json.dumps(mol.properties), json.dumps(mol.tags), mol.created_at
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to add molecule: {e}")
            raise
        
        return mol
    
    def add_molecules_bulk(
        self,
        smiles_list: List[str],
        names: List[str] = None,
        tags: List[str] = None,
    ) -> int:
        """複数分子を一括追加"""
        names = names or [None] * len(smiles_list)
        count = 0
        
        for smi, name in zip(smiles_list, names):
            try:
                self.add_molecule(smi, name=name, tags=tags)
                count += 1
            except Exception:
                continue
        
        return count
    
    def get_molecule(self, smiles: str) -> Optional[Molecule]:
        """SMILESで分子を取得"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM molecules WHERE smiles = ?', (smiles,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_molecule(row)
        return None
    
    def search_molecules(
        self,
        tags: List[str] = None,
        mw_range: Tuple[float, float] = None,
        name_pattern: str = None,
        limit: int = 100,
    ) -> List[Molecule]:
        """分子を検索"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM molecules WHERE 1=1"
        params = []
        
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')
        
        if mw_range:
            query += " AND molecular_weight BETWEEN ? AND ?"
            params.extend(mw_range)
        
        if name_pattern:
            query += " AND name LIKE ?"
            params.append(f"%{name_pattern}%")
        
        query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        return [self._row_to_molecule(row) for row in cursor.fetchall()]
    
    def get_all_molecules(self) -> List[Molecule]:
        """全分子を取得"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM molecules')
        return [self._row_to_molecule(row) for row in cursor.fetchall()]
    
    def _row_to_molecule(self, row) -> Molecule:
        """DBレコードをMoleculeに変換"""
        return Molecule(
            smiles=row[1],
            name=row[2],
            inchi=row[3],
            inchi_key=row[4],
            molecular_formula=row[5],
            molecular_weight=row[6],
            properties=json.loads(row[7]) if row[7] else {},
            tags=json.loads(row[8]) if row[8] else [],
            created_at=row[9],
        )
    
    # ========== プロジェクト操作 ==========
    
    def create_project(
        self,
        name: str,
        description: str = "",
        tags: List[str] = None,
    ) -> Project:
        """プロジェクトを作成"""
        import uuid
        project = Project(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            tags=tags or [],
        )
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO projects (id, name, description, tags, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (project.id, project.name, project.description,
              json.dumps(project.tags), project.created_at))
        self.conn.commit()
        
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """プロジェクトを取得"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()
        
        if row:
            return Project(
                id=row[0], name=row[1], description=row[2],
                tags=json.loads(row[3]) if row[3] else [],
                created_at=row[4],
            )
        return None
    
    def list_projects(self) -> List[Project]:
        """全プロジェクトを一覧"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM projects ORDER BY created_at DESC')
        return [
            Project(id=row[0], name=row[1], description=row[2],
                   tags=json.loads(row[3]) if row[3] else [],
                   created_at=row[4])
            for row in cursor.fetchall()
        ]
    
    # ========== 実験操作 ==========
    
    def create_experiment(
        self,
        name: str,
        project_id: str,
        model_type: str,
        target_property: str,
        molecules: List[str] = None,
    ) -> Experiment:
        """実験を作成"""
        import uuid
        exp = Experiment(
            id=str(uuid.uuid4())[:8],
            name=name,
            project_id=project_id,
            model_type=model_type,
            target_property=target_property,
            molecules=molecules or [],
        )
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiments 
            (id, name, project_id, model_type, target_property, molecules,
             results, metrics, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (exp.id, exp.name, exp.project_id, exp.model_type, exp.target_property,
              json.dumps(exp.molecules), json.dumps(exp.results),
              json.dumps(exp.metrics), exp.status, exp.created_at, exp.updated_at))
        self.conn.commit()
        
        return exp
    
    def update_experiment(
        self,
        experiment_id: str,
        results: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        status: str = None,
    ):
        """実験結果を更新"""
        cursor = self.conn.cursor()
        
        updates = ["updated_at = ?"]
        params = [datetime.now().isoformat()]
        
        if results:
            updates.append("results = ?")
            params.append(json.dumps(results))
        if metrics:
            updates.append("metrics = ?")
            params.append(json.dumps(metrics))
        if status:
            updates.append("status = ?")
            params.append(status)
        
        params.append(experiment_id)
        
        cursor.execute(
            f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
            params
        )
        self.conn.commit()
    
    def save_property_values(
        self,
        experiment_id: str,
        smiles_list: List[str],
        values: List[float],
        property_name: str,
        uncertainties: List[float] = None,
    ):
        """予測結果を保存"""
        cursor = self.conn.cursor()
        uncertainties = uncertainties or [None] * len(smiles_list)
        
        for smi, val, unc in zip(smiles_list, values, uncertainties):
            mol = self.get_molecule(smi)
            if not mol:
                mol = self.add_molecule(smi)
            
            cursor.execute('''
                INSERT INTO property_values 
                (molecule_id, experiment_id, property_name, value, uncertainty, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (mol.hash_id, experiment_id, property_name, val, unc,
                  datetime.now().isoformat()))
        
        self.conn.commit()
    
    # ========== エクスポート ==========
    
    def export_csv(self, filepath: str, include_properties: bool = True) -> None:
        """CSVにエクスポート"""
        molecules = self.get_all_molecules()
        
        data = []
        for mol in molecules:
            row = {
                'smiles': mol.smiles,
                'name': mol.name,
                'molecular_formula': mol.molecular_formula,
                'molecular_weight': mol.molecular_weight,
                'tags': ','.join(mol.tags),
            }
            if include_properties:
                row.update(mol.properties)
            data.append(row)
        
        pd.DataFrame(data).to_csv(filepath, index=False)
        logger.info(f"Exported {len(molecules)} molecules to {filepath}")
    
    def export_sdf(self, filepath: str) -> None:
        """SDFにエクスポート"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit is required for SDF export")
        
        molecules = self.get_all_molecules()
        
        writer = Chem.SDWriter(filepath)
        for mol in molecules:
            rdmol = Chem.MolFromSmiles(mol.smiles)
            if rdmol:
                rdmol.SetProp("_Name", mol.name or mol.smiles)
                for key, val in mol.properties.items():
                    rdmol.SetProp(key, str(val))
                writer.write(rdmol)
        
        writer.close()
        logger.info(f"Exported {len(molecules)} molecules to {filepath}")
    
    def export_json(self, filepath: str) -> None:
        """JSONにエクスポート"""
        molecules = self.get_all_molecules()
        
        data = {
            'molecules': [mol.to_dict() for mol in molecules],
            'exported_at': datetime.now().isoformat(),
            'count': len(molecules),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(molecules)} molecules to {filepath}")
    
    # ========== インポート ==========
    
    def import_csv(
        self,
        filepath: str,
        smiles_column: str = 'smiles',
        name_column: str = None,
        property_columns: List[str] = None,
    ) -> int:
        """CSVからインポート"""
        df = pd.read_csv(filepath)
        
        count = 0
        for _, row in df.iterrows():
            smiles = row[smiles_column]
            name = row.get(name_column) if name_column else None
            
            properties = {}
            if property_columns:
                for col in property_columns:
                    if col in row and pd.notna(row[col]):
                        properties[col] = row[col]
            
            try:
                self.add_molecule(smiles, name=name, properties=properties)
                count += 1
            except Exception:
                continue
        
        logger.info(f"Imported {count} molecules from {filepath}")
        return count
    
    def close(self):
        """データベース接続を閉じる"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_database(db_path: str = "molecules.db") -> MoleculeDatabase:
    """便利関数: データベースを取得"""
    return MoleculeDatabase(db_path)
