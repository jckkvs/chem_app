"""
MoleculeDatabaseのテスト

カバレッジ目標: 80%以上（249 stmts）
"""
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from core.services.data.molecule_database import (
    Molecule,
    Experiment,
    Project,
    MoleculeDatabase,
    get_database,
)


class TestMoleculeDataclass:
    """Moleculeデータクラステスト"""
    
    def test_molecule_creation_minimal(self):
        """最小限のMolecule作成"""
        mol = Molecule(smiles="CCO")
        
        assert mol.smiles == "CCO"
        assert mol.name is None
        assert mol.created_at is not None
        assert isinstance(mol.properties, dict)
        assert isinstance(mol.tags, list)
    
    def test_molecule_creation_full(self):
        """完全なMolecule作成"""
        mol = Molecule(
            smiles="CCO",
            name="Ethanol",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            molecular_formula="C2H6O",
            molecular_weight=46.07,
            properties={"solubility": 100},
            tags=["solvent", "alcohol"]
        )
        
        assert mol.name == "Ethanol"
        assert mol.properties["solubility"] == 100
        assert "solvent" in mol.tags
    
    def test_molecule_hash_id(self):
        """hash_id生成テスト"""
        mol1 = Molecule(smiles="CCO")
        mol2 = Molecule(smiles="CCO")
        mol3 = Molecule(smiles="CC")
        
        # 同じSMILESは同じhash_id
        assert mol1.hash_id == mol2.hash_id
        # 異なるSMILESは異なるhash_id
        assert mol1.hash_id != mol3.hash_id
        # hash_idは16文字
        assert len(mol1.hash_id) == 16
    
    def test_molecule_to_dict(self):
        """辞書変換テスト"""
        mol = Molecule(smiles="CCO", name="Ethanol", tags=["test"])
        d = mol.to_dict()
        
        assert isinstance(d, dict)
        assert d["smiles"] == "CCO"
        assert d["name"] == "Ethanol"
        assert "test" in d["tags"]
    
    @patch('rdkit.Chem.MolFromSmiles')
    @patch('rdkit.Chem.rdMolDescriptors.CalcMolFormula')
    @patch('rdkit.Chem.Descriptors.MolWt')
    @patch('rdkit.Chem.inchi.MolToInchi')
    @patch('rdkit.Chem.inchi.MolToInchiKey')
    def test_from_smiles_with_rdkit(self, mock_inchi_key, mock_inchi,
                                     mock_molwt, mock_formula, mock_mol):
        """from_smilesテスト（RDKit成功）"""
        mock_rdmol = MagicMock()
        mock_mol.return_value = mock_rdmol
        mock_formula.return_value = "C2H6O"
        mock_molwt.return_value = 46.07
        mock_inchi.return_value = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        mock_inchi_key.return_value = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        
        mol = Molecule.from_smiles("CCO", "Ethanol")
        
        assert mol.smiles == "CCO"
        assert mol.name == "Ethanol"
        assert mol.molecular_formula == "C2H6O"
        assert mol.molecular_weight == 46.07
    
    @patch('rdkit.Chem.MolFromSmiles')
    def test_from_smiles_rdkit_failure(self, mock_mol):
        """from_smilesテスト（RDKit失敗）"""
        mock_mol.side_effect = Exception("RDKit error")
        
        # エラーでも分子は作成される
        mol = Molecule.from_smiles("INVALID", "Test")
        assert mol.smiles == "INVALID"
        assert mol.name == "Test"
        # RDKit情報はNone
        assert mol.molecular_formula is None


class TestExperimentProject:
    """Experiment/Projectデータクラステスト"""
    
    def test_experiment_creation(self):
        """Experiment作成"""
        exp = Experiment(
            id="test001",
            name="Test Experiment",
            project_id="proj001",
            model_type="RandomForest",
            target_property="solubility"
        )
        
        assert exp.id == "test001"
        assert exp.status == "created"
        assert exp.created_at is not None
        assert exp.updated_at is not None
    
    def test_project_creation(self):
        """Project作成"""
        proj = Project(
            id="proj001",
            name="Test Project",
            description="Test description",
            tags=["test", "ml"]
        )
        
        assert proj.id == "proj001"
        assert proj.name == "Test Project"
        assert "test" in proj.tags
        assert proj.created_at is not None


class TestMoleculeDatabaseBasic:
    """MoleculeDatabase基本機能テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
    
    def teardown_method(self):
        self.db.close()
    
    def test_init_memory(self):
        """インメモリDB初期化"""
        db = MoleculeDatabase(":memory:")
        assert db.conn is not None
        db.close()
    
    def test_init_file(self):
        """ファイルDB初期化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = MoleculeDatabase(db_path)
            assert db.db_path == Path(db_path)
            assert os.path.exists(db_path)
            db.close()
    
    def test_context_manager(self):
        """コンテキストマネージャー"""
        with MoleculeDatabase(":memory:") as db:
            assert db.conn is not None
        # 自動的にcloseされる


class TestMoleculeDatabaseCRUD:
    """CRUD操作テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
    
    def teardown_method(self):
        self.db.close()
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_add_molecule(self, mock_from_smiles):
        """分子追加テスト"""
        mock_mol = Molecule(smiles="CCO", name="Ethanol")
        mock_from_smiles.return_value = mock_mol
        
        mol = self.db.add_molecule("CCO", name="Ethanol", tags=["test"])
        
        assert mol.smiles == "CCO"
        assert "test" in mol.tags
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_add_molecules_bulk(self, mock_from_smiles):
        """一括追加テスト"""
        def create_mol(smi, name):
            return Molecule(smiles=smi, name=name)
        
        mock_from_smiles.side_effect = lambda s, n: create_mol(s, n)
        
        count = self.db.add_molecules_bulk(
            ["CCO", "CC", "CCC"],
            names=["Ethanol", "Methane", "Propane"]
        )
        
        assert count == 3
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_get_molecule(self, mock_from_smiles):
        """分子取得テスト"""
        mock_mol = Molecule(smiles="CCO")
        mock_from_smiles.return_value = mock_mol
        
        self.db.add_molecule("CCO")
        mol = self.db.get_molecule("CCO")
        
        assert mol is not None
        assert mol.smiles == "CCO"
    
    def test_get_molecule_not_found(self):
        """存在しない分子"""
        mol = self.db.get_molecule("NOTEXIST")
        assert mol is None
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_get_all_molecules(self, mock_from_smiles):
        """全分子取得"""
        mock_from_smiles.side_effect = [
            Molecule(smiles="CCO"),
            Molecule(smiles="CC"),
        ]
        
        self.db.add_molecule("CCO")
        self.db.add_molecule("CC")
        
        mols = self.db.get_all_molecules()
        assert len(mols) == 2


class TestMoleculeDatabaseSearch:
    """検索機能テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
    
    def teardown_method(self):
        self.db.close()
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_search_by_tags(self, mock_from_smiles):
        """タグ検索"""
        mock_from_smiles.side_effect = [
            Molecule(smiles="CCO", tags=["alcohol"]),
            Molecule(smiles="CC", tags=["alkane"]),
        ]
        
        self.db.add_molecule("CCO", tags=["alcohol"])
        self.db.add_molecule("CC", tags=["alkane"])
        
        results = self.db.search_molecules(tags=["alcohol"])
        assert len(results) >= 1
        assert any(mol.smiles == "CCO" for mol in results)
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_search_by_mw_range(self, mock_from_smiles):
        """分子量範囲検索"""
        mol1 = Molecule(smiles="CCO", molecular_weight=46.0)
        mol2 = Molecule(smiles="CCCC", molecular_weight=58.0)
        mock_from_smiles.side_effect = [mol1, mol2]
        
        self.db.add_molecule("CCO")
        self.db.add_molecule("CCCC")
        
        results = self.db.search_molecules(mw_range=(40, 50))
        assert len(results) >= 1
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_search_by_name_pattern(self, mock_from_smiles):
        """名前パターン検索"""
        mock_from_smiles.return_value = Molecule(smiles="CCO", name="Ethanol")
        
        self.db.add_molecule("CCO", name="Ethanol")
        
        results = self.db.search_molecules(name_pattern="Eth")
        assert len(results) >= 1


class TestMoleculeDatabaseProject:
    """プロジェクト操作テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
    
    def teardown_method(self):
        self.db.close()
    
    def test_create_project(self):
        """プロジェクト作成"""
        proj = self.db.create_project("Test Project", "Description", tags=["ml"])
        
        assert proj.id is not None
        assert proj.name == "Test Project"
        assert "ml" in proj.tags
    
    def test_get_project(self):
        """プロジェクト取得"""
        proj1 = self.db.create_project("Test")
        proj2 = self.db.get_project(proj1.id)
        
        assert proj2 is not None
        assert proj2.id == proj1.id
        assert proj2.name == "Test"
    
    def test_get_project_not_found(self):
        """存在しないプロジェクト"""
        proj = self.db.get_project("nonexistent")
        assert proj is None
    
    def test_list_projects(self):
        """プロジェクト一覧"""
        self.db.create_project("Project 1")
        self.db.create_project("Project 2")
        
        projects = self.db.list_projects()
        assert len(projects) == 2


class TestMoleculeDatabaseExperiment:
    """実験操作テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
        self.project = self.db.create_project("Test Project")
    
    def teardown_method(self):
        self.db.close()
    
    def test_create_experiment(self):
        """実験作成"""
        exp = self.db.create_experiment(
            name="Test Exp",
            project_id=self.project.id,
            model_type="RF",
            target_property="solubility",
            molecules=["CCO", "CC"]
        )
        
        assert exp.id is not None
        assert exp.status == "created"
        assert len(exp.molecules) == 2
    
    def test_update_experiment(self):
        """実験更新"""
        exp = self.db.create_experiment(
            "Test", self.project.id, "RF", "sol"
        )
        
        self.db.update_experiment(
            exp.id,
            results={"rmse": 0.5},
            metrics={"r2": 0.95},
            status="completed"
        )
        
        # 更新が成功（エラーがないことを確認）
        assert True
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_save_property_values(self, mock_from_smiles):
        """プロパティ値保存"""
        mock_from_smiles.side_effect = [
            Molecule(smiles="CCO"),
            Molecule(smiles="CC"),
        ]
        
        exp = self.db.create_experiment(
            "Test", self.project.id, "RF", "sol"
        )
        
        self.db.save_property_values(
            exp.id,
            smiles_list=["CCO", "CC"],
            values=[1.5, 2.3],
            property_name="solubility",
            uncertainties=[0.1, 0.2]
        )
        
        # 保存が成功（エラーがないことを確認）
        assert True


class TestMoleculeDatabaseExport:
    """エクスポート機能テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
        self.tempdir = tempfile.mkdtemp()
    
    def teardown_method(self):
        self.db.close()
        import shutil
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_export_csv(self, mock_from_smiles):
        """CSVエクスポート"""
        mock_from_smiles.return_value = Molecule(
            smiles="CCO",
            name="Ethanol",
            molecular_formula="C2H6O",
            molecular_weight=46.07,
            tags=["test"]
        )
        
        self.db.add_molecule("CCO", name="Ethanol")
        
        csv_path = os.path.join(self.tempdir, "export.csv")
        self.db.export_csv(csv_path)
        
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["smiles"] == "CCO"
    
    @patch('rdkit.Chem.SDWriter')
    @patch('rdkit.Chem.MolFromSmiles')
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_export_sdf(self, mock_from_smiles, mock_mol_from_smiles, mock_writer):
        """SDFエクスポート"""
        mock_mol = Molecule(smiles="CCO", name="Ethanol", properties={"prop1": 1.0})
        mock_from_smiles.return_value = mock_mol
        
        mock_rdmol = MagicMock()
        mock_mol_from_smiles.return_value = mock_rdmol
        
        mock_writer_instance = MagicMock()
        mock_writer.return_value = mock_writer_instance
        
        self.db.add_molecule("CCO", name="Ethanol")
        
        sdf_path = os.path.join(self.tempdir, "export.sdf")
        self.db.export_sdf(sdf_path)
        
        mock_writer.assert_called_once()
        mock_writer_instance.write.assert_called()
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_export_json(self, mock_from_smiles):
        """JSONエクスポート"""
        mock_from_smiles.return_value = Molecule(smiles="CCO", name="Ethanol")
        
        self.db.add_molecule("CCO", name="Ethanol")
        
        json_path = os.path.join(self.tempdir, "export.json")
        self.db.export_json(json_path)
        
        assert os.path.exists(json_path)
        with open(json_path) as f:
            data = json.load(f)
        assert data["count"] == 1
        assert data["molecules"][0]["smiles"] == "CCO"


class TestMoleculeDatabaseImport:
    """インポート機能テスト"""
    
    def setup_method(self):
        self.db = MoleculeDatabase(":memory:")
        self.tempdir = tempfile.mkdtemp()
    
    def teardown_method(self):
        self.db.close()
        import shutil
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
    
    @patch('core.services.data.molecule_database.Molecule.from_smiles')
    def test_import_csv(self, mock_from_smiles):
        """CSVインポート"""
        # テストCSV作成
        csv_path = os.path.join(self.tempdir, "import.csv")
        df = pd.DataFrame({
            'smiles': ['CCO', 'CC', 'CCC'],
            'name': ['Ethanol', 'Methane', 'Propane'],
            'prop1': [1.0, 2.0, 3.0]
        })
        df.to_csv(csv_path, index=False)
        
        mock_from_smiles.side_effect = [
            Molecule(smiles="CCO", name="Ethanol"),
            Molecule(smiles="CC", name="Methane"),
            Molecule(smiles="CCC", name="Propane"),
        ]
        
        count = self.db.import_csv(
            csv_path,
            smiles_column='smiles',
            name_column='name',
            property_columns=['prop1']
        )
        
        assert count == 3


class TestUtilityFunctions:
    """ユーティリティ関数テスト"""
    
    def test_get_database(self):
        """get_database関数"""
        db = get_database(":memory:")
        assert isinstance(db, MoleculeDatabase)
        db.close()
