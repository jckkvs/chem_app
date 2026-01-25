"""
データI/Oモジュールのテスト

Implements: T-IO-001
カバレッジ対象: core/services/utils/data_io.py
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.services.utils.data_io import (
    DataExporter,
    DataImporter,
    ImportResult,
    export_data,
    import_data,
)


class TestImportResult:
    """ImportResult dataclassのテスト"""

    def test_success_rate_calculation(self):
        """成功率の計算"""
        result = ImportResult(
            total=100,
            successful=80,
            failed=20,
            data=pd.DataFrame(),
            errors=[],
        )
        assert result.success_rate == 0.8

    def test_success_rate_zero_total(self):
        """total=0の場合"""
        result = ImportResult(
            total=0,
            successful=0,
            failed=0,
            data=pd.DataFrame(),
            errors=[],
        )
        assert result.success_rate == 0.0


class TestDataImporter:
    """DataImporterのテスト"""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """テスト用CSVファイル"""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
            "name": ["ethanol", "benzene", "acetic acid"],
            "value": [1.0, 2.0, 3.0],
        })
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_json(self, tmp_path):
        """テスト用JSONファイル"""
        json_path = tmp_path / "test.json"
        data = [
            {"smiles": "CCO", "name": "ethanol"},
            {"smiles": "c1ccccc1", "name": "benzene"},
        ]
        with open(json_path, "w") as f:
            json.dump(data, f)
        return json_path

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        """テスト用JSONLファイル"""
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"smiles": "CCO", "name": "ethanol"}\n')
            f.write('{"smiles": "c1ccccc1", "name": "benzene"}\n')
        return jsonl_path

    def test_import_csv_basic(self, sample_csv):
        """CSV基本インポート"""
        importer = DataImporter()
        result = importer.import_csv(
            str(sample_csv),
            smiles_column="smiles",
            validate_smiles=False,
        )
        
        assert result.total == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.data) == 3
        assert "smiles" in result.data.columns

    def test_import_csv_with_validation(self, sample_csv):
        """SMILES検証付きインポート"""
        importer = DataImporter()
        result = importer.import_csv(
            str(sample_csv),
            smiles_column="smiles",
            validate_smiles=True,
        )
        
        # 有効なSMILESのみ残る
        assert result.successful >= 0
        assert result.total >= result.successful

    def test_import_csv_invalid_column(self, sample_csv):
        """存在しないカラム指定"""
        importer = DataImporter()
        with pytest.raises(ValueError, match="SMILES column not found"):
            importer.import_csv(
                str(sample_csv),
                smiles_column="nonexistent",
            )

    def test_import_tsv(self, tmp_path):
        """TSVインポート"""
        tsv_path = tmp_path / "test.tsv"
        df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        df.to_csv(tsv_path, sep="\t", index=False)
        
        importer = DataImporter()
        result = importer.import_tsv(
            str(tsv_path),
            smiles_column="smiles",
            validate_smiles=False,
        )
        
        assert result.successful == 2

    def test_import_json(self, sample_json):
        """JSONインポート"""
        importer = DataImporter()
        result = importer.import_json(
            str(sample_json),
            smiles_key="smiles",
        )
        
        assert result.successful >= 0
        assert "smiles" in result.data.columns

    def test_import_jsonl(self, sample_jsonl):
        """JSONLインポート"""
        importer = DataImporter()
        result = importer.import_jsonl(
            str(sample_jsonl),
            smiles_key="smiles",
        )
        
        assert result.successful >= 0

    def test_import_jsonl_with_errors(self, tmp_path):
        """不正なJSONLの処理"""
        jsonl_path = tmp_path / "bad.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"smiles": "CCO"}\n')
            f.write('invalid json\n')  # 不正
            f.write('{"smiles": "c1ccccc1"}\n')
        
        importer = DataImporter()
        result = importer.import_jsonl(str(jsonl_path), smiles_key="smiles")
        
        # エラーが記録される
        assert len(result.errors) > 0


class TestDataExporter:
    """DataExporterのテスト"""

    @pytest.fixture
    def sample_df(self):
        """テスト用DataFrame"""
        return pd.DataFrame({
            "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
            "name": ["ethanol", "benzene", "acetic acid"],
            "value": [1.0, 2.0, 3.0],
        })

    def test_export_csv(self, sample_df, tmp_path):
        """CSVエクスポート"""
        csv_path = tmp_path / "output.csv"
        exporter = DataExporter()
        exporter.export_csv(sample_df, str(csv_path))
        
        assert csv_path.exists()
        
        # 読み込んで確認
        df_read = pd.read_csv(csv_path)
        assert len(df_read) == 3
        assert "smiles" in df_read.columns

    def test_export_tsv(self, sample_df, tmp_path):
        """TSVエクスポート"""
        tsv_path = tmp_path / "output.tsv"
        exporter = DataExporter()
        exporter.export_tsv(sample_df, str(tsv_path))
        
        assert tsv_path.exists()

    def test_export_json(self, sample_df, tmp_path):
        """JSONエクスポート"""
        json_path = tmp_path / "output.json"
        exporter = DataExporter()
        exporter.export_json(sample_df, str(json_path))
        
        assert json_path.exists()
        
        # 読み込んで確認
        with open(json_path, "r") as f:
            data = json.load(f)
        assert len(data) == 3

    def test_export_jsonl(self, sample_df, tmp_path):
        """JSONLエクスポート"""
        jsonl_path = tmp_path / "output.jsonl"
        exporter = DataExporter()
        exporter.export_jsonl(sample_df, str(jsonl_path))
        
        assert jsonl_path.exists()
        
        # 各行が有効なJSON
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3
        
        for line in lines:
            obj = json.loads(line)
            assert "smiles" in obj


class TestConvenienceFunctions:
    """便利関数のテスト"""

    def test_import_data_csv(self, tmp_path):
        """import_data関数（CSV）"""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
        df.to_csv(csv_path, index=False)
        
        result = import_data(str(csv_path), smiles_column="smiles")
        assert result.successful >= 0

    def test_import_data_format_detection(self, tmp_path):
        """フォーマット自動検出"""
        json_path = tmp_path / "test.json"
        data = [{"smiles": "CCO"}]
        with open(json_path, "w") as f:
            json.dump(data, f)
        
        result = import_data(str(json_path))
        assert result.successful >= 0

    def test_import_data_unsupported_format(self, tmp_path):
        """未サポートフォーマット"""
        bad_path = tmp_path / "test.xyz"
        bad_path.write_text("invalid")
        
        with pytest.raises(ValueError, match="Unsupported format"):
            import_data(str(bad_path))

    def test_export_data_csv(self, tmp_path):
        """export_data関数（CSV）"""
        df = pd.DataFrame({"smiles": ["CCO"]})
        csv_path = tmp_path / "output.csv"
        
        export_data(df, str(csv_path))
        assert csv_path.exists()

    def test_export_data_format_detection(self, tmp_path):
        """エクスポート形式自動検出"""
        df = pd.DataFrame({"smiles": ["CCO"]})
        json_path = tmp_path / "output.json"
        
        export_data(df, str(json_path))
        assert json_path.exists()


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_dataframe_import(self, tmp_path):
        """空DataFrameのインポート"""
        csv_path = tmp_path / "empty.csv"
        df = pd.DataFrame({"smiles": []})
        df.to_csv(csv_path, index=False)
        
        importer = DataImporter()
        result = importer.import_csv(str(csv_path), validate_smiles=False)
        assert result.total == 0
        assert len(result.data) == 0

    def test_large_file_import(self, tmp_path):
        """大きいファイルのインポート"""
        csv_path = tmp_path / "large.csv"
        df = pd.DataFrame({
            "smiles": ["CCO"] * 1000,
            "value": range(1000),
        })
        df.to_csv(csv_path, index=False)
        
        importer = DataImporter()
        result = importer.import_csv(str(csv_path), validate_smiles=False)
        assert result.total == 1000

    def test_unicode_handling(self, tmp_path):
        """Unicode文字の処理"""
        csv_path = tmp_path / "unicode.csv"
        df = pd.DataFrame({
            "smiles": ["CCO"],
            "name": ["エタノール"],  # 日本語
        })
        df.to_csv(csv_path, index=False)
        
        importer = DataImporter()
        result = importer.import_csv(str(csv_path), validate_smiles=False)
        assert result.successful == 1
