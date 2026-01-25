"""
モデル永続化モジュールのテスト

Implements: T-MODEL-001
カバレッジ対象: core/services/utils/model_persistence.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.services.utils.model_persistence import (
    ModelMetadata,
    ModelPersistence,
    load_model,
    save_model,
)


class SimpleModel:
    """テスト用pickle可能モデル"""
    def predict(self, X):
        return [1, 2, 3]


class CustomModel:
    """テスト用カスタムモデル"""
    def predict(self, X):
        return [1, 2, 3]


class TestModelMetadata:
    """ModelMetadata dataclassのテスト"""

    def test_to_dict(self):
        """辞書への変換"""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0",
            model_type="RandomForest",
            created_at="2024-01-01T00:00:00",
            metrics={"r2": 0.95, "rmse": 0.1},
            parameters={"n_estimators": 100},
            features=["feature1", "feature2"],
            target="target",
            description="Test model",
        )
        
        data = metadata.to_dict()
        assert data["name"] == "test_model"
        assert data["metrics"]["r2"] == 0.95
        assert data["features"] == ["feature1", "feature2"]

    def test_from_dict(self):
        """辞書からの変換"""
        data = {
            "name": "test",
            "version": "1.0",
            "model_type": "RF",
            "created_at": "2024-01-01",
            "metrics": {},
            "parameters": {},
            "features": [],
            "target": "y",
        }
        
        metadata = ModelMetadata.from_dict(data)
        assert metadata.name == "test"
        assert metadata.version == "1.0"


class TestModelPersistence:
    """ModelPersistenceのテスト"""

    @pytest.fixture
    def tmp_model_dir(self, tmp_path):
        """一時モデルディレクトリ"""
        return str(tmp_path / "test_models")

    @pytest.fixture
    def sample_model(self):
        """テスト用モデル（Pickle可能）"""
        return SimpleModel()

    def test_initialization(self, tmp_model_dir):
        """初期化"""
        mp = ModelPersistence(tmp_model_dir)
        assert mp.base_dir.exists()
        assert (mp.base_dir / "registry.json").exists()

    def test_save_model_basic(self, tmp_model_dir, sample_model):
        """基本的なモデル保存"""
        mp = ModelPersistence(tmp_model_dir)
        
        path = mp.save(
            sample_model,
            name="test_model",
            version="1.0",
            metrics={"r2": 0.95},
        )
        
        assert Path(path).exists()
        assert (Path(path) / "model.pkl").exists()
        assert (Path(path) / "metadata.json").exists()

    def test_save_model_with_metadata(self, tmp_model_dir, sample_model):
        """メタデータ付きモデル保存"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(
            sample_model,
            name="advanced_model",
            version="2.0",
            metrics={"r2": 0.98, "mse": 0.02},
            parameters={"n_estimators": 100, "max_depth": 10},
            features=["f1", "f2", "f3"],
            target="target_var",
            description="Advanced test model",
        )
        
        metadata = mp.get_metadata("advanced_model", "2.0")
        assert metadata.name == "advanced_model"
        assert metadata.metrics["r2"] == 0.98
        assert metadata.parameters["n_estimators"] == 100
        assert len(metadata.features) == 3

    def test_save_model_auto_version(self, tmp_model_dir, sample_model):
        """自動バージョン付与"""
        mp = ModelPersistence(tmp_model_dir)
        
        path = mp.save(sample_model, name="auto_version_model")
        
        # バージョンは自動生成される（YYYYMMDD_HHMMSS形式）
        assert Path(path).exists()
        versions = mp.list_versions("auto_version_model")
        assert len(versions) == 1
        assert len(versions[0]) == 15  # YYYYMMDD_HHMMSS

    def test_load_model(self, tmp_model_dir, sample_model):
        """モデル読み込み"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, name="loadable_model", version="1.0")
        loaded_model = mp.load("loadable_model", version="1.0")
        
        # モデルが復元される
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')

    def test_load_latest_version(self, tmp_model_dir, sample_model):
        """最新バージョンの読み込み"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, name="multi_version", version="1.0")
        mp.save(sample_model, name="multi_version", version="2.0")
        mp.save(sample_model, name="multi_version", version="3.0")
        
        # バージョン指定なしで最新を読み込み
        loaded = mp.load("multi_version")
        assert loaded is not None

    def test_load_nonexistent_model(self, tmp_model_dir):
        """存在しないモデルの読み込み"""
        mp = ModelPersistence(tmp_model_dir)
        
        with pytest.raises(ValueError, match="Model not found"):
            mp.load("nonexistent")

    def test_list_models(self, tmp_model_dir, sample_model):
        """モデル一覧"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, "model1", version="1.0")
        mp.save(sample_model, "model2", version="1.0")
        mp.save(sample_model, "model3", version="1.0")
        
        models = mp.list_models()
        assert len(models) == 3
        assert "model1" in models
        assert "model2" in models

    def test_list_versions(self, tmp_model_dir, sample_model):
        """バージョン一覧"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, "versioned", version="1.0")
        mp.save(sample_model, "versioned", version="1.1")
        mp.save(sample_model, "versioned", version="2.0")
        
        versions = mp.list_versions("versioned")
        assert len(versions) == 3
        assert "1.0" in versions
        assert "2.0" in versions

    def test_list_versions_nonexistent(self, tmp_model_dir):
        """存在しないモデルのバージョン一覧"""
        mp = ModelPersistence(tmp_model_dir)
        versions = mp.list_versions("nonexistent")
        assert versions == []

    def test_delete_specific_version(self, tmp_model_dir, sample_model):
        """特定バージョンの削除"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, "deletable", version="1.0")
        mp.save(sample_model, "deletable", version="2.0")
        
        mp.delete("deletable", version="1.0")
        
        versions = mp.list_versions("deletable")
        assert len(versions) == 1
        assert "2.0" in versions
        assert "1.0" not in versions

    def test_delete_all_versions(self, tmp_model_dir, sample_model):
        """全バージョンの削除"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, "fully_deletable", version="1.0")
        mp.save(sample_model, "fully_deletable", version="2.0")
        
        mp.delete("fully_deletable")
        
        models = mp.list_models()
        assert "fully_deletable" not in models

    def test_compare_versions(self, tmp_model_dir, sample_model):
        """バージョン間のメトリクス比較"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, "comparable", version="1.0", metrics={"r2": 0.90})
        mp.save(sample_model, "comparable", version="2.0", metrics={"r2": 0.95})
        mp.save(sample_model, "comparable", version="3.0", metrics={"r2": 0.98})
        
        comparison = mp.compare_versions("comparable")
        
        assert len(comparison) == 3
        assert comparison["1.0"]["r2"] == 0.90
        assert comparison["3.0"]["r2"] == 0.98

    def test_compare_specific_versions(self, tmp_model_dir, sample_model):
        """特定バージョンの比較"""
        mp = ModelPersistence(tmp_model_dir)
        
        mp.save(sample_model, "selective", version="1.0", metrics={"r2": 0.90})
        mp.save(sample_model, "selective", version="2.0", metrics={"r2": 0.95})
        mp.save(sample_model, "selective", version="3.0", metrics={"r2": 0.98})
        
        comparison = mp.compare_versions("selective", versions=["1.0", "3.0"])
        
        assert len(comparison) == 2
        assert "2.0" not in comparison

    def test_registry_persistence(self, tmp_model_dir, sample_model):
        """レジストリの永続化"""
        mp1 = ModelPersistence(tmp_model_dir)
        mp1.save(sample_model, "persistent", version="1.0")
        
        # 新しいインスタンスでもレジストリが読める
        mp2 = ModelPersistence(tmp_model_dir)
        models = mp2.list_models()
        assert "persistent" in models


class TestConvenienceFunctions:
    """便利関数のテスト"""

    def test_save_model_function(self, tmp_path):
        """save_model便利関数"""
        # SimpleModel is global now
        model = SimpleModel()
        base_dir = str(tmp_path / "models")
        
        path = save_model(model, "convenience_test", base_dir=base_dir)
        assert Path(path).exists()

    def test_load_model_function(self, tmp_path):
        """load_model便利関数"""
        # SimpleModel is global now
        model = SimpleModel()
        base_dir = str(tmp_path / "models")
        
        save_model(model, "loadable", base_dir=base_dir)
        loaded = load_model("loadable", base_dir=base_dir)
        
        assert loaded is not None


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_metrics(self, tmp_path):
        """メトリクスなしで保存"""
        mp = ModelPersistence(str(tmp_path))
        
        # SimpleModel is global now
        model = SimpleModel()
        
        mp.save(model, "no_metrics", version="1.0")
        metadata = mp.get_metadata("no_metrics", "1.0")
        
        assert metadata.metrics == {}

    def test_complex_model_type(self, tmp_path):
        """複雑なモデルタイプ"""
        mp = ModelPersistence(str(tmp_path))
        
        # CustomModel is global now
        model = CustomModel()
        mp.save(model, "custom", version="1.0")
        
        loaded = mp.load("custom", "1.0")
        assert isinstance(loaded, CustomModel)
