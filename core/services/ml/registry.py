"""
モデルレジストリ（MLflow Model Registry inspired）

Implements: F-REGISTRY-001
設計思想:
- モデルバージョン管理
- ステージ管理
- メタデータ追跡
"""

from __future__ import annotations

import logging
import os
import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """モデルバージョン"""
    version: int
    name: str
    stage: str = "development"  # development, staging, production, archived
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""


class ModelRegistry:
    """
    モデルレジストリ（MLflow inspired）
    
    Features:
    - バージョン管理
    - ステージ遷移
    - モデル比較
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("my_model", model, metrics={"r2": 0.95})
        >>> prod_model = registry.load("my_model", stage="production")
    """
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.registry_dir / "registry.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """メタデータ読み込み"""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"models": {}}
    
    def _save_metadata(self) -> None:
        """メタデータ保存"""
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
    
    def register(
        self,
        name: str,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> ModelVersion:
        """モデル登録"""
        if name not in self._metadata["models"]:
            self._metadata["models"][name] = {"versions": []}
        
        versions = self._metadata["models"][name]["versions"]
        new_version = len(versions) + 1
        
        # バージョンディレクトリ作成
        version_dir = self.registry_dir / name / f"v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル保存
        model_path = version_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # メタデータ作成
        version_info = ModelVersion(
            version=new_version,
            name=name,
            stage="development",
            metrics=metrics or {},
            params=params or {},
            description=description,
        )
        
        versions.append(asdict(version_info))
        self._save_metadata()
        
        logger.info(f"Registered {name} v{new_version}")
        return version_info
    
    def load(
        self,
        name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """モデル読み込み"""
        if name not in self._metadata["models"]:
            raise ValueError(f"Model not found: {name}")
        
        versions = self._metadata["models"][name]["versions"]
        
        if stage:
            matching = [v for v in versions if v["stage"] == stage]
            if not matching:
                raise ValueError(f"No {stage} version for {name}")
            version_info = matching[-1]
        elif version:
            version_info = versions[version - 1]
        else:
            version_info = versions[-1]
        
        model_path = self.registry_dir / name / f"v{version_info['version']}" / "model.joblib"
        return joblib.load(model_path)
    
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: str,
    ) -> None:
        """ステージ遷移"""
        if name not in self._metadata["models"]:
            raise ValueError(f"Model not found: {name}")
        
        versions = self._metadata["models"][name]["versions"]
        
        if version < 1 or version > len(versions):
            raise ValueError(f"Invalid version: {version}")
        
        versions[version - 1]["stage"] = stage
        self._save_metadata()
        
        logger.info(f"Transitioned {name} v{version} to {stage}")
    
    def list_models(self) -> List[str]:
        """モデル一覧"""
        return list(self._metadata["models"].keys())
    
    def list_versions(self, name: str) -> List[ModelVersion]:
        """バージョン一覧"""
        if name not in self._metadata["models"]:
            return []
        
        return [ModelVersion(**v) for v in self._metadata["models"][name]["versions"]]
    
    def compare(self, name: str, metric: str = "r2") -> List[Dict[str, Any]]:
        """バージョン比較"""
        versions = self.list_versions(name)
        return sorted(
            [{"version": v.version, "stage": v.stage, metric: v.metrics.get(metric, 0)}
             for v in versions],
            key=lambda x: x[metric],
            reverse=True,
        )
