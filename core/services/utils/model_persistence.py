"""
モデル永続化とバージョン管理

Implements: F-MODEL-001
設計思想:
- 学習済みモデルの保存/読み込み
- バージョン管理とメタデータ
- 再現性の確保

機能:
- Pickle/Joblib保存
- モデルレジストリ
- メタデータ管理
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
import pickle

import numpy as np

logger = logging.getLogger(__name__)

# Joblib（オプショナル）
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


@dataclass
class ModelMetadata:
    """モデルメタデータ"""
    name: str
    version: str
    model_type: str
    created_at: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    features: List[str]
    target: str
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        return cls(**data)


class ModelPersistence:
    """
    モデル永続化
    
    Usage:
        mp = ModelPersistence()
        
        # 保存
        mp.save(model, "my_model", version="1.0", metrics={'r2': 0.95})
        
        # 読み込み
        loaded_model = mp.load("my_model")
    """
    
    def __init__(self, base_dir: str = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry_path = self.base_dir / "registry.json"
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """レジストリをロード"""
        if self._registry_path.exists():
            with open(self._registry_path, 'r') as f:
                return json.load(f)
        return {"models": {}}
    
    def _save_registry(self) -> None:
        """レジストリを保存"""
        with open(self._registry_path, 'w') as f:
            json.dump(self._registry, f, indent=2)
    
    def save(
        self,
        model: Any,
        name: str,
        version: str = None,
        metrics: Dict[str, float] = None,
        parameters: Dict[str, Any] = None,
        features: List[str] = None,
        target: str = None,
        description: str = "",
    ) -> str:
        """
        モデルを保存
        
        Args:
            model: 学習済みモデル
            name: モデル名
            version: バージョン（Noneで自動）
            metrics: 評価メトリクス
            parameters: ハイパーパラメータ
            features: 特徴量リスト
            target: ターゲット名
            description: 説明
            
        Returns:
            モデルパス
        """
        # バージョン自動生成
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ディレクトリ作成
        model_dir = self.base_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # メタデータ
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=type(model).__name__,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            parameters=parameters or {},
            features=features or [],
            target=target or "",
            description=description,
        )
        
        # メタデータ保存
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # モデル保存
        model_path = model_dir / "model.pkl"
        if JOBLIB_AVAILABLE:
            joblib.dump(model, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # レジストリ更新
        if name not in self._registry["models"]:
            self._registry["models"][name] = {"versions": []}
        
        self._registry["models"][name]["versions"].append({
            "version": version,
            "path": str(model_dir),
            "metrics": metrics or {},
            "created_at": metadata.created_at,
        })
        self._registry["models"][name]["latest"] = version
        
        self._save_registry()
        
        logger.info(f"Saved model: {name} v{version}")
        
        return str(model_dir)
    
    def load(
        self,
        name: str,
        version: str = None,
    ) -> Any:
        """
        モデルを読み込み
        
        Args:
            name: モデル名
            version: バージョン（Noneで最新）
            
        Returns:
            モデル
        """
        if name not in self._registry["models"]:
            raise ValueError(f"Model not found: {name}")
        
        if version is None:
            version = self._registry["models"][name]["latest"]
        
        model_dir = self.base_dir / name / version
        model_path = model_dir / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if JOBLIB_AVAILABLE:
            model = joblib.load(model_path)
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        logger.info(f"Loaded model: {name} v{version}")
        
        return model
    
    def get_metadata(
        self,
        name: str,
        version: str = None,
    ) -> ModelMetadata:
        """メタデータを取得"""
        if version is None:
            version = self._registry["models"][name]["latest"]
        
        model_dir = self.base_dir / name / version
        
        with open(model_dir / "metadata.json", 'r') as f:
            data = json.load(f)
        
        return ModelMetadata.from_dict(data)
    
    def list_models(self) -> List[str]:
        """モデル一覧"""
        return list(self._registry["models"].keys())
    
    def list_versions(self, name: str) -> List[str]:
        """バージョン一覧"""
        if name not in self._registry["models"]:
            return []
        return [v["version"] for v in self._registry["models"][name]["versions"]]
    
    def delete(self, name: str, version: str = None) -> None:
        """モデルを削除"""
        if name not in self._registry["models"]:
            return
        
        import shutil
        
        if version:
            model_dir = self.base_dir / name / version
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # レジストリ更新
            self._registry["models"][name]["versions"] = [
                v for v in self._registry["models"][name]["versions"]
                if v["version"] != version
            ]
        else:
            # 全バージョン削除
            model_dir = self.base_dir / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            del self._registry["models"][name]
        
        self._save_registry()
    
    def compare_versions(
        self,
        name: str,
        versions: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """バージョン間のメトリクス比較"""
        if name not in self._registry["models"]:
            return {}
        
        all_versions = self._registry["models"][name]["versions"]
        
        if versions:
            all_versions = [v for v in all_versions if v["version"] in versions]
        
        return {
            v["version"]: v["metrics"]
            for v in all_versions
        }


def save_model(
    model: Any,
    name: str,
    metrics: Dict[str, float] = None,
    base_dir: str = "models",
) -> str:
    """便利関数: モデル保存"""
    mp = ModelPersistence(base_dir)
    return mp.save(model, name, metrics=metrics)


def load_model(
    name: str,
    version: str = None,
    base_dir: str = "models",
) -> Any:
    """便利関数: モデル読み込み"""
    mp = ModelPersistence(base_dir)
    return mp.load(name, version)
