"""
フィーチャーストア（Feast inspired）

Implements: F-FEATURESTORE-001
設計思想:
- 特徴量の再利用
- バージョン管理
- オンライン/オフラインストア
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """特徴量定義"""
    name: str
    dtype: str
    description: str = ""
    source: str = ""
    created_at: str = ""
    tags: List[str] = None


class FeatureStore:
    """
    フィーチャーストア（Feast inspired）
    
    Features:
    - 特徴量登録/取得
    - バージョン管理
    - メタデータ追跡
    
    Example:
        >>> store = FeatureStore()
        >>> store.register_features("mol_features", df, definitions)
        >>> features = store.get_features("mol_features", ["MW", "LogP"])
    """
    
    def __init__(self, store_dir: str = "feature_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.store_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"feature_groups": {}}
    
    def _save_metadata(self) -> None:
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
    
    def register_features(
        self,
        group_name: str,
        df: pd.DataFrame,
        definitions: Optional[List[FeatureDefinition]] = None,
        entity_column: str = "id",
    ) -> None:
        """特徴量グループを登録"""
        group_dir = self.store_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # データ保存
        df.to_parquet(group_dir / "features.parquet", index=False)
        
        # 定義生成（なければ自動）
        if definitions is None:
            definitions = [
                FeatureDefinition(
                    name=col,
                    dtype=str(df[col].dtype),
                    created_at=datetime.now().isoformat(),
                )
                for col in df.columns if col != entity_column
            ]
        
        # メタデータ保存
        self._metadata["feature_groups"][group_name] = {
            "entity_column": entity_column,
            "features": [asdict(d) for d in definitions],
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,
            "updated_at": datetime.now().isoformat(),
        }
        self._save_metadata()
        
        logger.info(f"Registered {len(df.columns)-1} features in {group_name}")
    
    def get_features(
        self,
        group_name: str,
        feature_names: Optional[List[str]] = None,
        entity_ids: Optional[List[Any]] = None,
    ) -> pd.DataFrame:
        """特徴量を取得"""
        if group_name not in self._metadata["feature_groups"]:
            raise ValueError(f"Feature group not found: {group_name}")
        
        group_dir = self.store_dir / group_name
        df = pd.read_parquet(group_dir / "features.parquet")
        
        entity_column = self._metadata["feature_groups"][group_name]["entity_column"]
        
        # エンティティフィルタ
        if entity_ids is not None:
            df = df[df[entity_column].isin(entity_ids)]
        
        # 特徴量フィルタ
        if feature_names is not None:
            cols = [entity_column] + feature_names
            df = df[[c for c in cols if c in df.columns]]
        
        return df
    
    def list_feature_groups(self) -> List[str]:
        """グループ一覧"""
        return list(self._metadata["feature_groups"].keys())
    
    def get_feature_definitions(self, group_name: str) -> List[FeatureDefinition]:
        """定義を取得"""
        if group_name not in self._metadata["feature_groups"]:
            return []
        
        return [
            FeatureDefinition(**f)
            for f in self._metadata["feature_groups"][group_name]["features"]
        ]
    
    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """特徴量検索"""
        results = []
        
        for group_name, group_info in self._metadata["feature_groups"].items():
            for feature in group_info["features"]:
                if query.lower() in feature["name"].lower():
                    results.append({
                        "group": group_name,
                        "feature": feature["name"],
                        "dtype": feature["dtype"],
                    })
        
        return results
