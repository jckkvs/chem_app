"""
データバージョニング（DVC inspired）

Implements: F-DATAVER-001
設計思想:
- データセットバージョン管理
- 差分追跡
- 再現性確保
"""

from __future__ import annotations

import logging
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataVersion:
    """データバージョン"""
    version: int
    hash: str
    rows: int
    cols: int
    created_at: str
    description: str = ""
    schema: Dict[str, str] = None


class DataVersioning:
    """
    データバージョニング（DVC inspired）
    
    Features:
    - ハッシュベース追跡
    - スキーマ検証
    - 差分検出
    
    Example:
        >>> dv = DataVersioning()
        >>> dv.commit("dataset", df, description="Initial version")
        >>> df_v1 = dv.checkout("dataset", version=1)
    """
    
    def __init__(self, data_dir: str = "data_versions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.data_dir / "versions.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"datasets": {}}
    
    def _save_metadata(self) -> None:
        with open(self._metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)
    
    def _compute_hash(self, df: pd.DataFrame) -> str:
        """データハッシュ計算"""
        content = df.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()[:16]
    
    def commit(
        self,
        name: str,
        df: pd.DataFrame,
        description: str = "",
    ) -> DataVersion:
        """データをコミット"""
        if name not in self._metadata["datasets"]:
            self._metadata["datasets"][name] = {"versions": []}
        
        versions = self._metadata["datasets"][name]["versions"]
        new_version = len(versions) + 1
        data_hash = self._compute_hash(df)
        
        # 重複チェック
        if versions and versions[-1]["hash"] == data_hash:
            logger.info(f"No changes detected for {name}")
            return DataVersion(**versions[-1])
        
        # バージョンディレクトリ
        version_dir = self.data_dir / name / f"v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # データ保存
        df.to_parquet(version_dir / "data.parquet", index=False)
        
        # スキーマ
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        version_info = DataVersion(
            version=new_version,
            hash=data_hash,
            rows=len(df),
            cols=len(df.columns),
            created_at=datetime.now().isoformat(),
            description=description,
            schema=schema,
        )
        
        versions.append(asdict(version_info))
        self._save_metadata()
        
        logger.info(f"Committed {name} v{new_version} ({len(df)} rows)")
        return version_info
    
    def checkout(
        self,
        name: str,
        version: Optional[int] = None,
    ) -> pd.DataFrame:
        """データをチェックアウト"""
        if name not in self._metadata["datasets"]:
            raise ValueError(f"Dataset not found: {name}")
        
        versions = self._metadata["datasets"][name]["versions"]
        v = version or len(versions)
        
        data_path = self.data_dir / name / f"v{v}" / "data.parquet"
        return pd.read_parquet(data_path)
    
    def diff(self, name: str, v1: int, v2: int) -> Dict[str, Any]:
        """差分検出"""
        df1 = self.checkout(name, v1)
        df2 = self.checkout(name, v2)
        
        return {
            "rows_added": len(df2) - len(df1),
            "cols_v1": list(df1.columns),
            "cols_v2": list(df2.columns),
            "cols_added": [c for c in df2.columns if c not in df1.columns],
            "cols_removed": [c for c in df1.columns if c not in df2.columns],
        }
    
    def list_versions(self, name: str) -> List[DataVersion]:
        """バージョン一覧"""
        if name not in self._metadata["datasets"]:
            return []
        return [DataVersion(**v) for v in self._metadata["datasets"][name]["versions"]]
