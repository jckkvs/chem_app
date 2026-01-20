"""
ユーティリティ関数

Implements: F-UTILS-001
設計思想:
- 汎用ヘルパー
- データ変換
- ファイル操作
"""

from __future__ import annotations

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ================== ファイル操作 ==================

def save_pickle(obj: Any, path: str) -> None:
    """Pickleで保存"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Pickleから読み込み"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, path: str) -> None:
    """JSONで保存"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str) -> Dict:
    """JSONから読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ================== データ変換 ==================

def to_numpy(data: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> np.ndarray:
    """NumPy配列に変換"""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    return data


def to_dataframe(data: Union[np.ndarray, dict, list], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """DataFrameに変換"""
    if isinstance(data, dict):
        return pd.DataFrame(data)
    return pd.DataFrame(data, columns=columns)


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """ネストした辞書をフラット化"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ================== 時間・日付 ==================

def timestamp() -> str:
    """現在のタイムスタンプ"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def elapsed_str(seconds: float) -> str:
    """経過時間を文字列化"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


# ================== 数学・統計 ==================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """ゼロ除算を回避"""
    return a / b if b != 0 else default


def normalize(arr: np.ndarray) -> np.ndarray:
    """0-1正規化"""
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val) / (max_val - min_val + 1e-9)


def standardize(arr: np.ndarray) -> np.ndarray:
    """標準化"""
    return (arr - arr.mean()) / (arr.std() + 1e-9)


# ================== 分子関連 ==================

def canonical_smiles(smiles: str) -> Optional[str]:
    """標準化SMILES"""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def smiles_to_mol(smiles: str):
    """SMILES→Mol変換"""
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


# ================== ロギング ==================

def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """メトリクスをログ出力"""
    for k, v in metrics.items():
        logger.info(f"{prefix}{k}: {v:.4f}")
