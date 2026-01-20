"""
特徴量抽出器 基底クラス

Implements: F-BASE-001
設計思想:
- Strategy Patternによる抽出器の切り替え
- fit/transform/fit_transformインターフェース
- 永続化対応（save/load）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np


class BaseFeatureExtractor(ABC):
    """
    SMILES特徴量抽出器の抽象基底クラス
    
    Strategy Patternを実装し、RDKit/XTB/UMAPなどの抽出器を
    統一インターフェースで扱えるようにする。
    
    Subclasses:
    - RDKitFeatureExtractor: 分子記述子
    - XTBFeatureExtractor: 量子化学記述子
    - UMAFeatureExtractor: UMAP埋め込み
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: サブクラス固有の設定
        """
        self.config = kwargs
        self._is_fitted = False
    
    def fit(
        self, 
        smiles_list: List[str], 
        y: Optional[Any] = None
    ) -> 'BaseFeatureExtractor':
        """
        抽出器をデータにフィット
        
        Stateless抽出器（RDKit, XTB）ではno-op。
        Stateful抽出器（UMAP）では埋め込み空間を学習。
        
        Args:
            smiles_list: SMILESのリスト
            y: ターゲット変数（Supervised学習用）
            
        Returns:
            self
        """
        self._is_fitted = True
        return self
    
    @abstractmethod
    def transform(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        SMILESリストを特徴量DataFrameに変換
        
        Args:
            smiles_list: SMILESのリスト
            
        Returns:
            pd.DataFrame: 特徴量DataFrame
        """
        pass
    
    def fit_transform(
        self, 
        smiles_list: List[str], 
        y: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        fit + transform を一度に実行
        
        Args:
            smiles_list: SMILESのリスト
            y: ターゲット変数
            
        Returns:
            pd.DataFrame: 特徴量DataFrame
        """
        return self.fit(smiles_list, y).transform(smiles_list)
    
    @property
    def is_fitted(self) -> bool:
        """フィット済みか"""
        return self._is_fitted
    
    @property
    def descriptor_names(self) -> List[str]:
        """
        記述子名のリスト（サブクラスでオーバーライド推奨）
        """
        return []
    
    @property
    def n_descriptors(self) -> int:
        """記述子の数"""
        return len(self.descriptor_names)
    
    def save(self, path: str) -> None:
        """
        抽出器の状態を保存
        
        Stateless抽出器ではオーバーライド不要。
        Stateful抽出器（UMAP）ではモデルを保存。
        
        Args:
            path: 保存先パス
        """
        import joblib
        joblib.dump({'config': self.config, 'is_fitted': self._is_fitted}, path)
    
    def load(self, path: str) -> 'BaseFeatureExtractor':
        """
        抽出器の状態を読み込み
        
        Args:
            path: 読み込み元パス
            
        Returns:
            self
        """
        import joblib
        data = joblib.load(path)
        self.config = data.get('config', {})
        self._is_fitted = data.get('is_fitted', False)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
