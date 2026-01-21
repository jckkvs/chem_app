"""
UMAP特徴量抽出エンジン - Metric Learning対応版

Implements: F-004
設計思想:
- RDKit記述子をベースにUMAP次元削減
- Supervised UMAP（ターゲット変数を考慮）
- 学習済みモデルの永続化・再利用

参考文献:
- UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
  (McInnes et al., 2018)
- DOI: 10.21105/joss.00861
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

from .base import BaseFeatureExtractor
from .rdkit_eng import RDKitFeatureExtractor

logger = logging.getLogger(__name__)


class UMAFeatureExtractor(BaseFeatureExtractor):
    """
    UMAP (Uniform Manifold Approximation and Projection) 特徴量抽出器
    
    Workflow:
    SMILES → RDKit記述子 → StandardScaler → UMAP → 低次元埋め込み
    
    Features:
    - Supervised UMAP: ターゲット変数を考慮した埋め込み
    - Unsupervised UMAP: 構造類似性に基づく埋め込み
    - 埋め込みモデルの永続化
    
    Example:
        >>> extractor = UMAFeatureExtractor(n_components=10)
        >>> extractor.fit(smiles_list, y=target)
        >>> embeddings = extractor.transform(smiles_list)
    """
    
    def __init__(
        self,
        n_components: int = 10,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: int = 42,
        include_smiles: bool = True,
        base_extractor: Optional[BaseFeatureExtractor] = None,
        **kwargs
    ):
        """
        Args:
            n_components: 埋め込み次元数
            n_neighbors: 近傍サンプル数（大きいほどグローバル構造を重視）
            min_dist: 埋め込み空間での最小距離
            metric: 距離メトリック
            random_state: 乱数シード
            include_smiles: 出力にSMILESカラムを含めるか
            base_extractor: ベース特徴量抽出器（デフォルト: RDKit）
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.include_smiles = include_smiles
        
        # ベース抽出器
        self.base_extractor = base_extractor or RDKitFeatureExtractor(
            categories=['lipophilicity', 'structural', 'topological']
        )
        
        # 学習される状態
        self.scaler: Optional[StandardScaler] = None
        self.reducer: Optional[umap.UMAP] = None
        self._feature_columns: Optional[List[str]] = None
    
    def fit(
        self, 
        smiles_list: List[str], 
        y: Optional[Any] = None
    ) -> 'UMAFeatureExtractor':
        """
        UMAPモデルを学習
        
        Args:
            smiles_list: SMILESのリスト
            y: ターゲット変数（Supervised UMAPの場合）
            
        Returns:
            self
        """
        # 1. ベース特徴量抽出
        logger.info("ベース特徴量を抽出中...")
        df_base = self.base_extractor.transform(smiles_list)
        
        # 数値カラムのみ選択（SMILES等を除外）
        X = df_base.select_dtypes(include=[np.number])
        self._feature_columns = list(X.columns)
        
        # 欠損値処理
        X = X.fillna(0)
        
        # 2. スケーリング
        logger.info("スケーリング中...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)
        
        # 3. UMAP学習
        logger.info(f"UMAP学習中... (n_components={self.n_components})")
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        
        if y is not None:
            # Supervised UMAP
            logger.info("Supervised UMAPモード")
            self.reducer.fit(X_scaled, y=y)
        else:
            # Unsupervised UMAP
            self.reducer.fit(X_scaled)
        
        self._is_fitted = True
        logger.info("UMAP学習完了")
        return self
    
    def transform(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        学習済みUMAPで次元削減
        
        Args:
            smiles_list: SMILESのリスト
            
        Returns:
            pd.DataFrame: UMAP埋め込みDataFrame
        """
        if not self._is_fitted or self.scaler is None or self.reducer is None:
            logger.warning("未学習状態。ゼロ埋め込みを返します。")
            return self._return_empty(smiles_list)
        
        # 1. ベース特徴量抽出
        df_base = self.base_extractor.transform(smiles_list)
        
        # 学習時と同じカラムを使用
        if self._feature_columns:
            available_cols = [c for c in self._feature_columns if c in df_base.columns]
            X = df_base[available_cols].fillna(0)
        else:
            X = df_base.select_dtypes(include=[np.number]).fillna(0)
        
        # 2. スケーリング
        X_scaled = self.scaler.transform(X.values)
        
        # 3. UMAP変換
        embeddings = self.reducer.transform(X_scaled)
        
        # DataFrame作成
        columns = [f"UMA_{i}" for i in range(embeddings.shape[1])]
        df_emb = pd.DataFrame(embeddings, columns=columns)
        
        if self.include_smiles:
            df_emb.insert(0, 'SMILES', smiles_list)
        
        return df_emb
    
    def _return_empty(self, smiles_list: List[str]) -> pd.DataFrame:
        """未学習時の空埋め込み"""
        columns = [f"UMA_{i}" for i in range(self.n_components)]
        df = pd.DataFrame(0.0, index=range(len(smiles_list)), columns=columns)
        
        if self.include_smiles:
            df.insert(0, 'SMILES', smiles_list)
        
        return df
    
    def save(self, path: str) -> None:
        """
        学習済みモデルを保存
        
        Args:
            path: 保存先パス
        """
        if not self._is_fitted:
            raise RuntimeError("未学習モデルは保存できません")
        
        data = {
            'scaler': self.scaler,
            'reducer': self.reducer,
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'feature_columns': self._feature_columns,
        }
        joblib.dump(data, path)
        logger.info(f"UMAモデル保存: {path}")
    
    def load(self, path: str) -> 'UMAFeatureExtractor':
        """
        保存済みモデルを読み込み
        
        Args:
            path: 読み込み元パス
            
        Returns:
            self
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
        
        data = joblib.load(path)
        
        self.scaler = data['scaler']
        self.reducer = data['reducer']
        self.n_components = data.get('n_components', self.n_components)
        self.n_neighbors = data.get('n_neighbors', self.n_neighbors)
        self.min_dist = data.get('min_dist', self.min_dist)
        self.metric = data.get('metric', self.metric)
        self._feature_columns = data.get('feature_columns')
        self._is_fitted = True
        
        logger.info(f"UMAモデル読み込み: {path}")
        return self
    
    @property
    def descriptor_names(self) -> List[str]:
        """記述子名（埋め込み次元）のリスト"""
        return [f"UMA_{i}" for i in range(self.n_components)]
