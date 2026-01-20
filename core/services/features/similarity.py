"""
類似分子検索エンジン

Implements: F-SIMSEARCH-001
設計思想:
- 予測値が類似した分子を検索
- 構造類似性（フィンガープリント）
- 特徴量空間での類似性
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SimilarMolecule:
    """類似分子結果"""
    smiles: str
    distance: float
    similarity: float
    index: int
    properties: Optional[Dict[str, float]] = None


class SimilarMoleculeSearcher:
    """
    類似分子検索エンジン
    
    Features:
    - 特徴量空間での近傍検索
    - フィンガープリント類似性
    - 予測値での類似検索
    
    Example:
        >>> searcher = SimilarMoleculeSearcher()
        >>> searcher.fit(X, smiles_list)
        >>> similar = searcher.search("CCO", top_k=5)
    """
    
    def __init__(
        self,
        metric: str = 'euclidean',
        n_neighbors: int = 10,
        use_fingerprint: bool = True,
        fp_radius: int = 2,
        fp_bits: int = 2048,
    ):
        """
        Args:
            metric: 距離関数 ('euclidean', 'manhattan', 'cosine')
            n_neighbors: 最大近傍数
            use_fingerprint: フィンガープリントを使用するか
            fp_radius: Morgan FP radius
            fp_bits: FPビット数
        """
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.use_fingerprint = use_fingerprint
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        
        self.nn_model_: Optional[NearestNeighbors] = None
        self.scaler_: Optional[StandardScaler] = None
        self.X_scaled_: Optional[np.ndarray] = None
        self.smiles_list_: List[str] = []
        self.properties_: Optional[pd.DataFrame] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        smiles_list: List[str],
        properties: Optional[pd.DataFrame] = None,
    ) -> 'SimilarMoleculeSearcher':
        """
        検索インデックスを構築
        
        Args:
            X: 特徴量行列
            smiles_list: SMILESリスト
            properties: 各分子の物性値
        """
        self.smiles_list_ = smiles_list
        self.properties_ = properties
        
        # スケーリング
        self.scaler_ = StandardScaler()
        self.X_scaled_ = self.scaler_.fit_transform(X)
        
        # 近傍モデル構築
        self.nn_model_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(smiles_list)),
            metric=self.metric,
            algorithm='auto',
        )
        self.nn_model_.fit(self.X_scaled_)
        
        logger.info(f"Search index built: {len(smiles_list)} molecules")
        return self
    
    def search(
        self,
        query_smiles: str,
        top_k: int = 5,
        feature_extractor: Optional[Any] = None,
    ) -> List[SimilarMolecule]:
        """
        類似分子を検索
        
        Args:
            query_smiles: クエリSMILES
            top_k: 返す分子数
            feature_extractor: 特徴量抽出器
        """
        if self.nn_model_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        # 特徴量抽出
        if feature_extractor is None:
            from core.services.features.rdkit_eng import RDKitFeatureExtractor
            feature_extractor = RDKitFeatureExtractor()
        
        X_query = feature_extractor.transform([query_smiles])
        X_query = X_query.drop(columns=['SMILES'], errors='ignore')
        X_query_scaled = self.scaler_.transform(X_query)
        
        # 近傍検索
        distances, indices = self.nn_model_.kneighbors(
            X_query_scaled,
            n_neighbors=min(top_k, len(self.smiles_list_)),
        )
        
        # 結果構築
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            props = None
            if self.properties_ is not None and idx < len(self.properties_):
                props = self.properties_.iloc[idx].to_dict()
            
            # 類似度（距離の逆数ベース）
            similarity = 1.0 / (1.0 + dist)
            
            results.append(SimilarMolecule(
                smiles=self.smiles_list_[idx],
                distance=float(dist),
                similarity=float(similarity),
                index=int(idx),
                properties=props,
            ))
        
        return results
    
    def search_by_properties(
        self,
        target_properties: Dict[str, float],
        top_k: int = 5,
    ) -> List[SimilarMolecule]:
        """
        物性値で類似分子を検索
        
        Args:
            target_properties: 目標物性値
            top_k: 返す分子数
        """
        if self.properties_ is None:
            raise ValueError("propertiesを指定してfit()を呼び出してください")
        
        # 対象カラム
        cols = [c for c in target_properties.keys() if c in self.properties_.columns]
        if not cols:
            raise ValueError("指定された物性がデータに存在しません")
        
        # 距離計算
        target = np.array([target_properties[c] for c in cols])
        data = self.properties_[cols].values
        
        # 欠損を大きな値で置換
        data = np.nan_to_num(data, nan=1e10)
        
        # 正規化
        data_std = data.std(axis=0)
        data_std[data_std == 0] = 1
        distances = np.sqrt(((data - target) / data_std) ** 2).sum(axis=1)
        
        # ソート
        sorted_indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in sorted_indices:
            similarity = 1.0 / (1.0 + distances[idx])
            results.append(SimilarMolecule(
                smiles=self.smiles_list_[idx],
                distance=float(distances[idx]),
                similarity=float(similarity),
                index=int(idx),
                properties=self.properties_.iloc[idx].to_dict(),
            ))
        
        return results
    
    def compute_tanimoto_similarity(
        self,
        smiles1: str,
        smiles2: str,
    ) -> float:
        """
        Tanimoto類似度（Morgan FP）を計算
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit import DataStructs
            
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            fp1 = AllChem.GetMorganFingerprintAsBitVect(
                mol1, self.fp_radius, nBits=self.fp_bits
            )
            fp2 = AllChem.GetMorganFingerprintAsBitVect(
                mol2, self.fp_radius, nBits=self.fp_bits
            )
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
            
        except Exception as e:
            logger.warning(f"Tanimoto calculation failed: {e}")
            return 0.0
