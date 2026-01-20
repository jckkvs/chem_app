"""
分子類似度検索

Implements: F-SIMILARITY-001
設計思想:
- 高速な類似分子検索
- 複数のフィンガープリントと距離メトリック対応
- リード最適化、活性クリフ検出に活用

参考文献:
- Tanimoto Coefficient: Willett, Drug Discovery Today 2006
- ECFP/FCFP: Rogers & Hahn, J. Chem. Inf. Model. 2010
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimilaritySearchResult:
    """類似度検索結果"""
    query_smiles: str
    similar_smiles: List[str]
    similarities: List[float]
    indices: List[int]
    
    def top_k(self, k: int = 5) -> 'SimilaritySearchResult':
        """上位k件を返す"""
        return SimilaritySearchResult(
            query_smiles=self.query_smiles,
            similar_smiles=self.similar_smiles[:k],
            similarities=self.similarities[:k],
            indices=self.indices[:k],
        )
    
    def to_df(self) -> pd.DataFrame:
        """DataFrameに変換"""
        return pd.DataFrame({
            'smiles': self.similar_smiles,
            'similarity': self.similarities,
            'index': self.indices,
        })


class MolecularSimilaritySearch:
    """
    分子類似度検索エンジン
    
    Usage:
        search = MolecularSimilaritySearch()
        search.index(database_smiles)
        results = search.search('CCO', k=10)
    """
    
    def __init__(
        self,
        fp_type: Literal['morgan', 'rdkit', 'maccs', 'topological'] = 'morgan',
        fp_radius: int = 2,
        fp_bits: int = 2048,
        similarity_metric: Literal['tanimoto', 'dice', 'cosine'] = 'tanimoto',
    ):
        """
        Args:
            fp_type: フィンガープリントタイプ
            fp_radius: Morganフィンガープリントの半径
            fp_bits: ビット数
            similarity_metric: 類似度メトリック
        """
        self.fp_type = fp_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.similarity_metric = similarity_metric
        
        self._database_fps: Optional[np.ndarray] = None
        self._database_smiles: Optional[List[str]] = None
        self._indexed = False
    
    def _compute_fingerprint(self, smiles: str) -> Optional[np.ndarray]:
        """単一SMILESからフィンガープリントを計算"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, MACCSkeys, RDKFingerprint
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if self.fp_type == 'morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fp_radius, nBits=self.fp_bits
                )
            elif self.fp_type == 'rdkit':
                fp = RDKFingerprint(mol, fpSize=self.fp_bits)
            elif self.fp_type == 'maccs':
                fp = MACCSkeys.GenMACCSKeys(mol)
            elif self.fp_type == 'topological':
                fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(
                    mol, nBits=self.fp_bits
                )
            else:
                raise ValueError(f"Unknown fp_type: {self.fp_type}")
            
            return np.array(fp)
            
        except Exception as e:
            logger.debug(f"Fingerprint computation failed: {e}")
            return None
    
    def _compute_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """複数SMILESからフィンガープリント行列を計算"""
        fps = []
        
        for smi in smiles_list:
            fp = self._compute_fingerprint(smi)
            if fp is not None:
                fps.append(fp)
            else:
                # 失敗した場合はゼロベクトル
                fps.append(np.zeros(self.fp_bits))
        
        return np.array(fps)
    
    def _tanimoto_similarity(
        self, 
        query_fp: np.ndarray, 
        database_fps: np.ndarray
    ) -> np.ndarray:
        """Tanimoto類似度を計算"""
        # 効率的なベクトル化計算
        query_sum = query_fp.sum()
        db_sums = database_fps.sum(axis=1)
        
        intersection = np.dot(database_fps, query_fp)
        union = query_sum + db_sums - intersection
        
        # ゼロ除算対策
        similarities = np.where(
            union > 0,
            intersection / union,
            0.0
        )
        
        return similarities
    
    def _dice_similarity(
        self, 
        query_fp: np.ndarray, 
        database_fps: np.ndarray
    ) -> np.ndarray:
        """Dice類似度を計算"""
        query_sum = query_fp.sum()
        db_sums = database_fps.sum(axis=1)
        
        intersection = np.dot(database_fps, query_fp)
        
        similarities = np.where(
            (query_sum + db_sums) > 0,
            2 * intersection / (query_sum + db_sums),
            0.0
        )
        
        return similarities
    
    def _cosine_similarity(
        self, 
        query_fp: np.ndarray, 
        database_fps: np.ndarray
    ) -> np.ndarray:
        """コサイン類似度を計算"""
        query_norm = np.linalg.norm(query_fp)
        db_norms = np.linalg.norm(database_fps, axis=1)
        
        dot_products = np.dot(database_fps, query_fp)
        
        similarities = np.where(
            (query_norm * db_norms) > 0,
            dot_products / (query_norm * db_norms),
            0.0
        )
        
        return similarities
    
    def index(self, smiles_list: List[str]) -> 'MolecularSimilaritySearch':
        """データベースをインデックス化"""
        logger.info(f"Indexing {len(smiles_list)} molecules...")
        
        self._database_smiles = smiles_list
        self._database_fps = self._compute_fingerprints(smiles_list)
        self._indexed = True
        
        logger.info(f"Indexed {len(smiles_list)} molecules")
        
        return self
    
    def search(
        self, 
        query_smiles: str, 
        k: int = 10,
        threshold: float = 0.0,
    ) -> SimilaritySearchResult:
        """
        類似分子を検索
        
        Args:
            query_smiles: クエリSMILES
            k: 返す結果数
            threshold: 類似度閾値
            
        Returns:
            SimilaritySearchResult: 検索結果
        """
        if not self._indexed:
            raise RuntimeError("index()を先に呼び出してください")
        
        # クエリのフィンガープリント
        query_fp = self._compute_fingerprint(query_smiles)
        if query_fp is None:
            return SimilaritySearchResult(
                query_smiles=query_smiles,
                similar_smiles=[],
                similarities=[],
                indices=[],
            )
        
        # 類似度計算
        if self.similarity_metric == 'tanimoto':
            similarities = self._tanimoto_similarity(query_fp, self._database_fps)
        elif self.similarity_metric == 'dice':
            similarities = self._dice_similarity(query_fp, self._database_fps)
        else:  # cosine
            similarities = self._cosine_similarity(query_fp, self._database_fps)
        
        # 閾値でフィルタリング
        valid_mask = similarities >= threshold
        valid_indices = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_mask]
        
        # ソート
        sorted_order = np.argsort(valid_similarities)[::-1][:k]
        
        result_indices = valid_indices[sorted_order].tolist()
        result_similarities = valid_similarities[sorted_order].tolist()
        result_smiles = [self._database_smiles[i] for i in result_indices]
        
        return SimilaritySearchResult(
            query_smiles=query_smiles,
            similar_smiles=result_smiles,
            similarities=result_similarities,
            indices=result_indices,
        )
    
    def batch_search(
        self,
        query_smiles_list: List[str],
        k: int = 10,
        threshold: float = 0.0,
    ) -> List[SimilaritySearchResult]:
        """複数クエリを一括検索"""
        return [
            self.search(smi, k=k, threshold=threshold) 
            for smi in query_smiles_list
        ]
    
    def find_nearest_neighbors(
        self,
        smiles_list: List[str],
        k: int = 5,
    ) -> pd.DataFrame:
        """
        データセット内の最近傍を見つける
        
        各分子について、データセット内で最も類似した分子を返す。
        活性クリフの検出に有用。
        """
        if not self._indexed:
            self.index(smiles_list)
        
        results = []
        
        for i, smi in enumerate(smiles_list):
            # 自分自身を除くために k+1 件取得
            search_result = self.search(smi, k=k+1)
            
            # 自分自身を除外
            filtered = [
                (s, sim, idx) 
                for s, sim, idx in zip(
                    search_result.similar_smiles,
                    search_result.similarities,
                    search_result.indices,
                )
                if idx != i
            ][:k]
            
            if filtered:
                nn_smiles, nn_sim, nn_idx = filtered[0]
            else:
                nn_smiles, nn_sim, nn_idx = '', 0.0, -1
            
            results.append({
                'smiles': smi,
                'index': i,
                'nearest_neighbor_smiles': nn_smiles,
                'nearest_neighbor_similarity': nn_sim,
                'nearest_neighbor_index': nn_idx,
            })
        
        return pd.DataFrame(results)


class ActivityCliffDetector:
    """
    活性クリフ検出
    
    構造的に類似しているが活性が大きく異なる分子ペアを検出。
    
    Reference: 
        Stumpfe & Bajorath, "Exploring Activity Cliffs in 
        Medicinal Chemistry", J. Med. Chem. 2012
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        activity_ratio_threshold: float = 10.0,
    ):
        """
        Args:
            similarity_threshold: 類似度閾値（これ以上で類似と判定）
            activity_ratio_threshold: 活性比閾値（これ以上で活性クリフ）
        """
        self.similarity_threshold = similarity_threshold
        self.activity_ratio_threshold = activity_ratio_threshold
        
        self._search_engine: Optional[MolecularSimilaritySearch] = None
    
    def detect(
        self,
        smiles_list: List[str],
        activities: np.ndarray,
    ) -> pd.DataFrame:
        """
        活性クリフを検出
        
        Args:
            smiles_list: SMILESリスト
            activities: 活性値（数値）
            
        Returns:
            DataFrame: 活性クリフペア
        """
        # 類似度検索エンジン
        self._search_engine = MolecularSimilaritySearch()
        self._search_engine.index(smiles_list)
        
        cliffs = []
        
        for i, smi in enumerate(smiles_list):
            # 類似分子を検索
            result = self._search_engine.search(
                smi, 
                k=50, 
                threshold=self.similarity_threshold
            )
            
            for j, (sim_smi, sim, idx) in enumerate(zip(
                result.similar_smiles,
                result.similarities,
                result.indices,
            )):
                if idx <= i:  # 重複を避ける
                    continue
                
                # 活性比を計算
                act_i = activities[i]
                act_j = activities[idx]
                
                if act_i <= 0 or act_j <= 0:
                    continue
                
                ratio = max(act_i, act_j) / min(act_i, act_j)
                
                if ratio >= self.activity_ratio_threshold:
                    cliffs.append({
                        'smiles_1': smi,
                        'smiles_2': sim_smi,
                        'index_1': i,
                        'index_2': idx,
                        'similarity': sim,
                        'activity_1': act_i,
                        'activity_2': act_j,
                        'activity_ratio': ratio,
                    })
        
        df = pd.DataFrame(cliffs)
        
        if not df.empty:
            df = df.sort_values('activity_ratio', ascending=False)
        
        logger.info(f"Detected {len(df)} activity cliffs")
        
        return df


def search_similar(
    query_smiles: str,
    database_smiles: List[str],
    k: int = 10,
) -> SimilaritySearchResult:
    """便利関数: 類似分子検索"""
    search = MolecularSimilaritySearch()
    search.index(database_smiles)
    return search.search(query_smiles, k=k)


def detect_activity_cliffs(
    smiles_list: List[str],
    activities: np.ndarray,
    similarity_threshold: float = 0.7,
) -> pd.DataFrame:
    """便利関数: 活性クリフ検出"""
    detector = ActivityCliffDetector(similarity_threshold=similarity_threshold)
    return detector.detect(smiles_list, activities)
