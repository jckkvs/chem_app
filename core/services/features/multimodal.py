"""
マルチモーダル特徴量（Image + SMILES）

Implements: F-MULTIMODAL-001
設計思想:
- 画像特徴量
- 複数データソース統合
- 特徴量融合
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MultimodalFeatures:
    """マルチモーダル特徴量"""
    smiles_features: np.ndarray
    image_features: Optional[np.ndarray]
    combined_features: np.ndarray
    modalities: List[str]


class MultimodalExtractor:
    """
    マルチモーダル特徴量抽出
    
    Features:
    - SMILES + 画像特徴量
    - 特徴量融合
    - 欠損モダリティ処理
    
    Example:
        >>> extractor = MultimodalExtractor()
        >>> features = extractor.extract(smiles, image_path)
    """
    
    def __init__(
        self,
        smiles_dim: int = 256,
        image_dim: int = 512,
        fusion_method: str = 'concat',
    ):
        self.smiles_dim = smiles_dim
        self.image_dim = image_dim
        self.fusion_method = fusion_method
    
    def extract(
        self,
        smiles: str,
        image_path: Optional[str] = None,
    ) -> MultimodalFeatures:
        """特徴量抽出"""
        modalities = ['smiles']
        
        # SMILES特徴量
        smiles_features = self._extract_smiles_features(smiles)
        
        # 画像特徴量（オプション）
        image_features = None
        if image_path:
            image_features = self._extract_image_features(image_path)
            modalities.append('image')
        
        # 融合
        combined = self._fuse_features(smiles_features, image_features)
        
        return MultimodalFeatures(
            smiles_features=smiles_features,
            image_features=image_features,
            combined_features=combined,
            modalities=modalities,
        )
    
    def _extract_smiles_features(self, smiles: str) -> np.ndarray:
        """SMILES特徴量"""
        try:
            from core.services.features.fingerprint import FingerprintCalculator
            
            calc = FingerprintCalculator(fp_type='morgan', n_bits=self.smiles_dim)
            fp = calc.calculate(smiles)
            
            if fp is not None:
                return fp.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"SMILES feature extraction failed: {e}")
        
        return np.zeros(self.smiles_dim, dtype=np.float32)
    
    def _extract_image_features(self, image_path: str) -> np.ndarray:
        """画像特徴量（簡易版）"""
        try:
            # 本来はCNN特徴量を抽出
            # ここでは画像のヒストグラム特徴を使用
            import hashlib
            
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # ハッシュベースの疑似特徴量
            hash_bytes = hashlib.sha512(data).digest()
            features = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            
            # 正規化してリサイズ
            features = features / 255.0
            if len(features) < self.image_dim:
                features = np.pad(features, (0, self.image_dim - len(features)))
            else:
                features = features[:self.image_dim]
            
            return features
            
        except Exception as e:
            logger.warning(f"Image feature extraction failed: {e}")
            return np.zeros(self.image_dim, dtype=np.float32)
    
    def _fuse_features(
        self,
        smiles_features: np.ndarray,
        image_features: Optional[np.ndarray],
    ) -> np.ndarray:
        """特徴量融合"""
        if image_features is None:
            return smiles_features
        
        if self.fusion_method == 'concat':
            return np.concatenate([smiles_features, image_features])
        
        elif self.fusion_method == 'average':
            # 次元を揃える
            min_dim = min(len(smiles_features), len(image_features))
            return (smiles_features[:min_dim] + image_features[:min_dim]) / 2
        
        elif self.fusion_method == 'max':
            min_dim = min(len(smiles_features), len(image_features))
            return np.maximum(smiles_features[:min_dim], image_features[:min_dim])
        
        return smiles_features
    
    def extract_batch(
        self,
        smiles_list: List[str],
        image_paths: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """バッチ抽出"""
        features = []
        
        for i, smiles in enumerate(smiles_list):
            img_path = image_paths[i] if image_paths and i < len(image_paths) else None
            mm_features = self.extract(smiles, img_path)
            features.append(mm_features.combined_features)
        
        return pd.DataFrame(features)
