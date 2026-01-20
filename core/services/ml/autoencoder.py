"""
オートエンコーダ（VAE for molecules）

Implements: F-AE-001
設計思想:
- 次元削減
- 潜在空間学習
- 分子生成準備
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class AutoencoderResult:
    """オートエンコーダ結果"""
    latent: np.ndarray
    reconstructed: np.ndarray
    reconstruction_error: float


class MolecularAutoencoder:
    """
    分子オートエンコーダ（VAE inspired）
    
    Features:
    - 次元削減
    - 潜在空間学習
    - 特徴量再構成
    
    Example:
        >>> ae = MolecularAutoencoder(latent_dim=32)
        >>> ae.fit(X)
        >>> latent = ae.encode(X_new)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (128, 64),
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        self.scaler_: Optional[StandardScaler] = None
        self.encoder_: Optional[PCA] = None
        self.input_dim_: int = 0
    
    def fit(self, X: pd.DataFrame) -> 'MolecularAutoencoder':
        """学習"""
        self.input_dim_ = X.shape[1]
        
        # 正規化
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # PCAベースのエンコーダ（簡易版）
        n_components = min(self.latent_dim, X.shape[1], X.shape[0])
        self.encoder_ = PCA(n_components=n_components)
        self.encoder_.fit(X_scaled)
        
        logger.info(f"Autoencoder fitted: {X.shape[1]} -> {n_components}D")
        
        return self
    
    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """エンコード"""
        if self.scaler_ is None or self.encoder_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        X_scaled = self.scaler_.transform(X)
        latent = self.encoder_.transform(X_scaled)
        
        # 次元パディング
        if latent.shape[1] < self.latent_dim:
            padding = np.zeros((latent.shape[0], self.latent_dim - latent.shape[1]))
            latent = np.hstack([latent, padding])
        
        return latent
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """デコード"""
        if self.encoder_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        # PCAの次元のみ使用
        n_components = self.encoder_.n_components_
        latent_truncated = latent[:, :n_components]
        
        reconstructed = self.encoder_.inverse_transform(latent_truncated)
        reconstructed = self.scaler_.inverse_transform(reconstructed)
        
        return reconstructed
    
    def transform(self, X: pd.DataFrame) -> AutoencoderResult:
        """エンコード→デコード"""
        latent = self.encode(X)
        reconstructed = self.decode(latent)
        
        # 再構成誤差
        X_np = X.values
        error = np.mean(np.abs(X_np - reconstructed))
        
        return AutoencoderResult(
            latent=latent,
            reconstructed=reconstructed,
            reconstruction_error=error,
        )
    
    def sample_latent(self, n_samples: int = 10) -> np.ndarray:
        """潜在空間からサンプリング"""
        return np.random.randn(n_samples, self.latent_dim)
    
    def interpolate(
        self,
        X1: pd.DataFrame,
        X2: pd.DataFrame,
        n_steps: int = 5,
    ) -> np.ndarray:
        """潜在空間で補間"""
        z1 = self.encode(X1)[0]
        z2 = self.encode(X2)[0]
        
        alphas = np.linspace(0, 1, n_steps)
        interpolated = np.array([
            (1 - alpha) * z1 + alpha * z2
            for alpha in alphas
        ])
        
        return self.decode(interpolated)
