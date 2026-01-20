"""
エッジ推論最適化（TensorRT/ONNX Runtime inspired）

Implements: F-EDGE-001
設計思想:
- 推論最適化
- バッチ処理
- 遅延測定
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """推論メトリクス"""
    latency_ms: float
    throughput: float  # samples/sec
    batch_size: int
    memory_mb: Optional[float] = None


class EdgeOptimizer:
    """
    エッジ推論最適化
    
    Features:
    - 推論遅延測定
    - バッチサイズ最適化
    - キャッシング
    
    Example:
        >>> optimizer = EdgeOptimizer(model)
        >>> metrics = optimizer.benchmark(X, batch_size=32)
    """
    
    def __init__(
        self,
        model,
        cache_enabled: bool = True,
    ):
        self.model = model
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, np.ndarray] = {}
    
    def benchmark(
        self,
        X: pd.DataFrame,
        batch_size: int = 32,
        n_runs: int = 10,
    ) -> InferenceMetrics:
        """推論ベンチマーク"""
        latencies = []
        
        for _ in range(n_runs):
            start = time.perf_counter()
            
            # バッチ処理
            for i in range(0, len(X), batch_size):
                batch = X.iloc[i:i+batch_size]
                self.model.predict(batch)
            
            latency = (time.perf_counter() - start) * 1000  # ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        throughput = len(X) / (avg_latency / 1000)
        
        return InferenceMetrics(
            latency_ms=avg_latency,
            throughput=throughput,
            batch_size=batch_size,
        )
    
    def find_optimal_batch_size(
        self,
        X: pd.DataFrame,
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
    ) -> int:
        """最適バッチサイズを探索"""
        best_throughput = 0
        best_batch_size = 1
        
        for bs in batch_sizes:
            if bs > len(X):
                continue
            
            metrics = self.benchmark(X, batch_size=bs, n_runs=3)
            
            if metrics.throughput > best_throughput:
                best_throughput = metrics.throughput
                best_batch_size = bs
        
        logger.info(f"Optimal batch size: {best_batch_size} ({best_throughput:.0f} samples/sec)")
        return best_batch_size
    
    def cached_predict(
        self,
        X: pd.DataFrame,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """キャッシュ付き予測"""
        if not self.cache_enabled:
            return self.model.predict(X)
        
        if cache_key is None:
            cache_key = str(hash(X.values.tobytes()))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        predictions = self.model.predict(X)
        self._cache[cache_key] = predictions
        
        return predictions
    
    def warmup(self, X: pd.DataFrame, n_runs: int = 5) -> None:
        """ウォームアップ"""
        sample = X.iloc[:min(10, len(X))]
        
        for _ in range(n_runs):
            self.model.predict(sample)
        
        logger.info(f"Warmup complete ({n_runs} runs)")
    
    def profile(self, X: pd.DataFrame) -> Dict[str, float]:
        """プロファイリング"""
        # 各ステップの時間計測
        times = {}
        
        # 特徴量変換（もしあれば）
        start = time.perf_counter()
        X_np = X.values
        times['to_numpy_ms'] = (time.perf_counter() - start) * 1000
        
        # 予測
        start = time.perf_counter()
        self.model.predict(X)
        times['predict_ms'] = (time.perf_counter() - start) * 1000
        
        times['total_ms'] = sum(times.values())
        
        return times
    
    def clear_cache(self) -> None:
        """キャッシュクリア"""
        self._cache.clear()
