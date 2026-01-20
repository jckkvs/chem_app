"""
バッチ処理エンジン

Implements: F-BATCH-001
設計思想:
- 大規模データの効率的処理
- 進捗表示とエラーハンドリング
- 並列処理とメモリ管理

機能:
- バッチ特徴量抽出
- バッチ予測
- バッチ評価
- 進捗コールバック
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Optional, Any, Callable, Iterable, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """バッチ処理結果"""
    total: int
    successful: int
    failed: int
    results: List[Any]
    errors: List[Dict[str, Any]]
    elapsed_time: float
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0
    
    def summary(self) -> str:
        return (
            f"Processed {self.total} items: "
            f"{self.successful} OK, {self.failed} failed "
            f"({self.success_rate:.1%}) in {self.elapsed_time:.2f}s"
        )


class ProgressCallback:
    """進捗コールバック"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """進捗を更新"""
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        
        pct = self.current / self.total * 100
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = '█' * filled + '░' * (bar_len - filled)
        
        sys.stdout.write(
            f"\r{self.desc}: {bar} {pct:.1f}% "
            f"({self.current}/{self.total}) "
            f"[{elapsed:.0f}s < {eta:.0f}s]"
        )
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()
    
    def close(self):
        if self.current < self.total:
            print()


def batch_iterator(
    items: List[Any],
    batch_size: int = 100,
) -> Generator[List[Any], None, None]:
    """バッチイテレータ"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class BatchProcessor:
    """
    バッチ処理エンジン
    
    Usage:
        processor = BatchProcessor()
        result = processor.process(
            items=smiles_list,
            func=extract_features,
            batch_size=100,
        )
    """
    
    def __init__(
        self,
        n_workers: int = 4,
        show_progress: bool = True,
    ):
        self.n_workers = n_workers
        self.show_progress = show_progress
    
    def process(
        self,
        items: List[Any],
        func: Callable[[Any], Any],
        batch_size: int = 100,
        error_handler: Callable[[Exception, Any], Any] = None,
    ) -> BatchResult:
        """
        バッチ処理を実行
        
        Args:
            items: 処理対象リスト
            func: 各アイテムに適用する関数
            batch_size: バッチサイズ
            error_handler: エラーハンドラ
            
        Returns:
            BatchResult
        """
        start_time = time.time()
        results = []
        errors = []
        
        progress = None
        if self.show_progress:
            progress = ProgressCallback(len(items))
        
        for batch in batch_iterator(items, batch_size):
            batch_results = self._process_batch(
                batch, func, error_handler
            )
            
            for item, result, error in batch_results:
                if error is None:
                    results.append(result)
                else:
                    errors.append({'item': item, 'error': str(error)})
                
                if progress:
                    progress.update()
        
        if progress:
            progress.close()
        
        elapsed = time.time() - start_time
        
        return BatchResult(
            total=len(items),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            elapsed_time=elapsed,
        )
    
    def _process_batch(
        self,
        batch: List[Any],
        func: Callable,
        error_handler: Callable = None,
    ) -> List[tuple]:
        """単一バッチを処理"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(func, item): item for item in batch}
            
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    results.append((item, result, None))
                except Exception as e:
                    if error_handler:
                        try:
                            result = error_handler(e, item)
                            results.append((item, result, None))
                        except Exception as e2:
                            results.append((item, None, e2))
                    else:
                        results.append((item, None, e))
        
        return results
    
    def process_sequential(
        self,
        items: List[Any],
        func: Callable[[Any], Any],
    ) -> BatchResult:
        """逐次処理（デバッグ用）"""
        start_time = time.time()
        results = []
        errors = []
        
        progress = ProgressCallback(len(items)) if self.show_progress else None
        
        for item in items:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                errors.append({'item': item, 'error': str(e)})
            
            if progress:
                progress.update()
        
        if progress:
            progress.close()
        
        return BatchResult(
            total=len(items),
            successful=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            elapsed_time=time.time() - start_time,
        )


class BatchFeatureExtractor:
    """
    バッチ特徴量抽出
    
    Usage:
        extractor = BatchFeatureExtractor()
        df = extractor.extract(smiles_list)
    """
    
    def __init__(
        self,
        feature_engine=None,
        batch_size: int = 100,
        n_workers: int = 4,
    ):
        self.feature_engine = feature_engine
        self.batch_size = batch_size
        self.n_workers = n_workers
    
    def extract(
        self,
        smiles_list: List[str],
        target_property: str = 'general',
    ) -> pd.DataFrame:
        """バッチで特徴量を抽出"""
        from core.services.features import SmartFeatureEngine
        
        engine = self.feature_engine or SmartFeatureEngine(target_property)
        
        all_features = []
        processor = BatchProcessor(n_workers=1, show_progress=True)
        
        for batch in batch_iterator(smiles_list, self.batch_size):
            try:
                result = engine.fit_transform(batch)
                all_features.append(result.features)
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                # 個別処理にフォールバック
                for smi in batch:
                    try:
                        result = engine.fit_transform([smi])
                        all_features.append(result.features)
                    except Exception:
                        pass
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        return pd.DataFrame()


class BatchPredictor:
    """
    バッチ予測
    
    Usage:
        predictor = BatchPredictor(model)
        predictions = predictor.predict(X)
    """
    
    def __init__(
        self,
        model,
        batch_size: int = 1000,
    ):
        self.model = model
        self.batch_size = batch_size
    
    def predict(
        self,
        X: pd.DataFrame,
        return_uncertainty: bool = False,
    ) -> np.ndarray:
        """バッチ予測"""
        predictions = []
        
        for i in range(0, len(X), self.batch_size):
            batch_X = X.iloc[i:i + self.batch_size]
            
            if return_uncertainty and hasattr(self.model, 'predict_with_interval'):
                mean, lower, upper = self.model.predict_with_interval(batch_X)
                predictions.append(mean)
            else:
                pred = self.model.predict(batch_X)
                predictions.append(pred)
        
        return np.concatenate(predictions)


def run_batch_extraction(
    smiles_list: List[str],
    target_property: str = 'general',
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    便利関数: バッチ特徴量抽出
    """
    extractor = BatchFeatureExtractor(batch_size=batch_size)
    return extractor.extract(smiles_list, target_property)


def run_batch_process(
    items: List[Any],
    func: Callable,
    batch_size: int = 100,
) -> BatchResult:
    """
    便利関数: 汎用バッチ処理
    """
    processor = BatchProcessor()
    return processor.process(items, func, batch_size)
