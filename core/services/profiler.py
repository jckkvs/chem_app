"""
パフォーマンスプロファイラー

Implements: F-PROFILE-001
設計思想:
- 実行時間計測
- メモリ使用量
- ボトルネック検出
"""

from __future__ import annotations

import logging
import time
import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """プロファイル結果"""
    name: str
    duration_seconds: float
    calls: int = 1
    memory_mb: Optional[float] = None


class Profiler:
    """
    パフォーマンスプロファイラー
    
    Features:
    - 実行時間計測
    - 複数計測の集計
    - レポート生成
    
    Example:
        >>> profiler = Profiler()
        >>> with profiler.measure("feature_extraction"):
        ...     extract_features(data)
        >>> profiler.report()
    """
    
    def __init__(self):
        self.results: Dict[str, ProfileResult] = {}
        self._start_times: Dict[str, float] = {}
    
    @contextmanager
    def measure(self, name: str):
        """計測コンテキスト"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._record(name, duration)
    
    def _record(self, name: str, duration: float) -> None:
        """結果を記録"""
        if name in self.results:
            existing = self.results[name]
            self.results[name] = ProfileResult(
                name=name,
                duration_seconds=existing.duration_seconds + duration,
                calls=existing.calls + 1,
            )
        else:
            self.results[name] = ProfileResult(
                name=name,
                duration_seconds=duration,
                calls=1,
            )
    
    def start(self, name: str) -> None:
        """計測開始"""
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """計測終了"""
        if name not in self._start_times:
            return 0.0
        
        duration = time.perf_counter() - self._start_times[name]
        self._record(name, duration)
        del self._start_times[name]
        return duration
    
    def report(self) -> str:
        """レポート生成"""
        lines = ["Performance Report", "=" * 50]
        
        total = sum(r.duration_seconds for r in self.results.values())
        
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.duration_seconds,
            reverse=True,
        )
        
        for r in sorted_results:
            pct = (r.duration_seconds / total * 100) if total > 0 else 0
            avg = r.duration_seconds / r.calls
            lines.append(
                f"{r.name}: {r.duration_seconds:.3f}s ({pct:.1f}%) "
                f"[{r.calls} calls, avg {avg:.3f}s]"
            )
        
        lines.append(f"\nTotal: {total:.3f}s")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """リセット"""
        self.results.clear()
        self._start_times.clear()


def profiled(name: Optional[str] = None):
    """プロファイルデコレータ"""
    _profiler = Profiler()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            with _profiler.measure(func_name):
                result = func(*args, **kwargs)
            logger.debug(f"{func_name}: {_profiler.results[func_name].duration_seconds:.3f}s")
            return result
        wrapper._profiler = _profiler
        return wrapper
    return decorator


# グローバルプロファイラー
global_profiler = Profiler()
