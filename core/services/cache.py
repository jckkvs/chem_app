"""
キャッシュマネージャー

Implements: F-CACHE-001
設計思想:
- メモリ/ディスクキャッシュ
- TTL管理
- LRU eviction
"""

from __future__ import annotations

import logging
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    キャッシュマネージャー
    
    Features:
    - メモリキャッシュ（LRU）
    - ディスクキャッシュ
    - TTL管理
    
    Example:
        >>> cache = CacheManager()
        >>> cache.set("key", value, ttl=3600)
        >>> value = cache.get("key")
    """
    
    def __init__(
        self,
        cache_dir: str = ".cache",
        max_memory_items: int = 1000,
        default_ttl: int = 3600,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_items = max_memory_items
        self.default_ttl = default_ttl
        
        self._memory_cache: Dict[str, tuple] = {}  # key -> (value, expiry)
    
    def _hash_key(self, key: str) -> str:
        """キーをハッシュ化"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        disk: bool = False,
    ) -> None:
        """キャッシュに保存"""
        expiry = time.time() + (ttl or self.default_ttl)
        
        if disk:
            self._set_disk(key, value, expiry)
        else:
            self._set_memory(key, value, expiry)
    
    def _set_memory(self, key: str, value: Any, expiry: float) -> None:
        """メモリキャッシュに保存"""
        if len(self._memory_cache) >= self.max_memory_items:
            self._evict_oldest()
        
        self._memory_cache[key] = (value, expiry)
    
    def _set_disk(self, key: str, value: Any, expiry: float) -> None:
        """ディスクキャッシュに保存"""
        hash_key = self._hash_key(key)
        path = self.cache_dir / f"{hash_key}.json"
        
        try:
            with open(path, 'w') as f:
                json.dump({
                    'key': key,
                    'value': value,
                    'expiry': expiry,
                }, f)
        except Exception as e:
            logger.warning(f"Disk cache write failed: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """キャッシュから取得"""
        # メモリから
        if key in self._memory_cache:
            value, expiry = self._memory_cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._memory_cache[key]
        
        # ディスクから
        return self._get_disk(key, default)
    
    def _get_disk(self, key: str, default: Any) -> Any:
        """ディスクキャッシュから取得"""
        hash_key = self._hash_key(key)
        path = self.cache_dir / f"{hash_key}.json"
        
        if not path.exists():
            return default
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            if time.time() < data['expiry']:
                return data['value']
            else:
                path.unlink()
                return default
                
        except Exception:
            return default
    
    def _evict_oldest(self) -> None:
        """最古のエントリを削除"""
        if not self._memory_cache:
            return
        
        oldest_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k][1]
        )
        del self._memory_cache[oldest_key]
    
    def clear(self) -> None:
        """全キャッシュクリア"""
        self._memory_cache.clear()
        
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
    
    def stats(self) -> Dict[str, int]:
        """統計情報"""
        disk_files = list(self.cache_dir.glob("*.json"))
        return {
            'memory_items': len(self._memory_cache),
            'disk_items': len(disk_files),
        }


def cached(ttl: int = 3600):
    """キャッシュデコレータ"""
    _cache = {}
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            
            if key in _cache:
                value, expiry = _cache[key]
                if time.time() < expiry:
                    return value
            
            result = func(*args, **kwargs)
            _cache[key] = (result, time.time() + ttl)
            return result
        
        return wrapper
    return decorator
