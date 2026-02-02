"""
CacheManagerのテスト

カバレッジ目標: 85%以上（83 stmts）
"""
import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.services.cache import CacheManager, cached


class TestCacheManagerBasic:
    """基本機能テスト"""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir, default_ttl=10)
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_init(self):
        """初期化テスト"""
        assert self.cache.cache_dir == Path(self.tmpdir)
        assert self.cache.default_ttl == 10
        assert os.path.exists(self.tmpdir)
    
    def test_hash_key(self):
        """キーハッシュ化テスト"""
        hash1 = self.cache._hash_key("test_key")
        hash2 = self.cache._hash_key("test_key")
        hash3 = self.cache._hash_key("other_key")
        
        assert hash1 == hash2  # 同じキーは同じハッシュ
        assert hash1 != hash3  # 異なるキーは異なるハッシュ
        assert len(hash1) == 32  # MD5ハッシュは32文字


class TestCacheManagerMemory:
    """メモリキャッシュテスト"""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir, default_ttl=10)
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_set_and_get_memory(self):
        """メモリキャッシュ設定・取得"""
        self.cache.set("key1", "value1", ttl=10, disk=False)
        
        result = self.cache.get("key1")
        assert result == "value1"
    
    def test_get_nonexistent_key(self):
        """存在しないキーの取得"""
        result = self.cache.get("nonexistent", default="default_value")
        assert result == "default_value"
    
    def test_memory_ttl_expiry(self):
        """TTL期限切れテスト"""
        self.cache.set("key1", "value1", ttl=0.1, disk=False)
        
        # 期限内
        result = self.cache.get("key1")
        assert result == "value1"
        
        # 期限切れ
        time.sleep(0.2)
        result = self.cache.get("key1", default="expired")
        assert result == "expired"
    
    def test_evict_oldest(self):
        """LRU evictionテスト"""
        # max_memory_items=2のキャッシュ作成
        cache = CacheManager(cache_dir=self.tmpdir, max_memory_items=2)
        
        cache.set("key1", "value1", ttl=100, disk=False)
        cache.set("key2", "value2", ttl=100, disk=False)
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        
        # 3つ目を追加すると最古が削除される
        cache.set("key3", "value3", ttl=100, disk=False)
        
        # メモリには2つだけ存在
        assert len(cache._memory_cache) == 2
    
    def test_evict_oldest_empty_cache(self):
        """空のキャッシュでevict"""
        self.cache._evict_oldest()
        # エラーにならないことを確認
        assert len(self.cache._memory_cache) == 0


class TestCacheManagerDisk:
    """ディスクキャッシュテスト"""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir, default_ttl=10)
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_set_and_get_disk(self):
        """ディスクキャッシュ設定・取得"""
        self.cache.set("key1", {"data": "value1"}, ttl=10, disk=True)
        
        # メモリには存在しない
        assert "key1" not in self.cache._memory_cache
        
        # ディスクから取得
        result = self.cache.get("key1")
        assert result == {"data": "value1"}
    
    def test_disk_ttl_expiry(self):
        """ディスクTTL期限切れテスト"""
        self.cache.set("key1", "value1", ttl=0.1, disk=True)
        
        # 期限内
        result = self.cache.get("key1")
        assert result == "value1"
        
        # 期限切れ
        time.sleep(0.2)
        result = self.cache.get("key1", default="expired")
        assert result == "expired"
    
    def test_disk_write_error(self):
        """ディスク書き込みエラー"""
        # open関数をモックしてエラーを発生させる
        with patch('builtins.open', side_effect=IOError("Disk full")):
            with patch('core.services.cache.logger') as mock_logger:
                self.cache.set("key1", "value1", disk=True)
                # warningが呼ばれることを確認
                assert mock_logger.warning.called
    
    def test_disk_read_corrupted(self):
        """破損したディスクキャッシュ読み込み"""
        # 破損したJSONファイルを作成
        hash_key = self.cache._hash_key("corrupted_key")
        path = self.cache.cache_dir / f"{hash_key}.json"
        
        with open(path, 'w') as f:
            f.write("INVALID JSON")
        
        # デフォルト値が返される
        result = self.cache.get("corrupted_key", default="default")
        assert result == "default"


class TestCacheManagerOperations:
    """キャッシュ操作テスト"""
    
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.tmpdir)
    
    def teardown_method(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
    
    def test_clear(self):
        """全クリアテスト"""
        # メモリキャッシュ
        self.cache.set("mem1", "value1", disk=False)
        self.cache.set("mem2", "value2", disk=False)
        
        # ディスクキャッシュ
        self.cache.set("disk1", "value3", disk=True)
        self.cache.set("disk2", "value4", disk=True)
        
        self.cache.clear()
        
        # 全て削除されている
        assert len(self.cache._memory_cache) == 0
        assert len(list(self.cache.cache_dir.glob("*.json"))) == 0
    
    def test_stats(self):
        """統計情報テスト"""
        # メモリに2個
        self.cache.set("mem1", "value1", disk=False)
        self.cache.set("mem2", "value2", disk=False)
        
        # ディスクに3個
        self.cache.set("disk1", "value3", disk=True)
        self.cache.set("disk2", "value4", disk=True)
        self.cache.set("disk3", "value5", disk=True)
        
        stats = self.cache.stats()
        assert stats['memory_items'] == 2
        assert stats['disk_items'] == 3


class TestCachedDecorator:
    """cachedデコレータテスト"""
    
    def test_cached_decorator(self):
        """デコレータ基本テスト"""
        call_count = 0
        
        @cached(ttl=10)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # 初回呼び出し
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # キャッシュから取得（関数は呼ばれない）
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1
        
        # 異なる引数（関数が呼ばれる）
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_cached_decorator_expiry(self):
        """デコレータTTL期限切れテスト"""
        call_count = 0
        
        @cached(ttl=0.1)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # 初回
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # 期限切れ後
        time.sleep(0.2)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 2  # 再計算
    
    def test_cached_decorator_kwargs(self):
        """デコレータkwargs対応テスト"""
        call_count = 0
        
        @cached(ttl=10)
        def func_with_kwargs(a, b=10):
            nonlocal call_count
            call_count += 1
            return a + b
        
        # キーワード引数を使用
        result1 = func_with_kwargs(5, b=20)
        assert result1 == 25
        assert call_count == 1
        
        # 同じキーワード引数
        result2 = func_with_kwargs(5, b=20)
        assert result2 == 25
        assert call_count == 1  # キャッシュから


class TestCacheManagerEdgeCases:
    """エッジケーステスト"""
    
    def test_default_ttl_usage(self):
        """デフォルトTTL使用"""
        cache = CacheManager(cache_dir=tempfile.mkdtemp(), default_ttl=100)
        
        # ttl指定なし
        cache.set("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
    
    def test_complex_value_types(self):
        """複雑な値の型テスト"""
        cache = CacheManager(cache_dir=tempfile.mkdtemp())
        
        # リスト
        cache.set("list", [1, 2, 3], disk=False)
        assert cache.get("list") == [1, 2, 3]
        
        # 辞書（ディスク）
        cache.set("dict", {"a": 1, "b": 2}, disk=True)
        assert cache.get("dict") == {"a": 1, "b": 2}
