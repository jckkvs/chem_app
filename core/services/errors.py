"""
エラーハンドリング

Implements: F-ERROR-001
設計思想:
- カスタム例外
- エラーコード
- リカバリー
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ================== カスタム例外 ==================

class ChemMLError(Exception):
    """基底例外クラス"""
    
    def __init__(self, message: str, code: str = "ERR_UNKNOWN"):
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")


class SMILESError(ChemMLError):
    """SMILES関連エラー"""
    
    def __init__(self, smiles: str, message: str = "Invalid SMILES"):
        super().__init__(f"{message}: {smiles}", "ERR_SMILES")
        self.smiles = smiles


class FeatureExtractionError(ChemMLError):
    """特徴量抽出エラー"""
    
    def __init__(self, message: str):
        super().__init__(message, "ERR_FEATURE")


class ModelError(ChemMLError):
    """モデル関連エラー"""
    
    def __init__(self, message: str):
        super().__init__(message, "ERR_MODEL")


class ConfigError(ChemMLError):
    """設定エラー"""
    
    def __init__(self, message: str):
        super().__init__(message, "ERR_CONFIG")


class DataError(ChemMLError):
    """データエラー"""
    
    def __init__(self, message: str):
        super().__init__(message, "ERR_DATA")


# ================== エラーハンドラー ==================

class ErrorHandler:
    """
    エラーハンドラー
    
    Example:
        >>> handler = ErrorHandler()
        >>> with handler.catch():
        ...     risky_operation()
    """
    
    def __init__(self, raise_errors: bool = True, log_errors: bool = True):
        self.raise_errors = raise_errors
        self.log_errors = log_errors
        self.last_error: Optional[Exception] = None
    
    class catch:
        """エラーキャッチコンテキスト"""
        
        def __init__(self, handler: 'ErrorHandler' = None, default: Any = None):
            self.handler = handler or ErrorHandler()
            self.default = default
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val:
                self.handler.last_error = exc_val
                
                if self.handler.log_errors:
                    logger.error(f"Caught error: {exc_val}")
                
                if self.handler.raise_errors:
                    return False  # 再raise
                return True  # 抑制
            return False
    
    def safe_call(self, func, *args, default=None, **kwargs) -> Any:
        """安全な関数呼び出し"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.last_error = e
            if self.log_errors:
                logger.error(f"Error in {func.__name__}: {e}")
            if self.raise_errors:
                raise
            return default


# ================== リカバリー ==================

def retry(func, max_attempts: int = 3, delay: float = 1.0):
    """リトライデコレータ"""
    import time
    
    def wrapper(*args, **kwargs):
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
        
        raise last_error
    
    return wrapper


def fallback(*functions):
    """フォールバックチェーン"""
    def wrapper(*args, **kwargs):
        for func in functions:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"Fallback from {func.__name__}: {e}")
                continue
        raise ChemMLError("All fallbacks failed", "ERR_FALLBACK")
    
    return wrapper
