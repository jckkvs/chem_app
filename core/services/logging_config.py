"""
ログ設定モジュール

Implements: F-LOG-001
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_style: str = "detailed",
) -> logging.Logger:
    """
    ログ設定
    
    Args:
        level: ログレベル
        log_file: ログファイルパス
        format_style: 'simple' or 'detailed'
    """
    if format_style == "detailed":
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    else:
        fmt = "%(levelname)s: %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
        handlers=handlers,
    )
    
    return logging.getLogger("chemml")


def get_logger(name: str) -> logging.Logger:
    """ロガー取得"""
    return logging.getLogger(name)


class ProgressLogger:
    """プログレスログ"""
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.logger = get_logger("progress")
    
    def update(self, n: int = 1) -> None:
        self.current += n
        pct = (self.current / self.total) * 100
        self.logger.info(f"{self.name}: {self.current}/{self.total} ({pct:.1f}%)")
    
    def finish(self) -> None:
        self.logger.info(f"{self.name}: Complete!")
