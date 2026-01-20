"""
ロギング設定

Implements: F-LOG-001
設計思想:
- 構造化ログ
- ファイル/コンソール出力
- ログレベル設定
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    format_style: str = "simple",
) -> None:
    """
    ロギングを設定
    
    Args:
        level: ログレベル (DEBUG, INFO, WARNING, ERROR)
        log_file: ログファイルパス（Noneでコンソールのみ）
        format_style: フォーマット (simple, detailed, json)
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # フォーマット
    if format_style == "detailed":
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    elif format_style == "json":
        fmt = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    
    handlers = []
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    handlers.append(console_handler)
    
    # ファイルハンドラ
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        handlers.append(file_handler)
    
    # ルートロガー設定
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )
    
    # サードパーティのログを抑制
    logging.getLogger("rdkit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """ロガーを取得"""
    return logging.getLogger(name)


class ExperimentLogger:
    """
    実験ロガー
    
    実験の進捗と結果を記録。
    
    Usage:
        logger = ExperimentLogger("exp_001")
        logger.log_start({"model": "rf", "samples": 1000})
        logger.log_metric("train_r2", 0.95)
        logger.log_end()
    """
    
    def __init__(self, experiment_id: str, log_dir: str = "logs"):
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger = logging.getLogger(f"experiment.{experiment_id}")
        self._start_time: Optional[datetime] = None
        self._metrics = {}
    
    def log_start(self, params: dict = None) -> None:
        """実験開始をログ"""
        self._start_time = datetime.now()
        self._logger.info(f"Experiment started: {self.experiment_id}")
        if params:
            self._logger.info(f"Parameters: {params}")
    
    def log_metric(self, name: str, value: float) -> None:
        """メトリクスをログ"""
        self._metrics[name] = value
        self._logger.info(f"Metric {name}: {value:.4f}")
    
    def log_message(self, message: str, level: str = "info") -> None:
        """メッセージをログ"""
        getattr(self._logger, level.lower())(message)
    
    def log_end(self) -> None:
        """実験終了をログ"""
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
            self._logger.info(f"Experiment completed in {duration:.1f}s")
            self._logger.info(f"Final metrics: {self._metrics}")
    
    def save_summary(self) -> None:
        """サマリーを保存"""
        import json
        
        summary = {
            "experiment_id": self.experiment_id,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": datetime.now().isoformat(),
            "metrics": self._metrics,
        }
        
        filepath = self.log_dir / f"{self.experiment_id}_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
