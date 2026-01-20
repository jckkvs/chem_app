"""
設定管理（Hydra/OmegaConf inspired）

Implements: F-CONFIG-001
設計思想:
- 階層的設定
- 環境変数統合
- 設定検証
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特徴量設定"""
    rdkit_enabled: bool = True
    xtb_enabled: bool = False
    uma_enabled: bool = False
    tarte_enabled: bool = False  # TARTE (Transformer tabular features)
    tarte_mode: str = "featurizer"  # "featurizer" | "finetuning" | "boosting"
    fingerprint_type: str = "morgan"
    fingerprint_bits: int = 2048


@dataclass
class ModelConfig:
    """モデル設定"""
    model_type: str = "lightgbm"
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42


@dataclass
class TrainingConfig:
    """訓練設定"""
    test_size: float = 0.2
    cv_folds: int = 5
    early_stopping: bool = True
    patience: int = 10


@dataclass
class AppConfig:
    """アプリケーション設定"""
    debug: bool = False
    log_level: str = "INFO"
    mlflow_tracking_uri: str = "mlruns"
    checkpoint_dir: str = "checkpoints"
    
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


class ConfigManager:
    """
    設定管理（Hydra inspired）
    
    Features:
    - JSON/YAML設定読み込み
    - 環境変数オーバーライド
    - 設定検証
    
    Example:
        >>> config = ConfigManager.load("config.json")
        >>> print(config.model.n_estimators)
    """
    
    @staticmethod
    def load(filepath: Optional[str] = None) -> AppConfig:
        """設定を読み込み"""
        config = AppConfig()
        
        if filepath and Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                config = ConfigManager._from_dict(data)
            except Exception as e:
                logger.warning(f"Config load failed: {e}")
        
        # 環境変数オーバーライド
        config = ConfigManager._apply_env_overrides(config)
        
        return config
    
    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> AppConfig:
        """辞書から設定を構築"""
        features = FeatureConfig(**data.get('features', {}))
        model = ModelConfig(**data.get('model', {}))
        training = TrainingConfig(**data.get('training', {}))
        
        return AppConfig(
            debug=data.get('debug', False),
            log_level=data.get('log_level', 'INFO'),
            mlflow_tracking_uri=data.get('mlflow_tracking_uri', 'mlruns'),
            checkpoint_dir=data.get('checkpoint_dir', 'checkpoints'),
            features=features,
            model=model,
            training=training,
        )
    
    @staticmethod
    def _apply_env_overrides(config: AppConfig) -> AppConfig:
        """環境変数を適用"""
        if os.getenv('CHEMML_DEBUG'):
            config.debug = True
        
        if os.getenv('CHEMML_LOG_LEVEL'):
            config.log_level = os.getenv('CHEMML_LOG_LEVEL')
        
        if os.getenv('CHEMML_MODEL_TYPE'):
            config.model.model_type = os.getenv('CHEMML_MODEL_TYPE')
        
        return config
    
    @staticmethod
    def save(config: AppConfig, filepath: str) -> None:
        """設定を保存"""
        data = asdict(config)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def validate(config: AppConfig) -> bool:
        """設定を検証"""
        errors = []
        
        if config.model.n_estimators < 1:
            errors.append("n_estimators must be >= 1")
        
        if config.training.test_size <= 0 or config.training.test_size >= 1:
            errors.append("test_size must be between 0 and 1")
        
        if config.training.cv_folds < 2:
            errors.append("cv_folds must be >= 2")
        
        if errors:
            for e in errors:
                logger.error(f"Config validation: {e}")
            return False
        
        return True
    
    @staticmethod
    def get_default() -> AppConfig:
        """デフォルト設定を取得"""
        return AppConfig()
