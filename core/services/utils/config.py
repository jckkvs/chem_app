"""
設定管理

Implements: F-CONFIG-001
設計思想:
- YAML/JSON設定ファイル
- 環境変数オーバーライド
- デフォルト値管理
- 設定の検証

機能:
- 設定のロード/保存
- 環境別設定（dev/prod）
- 設定値の型検証
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# YAML対応（オプショナル）
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class FeatureConfig:
    """特徴量設定"""
    default_preset: str = "general"
    use_pretrained: bool = False
    pretrained_models: list = field(default_factory=lambda: ["chemberta"])
    morgan_fp_bits: int = 1024
    morgan_radius: int = 2
    enable_3d: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """モデル設定"""
    default_model: str = "random_forest"
    n_estimators: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    enable_uncertainty: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatabaseConfig:
    """データベース設定"""
    db_path: str = "molecules.db"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AppConfig:
    """アプリケーション設定"""
    name: str = "ChemML"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # サブ設定
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # カスタム設定
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "debug": self.debug,
            "log_level": self.log_level,
            "features": self.features.to_dict(),
            "model": self.model.to_dict(),
            "database": self.database.to_dict(),
            "custom": self.custom,
        }


class ConfigManager:
    """
    設定管理クラス
    
    Usage:
        config = ConfigManager()
        config.load("config.yaml")
        
        # 設定値にアクセス
        print(config.features.default_preset)
        
        # 環境変数でオーバーライド
        # CHEMML_DEBUG=true
        print(config.app.debug)
    """
    
    ENV_PREFIX = "CHEMML_"
    
    def __init__(self, config_path: str = None):
        self._config = AppConfig()
        
        if config_path:
            self.load(config_path)
        
        self._apply_env_overrides()
    
    @property
    def app(self) -> AppConfig:
        return self._config
    
    @property
    def features(self) -> FeatureConfig:
        return self._config.features
    
    @property
    def model(self) -> ModelConfig:
        return self._config.model
    
    @property
    def database(self) -> DatabaseConfig:
        return self._config.database
    
    def load(self, path: str) -> None:
        """設定ファイルをロード"""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return
        
        if path.suffix in ['.yaml', '.yml']:
            self._load_yaml(path)
        elif path.suffix == '.json':
            self._load_json(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        logger.info(f"Loaded config from {path}")
    
    def _load_yaml(self, path: Path) -> None:
        """YAMLをロード"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML config")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._apply_dict(data)
    
    def _load_json(self, path: Path) -> None:
        """JSONをロード"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._apply_dict(data)
    
    def _apply_dict(self, data: Dict[str, Any]) -> None:
        """辞書から設定を適用"""
        if not data:
            return
        
        # トップレベル
        if 'name' in data:
            self._config.name = data['name']
        if 'version' in data:
            self._config.version = data['version']
        if 'debug' in data:
            self._config.debug = data['debug']
        if 'log_level' in data:
            self._config.log_level = data['log_level']
        
        # Features
        if 'features' in data:
            f = data['features']
            self._config.features = FeatureConfig(
                default_preset=f.get('default_preset', 'general'),
                use_pretrained=f.get('use_pretrained', False),
                pretrained_models=f.get('pretrained_models', ['chemberta']),
                morgan_fp_bits=f.get('morgan_fp_bits', 1024),
                morgan_radius=f.get('morgan_radius', 2),
                enable_3d=f.get('enable_3d', False),
            )
        
        # Model
        if 'model' in data:
            m = data['model']
            self._config.model = ModelConfig(
                default_model=m.get('default_model', 'random_forest'),
                n_estimators=m.get('n_estimators', 100),
                cv_folds=m.get('cv_folds', 5),
                test_size=m.get('test_size', 0.2),
                random_state=m.get('random_state', 42),
                enable_uncertainty=m.get('enable_uncertainty', True),
            )
        
        # Database
        if 'database' in data:
            d = data['database']
            self._config.database = DatabaseConfig(
                db_path=d.get('db_path', 'molecules.db'),
                auto_backup=d.get('auto_backup', True),
                backup_interval_hours=d.get('backup_interval_hours', 24),
            )
        
        # Custom
        if 'custom' in data:
            self._config.custom = data['custom']
    
    def _apply_env_overrides(self) -> None:
        """環境変数でオーバーライド"""
        env_mappings = {
            'DEBUG': ('debug', bool),
            'LOG_LEVEL': ('log_level', str),
            'DB_PATH': ('database.db_path', str),
            'DEFAULT_PRESET': ('features.default_preset', str),
            'DEFAULT_MODEL': ('model.default_model', str),
        }
        
        for env_key, (config_path, type_fn) in env_mappings.items():
            full_key = f"{self.ENV_PREFIX}{env_key}"
            value = os.environ.get(full_key)
            
            if value is not None:
                self._set_nested(config_path, self._convert_value(value, type_fn))
                logger.debug(f"Override from env: {full_key}")
    
    def _convert_value(self, value: str, type_fn: type) -> Any:
        """環境変数の値を型変換"""
        if type_fn == bool:
            return value.lower() in ('true', '1', 'yes')
        return type_fn(value)
    
    def _set_nested(self, path: str, value: Any) -> None:
        """ネストした設定値を設定"""
        parts = path.split('.')
        obj = self._config
        
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        setattr(obj, parts[-1], value)
    
    def save(self, path: str) -> None:
        """設定を保存"""
        path = Path(path)
        data = self._config.to_dict()
        
        if path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required")
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved config to {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得（ドット記法）"""
        parts = key.split('.')
        obj = self._config
        
        try:
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """設定値を設定（ドット記法）"""
        self._set_nested(key, value)


# グローバル設定インスタンス
_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """グローバル設定を取得"""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


def load_config(path: str) -> ConfigManager:
    """設定をロード"""
    global _config
    _config = ConfigManager(path)
    return _config


def create_default_config(path: str = "config.json") -> None:
    """デフォルト設定ファイルを作成"""
    config = ConfigManager()
    config.save(path)
    print(f"Created default config: {path}")
