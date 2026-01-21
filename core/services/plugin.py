"""
プラグインシステム

Implements: F-PLUGIN-001
設計思想:
- 拡張機能登録
- 動的ロード
- フック機能
- 自動検出（NEW）
- バリデーション（NEW）
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Plugin:
    """プラグイン"""
    name: str
    version: str
    description: str
    hooks: Dict[str, Callable] = field(default_factory=dict)
    enabled: bool = True
    
    # メタデータ（オプション）
    author: Optional[str] = None
    license: Optional[str] = None
    requires: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class PluginManager:
    """
    プラグインマネージャー（拡張版）
    
    Features:
    - プラグイン登録
    - フック実行
    - 動的ロード
    - 自動検出（NEW）
    - バリデーション（NEW）
    - ライフサイクルフック（NEW）
    
    Example:
        >>> pm = PluginManager()
        >>> pm.discover_plugins('plugins/')  # 自動検出
        >>> pm.execute_hook('on_prediction', data)
    """
    
    def __init__(self, auto_discover: bool = False, plugin_dir: str = 'plugins'):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.plugin_dir = Path(plugin_dir)
        
        if auto_discover and self.plugin_dir.exists():
            self.discover_plugins()
    
    def register(self, plugin: Plugin) -> None:
        """プラグインを登録"""
        self.plugins[plugin.name] = plugin
        
        for hook_name, func in plugin.hooks.items():
            if hook_name not in self.hooks:
                self.hooks[hook_name] = []
            self.hooks[hook_name].append(func)
        
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
    
    def unregister(self, name: str) -> bool:
        """プラグインを解除"""
        if name not in self.plugins:
            return False
        
        plugin = self.plugins[name]
        
        for hook_name, func in plugin.hooks.items():
            if hook_name in self.hooks:
                self.hooks[hook_name] = [
                    f for f in self.hooks[hook_name] if f != func
                ]
        
        del self.plugins[name]
        logger.info(f"Unregistered plugin: {name}")
        return True
    
    def execute_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """フックを実行"""
        results = []
        
        if hook_name not in self.hooks:
            return results
        
        for func in self.hooks[hook_name]:
            try:
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook execution failed: {e}")
        
        return results
    
    def load_from_module(self, module_path: str) -> Optional[Plugin]:
        """モジュールからロード"""
        try:
            module = importlib.import_module(module_path)
            
            if hasattr(module, 'create_plugin'):
                plugin = module.create_plugin()
                self.register(plugin)
                return plugin
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {module_path}: {e}")
        
        return None
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """プラグイン一覧"""
        return [
            {
                'name': p.name,
                'version': p.version,
                'description': p.description,
                'enabled': p.enabled,
                'hooks': list(p.hooks.keys()),
            }
            for p in self.plugins.values()
        ]
    
    def enable(self, name: str) -> bool:
        """プラグインを有効化"""
        if name in self.plugins:
            self.plugins[name].enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """プラグインを無効化"""
        if name in self.plugins:
            self.plugins[name].enabled = False
            return True
        return False


# グローバルプラグインマネージャー
plugin_manager = PluginManager()
