"""
可視化機能 基底クラス

新しい可視化機能を追加する際は、このクラスを継承してください。

Implements: F-VIS-BASE-001
設計思想:
- Strategy Patternによる可視化の切り替え
- 複数出力形式対応（PNG/SVG/HTML/JSON）
- 設定の柔軟性
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Literal
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

OutputFormat = Literal['png', 'svg', 'html', 'json', 'plotly']


class BaseVisualizer(ABC):
    """
    可視化機能の抽象基底クラス
    
    統一インターフェースで異なる可視化を扱う。
    
    Subclasses:
    - SHAPVisualizer: SHAP説明
    - PDPVisualizer: Partial Dependence Plot
    - MoleculeVisualizer: 分子構造表示
    - ChemSpaceVisualizer: 化学空間マップ
    
    Example:
        >>> viz = SHAPVisualizer()
        >>> fig = viz.plot(model, X, y)
        >>> viz.save(fig, 'shap_plot.png')
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: 可視化固有の設定
        """
        self.config = kwargs
        self._default_format: OutputFormat = 'png'
    
    @abstractmethod
    def plot(self, *args, **kwargs) -> Any:
        """
        プロットを生成
        
        Args:
            *args: プロット用データ
            **kwargs: プロット設定
            
        Returns:
            図オブジェクト（matplotlib.Figure, plotly.Figure等）
        """
        pass
    
    def save(
        self,
        fig: Any,
        path: str | Path,
        format: Optional[OutputFormat] = None,
        **kwargs
    ) -> None:
        """
        図を保存
        
        Args:
            fig: 図オブジェクト
            path: 保存先パス
            format: 出力形式（自動判定可能）
            **kwargs: 保存オプション（dpi, width, heightなど）
        """
        path = Path(path)
        
        # フォーマット自動判定
        if format is None:
            format = self._infer_format(path)
        
        # フォーマット別保存
        if format == 'png':
            self._save_png(fig, path, **kwargs)
        elif format == 'svg':
            self._save_svg(fig, path, **kwargs)
        elif format == 'html':
            self._save_html(fig, path, **kwargs)
        elif format == 'json':
            self._save_json(fig, path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved visualization to {path}")
    
    def _infer_format(self, path: Path) -> OutputFormat:
        """拡張子からフォーマットを推定"""
        suffix = path.suffix.lower()
        format_map = {
            '.png': 'png',
            '.svg': 'svg',
            '.html': 'html',
            '.json': 'json',
        }
        return format_map.get(suffix, self._default_format)
    
    def _save_png(self, fig: Any, path: Path, dpi: int = 300, **kwargs) -> None:
        """PNG形式で保存（matplotlibの場合）"""
        if hasattr(fig, 'savefig'):
            fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
        else:
            raise NotImplementedError("PNG export not supported for this figure type")
    
    def _save_svg(self, fig: Any, path: Path, **kwargs) -> None:
        """SVG形式で保存"""
        if hasattr(fig, 'savefig'):
            fig.savefig(path, format='svg', **kwargs)
        else:
            raise NotImplementedError("SVG export not supported for this figure type")
    
    def _save_html(self, fig: Any, path: Path, **kwargs) -> None:
        """HTML形式で保存（Plotlyの場合）"""
        if hasattr(fig, 'write_html'):
            fig.write_html(path, **kwargs)
        else:
            raise NotImplementedError("HTML export not supported for this figure type")
    
    def _save_json(self, fig: Any, path: Path, **kwargs) -> None:
        """JSON形式で保存（Plotlyの場合）"""
        import json
        if hasattr(fig, 'to_json'):
            with open(path, 'w') as f:
                f.write(fig.to_json())
        else:
            raise NotImplementedError("JSON export not supported for this figure type")
    
    def to_base64(self, fig: Any, format: OutputFormat = 'png') -> str:
        """
        Base64エンコード文字列を生成（Web表示用）
        
        Args:
            fig: 図オブジェクト
            format: 出力形式
            
        Returns:
            Base64エンコード文字列
        """
        import io
        import base64
        
        buffer = io.BytesIO()
        
        if format == 'png':
            if hasattr(fig, 'savefig'):
                fig.savefig(buffer, format='png', bbox_inches='tight')
            else:
                raise NotImplementedError()
        elif format == 'svg':
            if hasattr(fig, 'savefig'):
                fig.savefig(buffer, format='svg')
            else:
                raise NotImplementedError()
        else:
            raise ValueError(f"Format {format} not supported for base64 encoding")
        
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def get_config(self) -> Dict[str, Any]:
        """設定を取得"""
        return self.config.copy()
    
    def set_config(self, **kwargs) -> 'BaseVisualizer':
        """
        設定を更新
        
        Args:
            **kwargs: 更新する設定
            
        Returns:
            self
        """
        self.config.update(kwargs)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
