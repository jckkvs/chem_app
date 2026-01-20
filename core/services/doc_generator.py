"""
自動ドキュメント生成

Implements: F-AUTODOC-001
設計思想:
- コードからドキュメント生成
- API仕様書
- 使用例生成
"""

from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FunctionDoc:
    """関数ドキュメント"""
    name: str
    signature: str
    docstring: str
    parameters: Dict[str, str]
    returns: str
    example: str = ""


@dataclass
class ClassDoc:
    """クラスドキュメント"""
    name: str
    docstring: str
    methods: List[FunctionDoc]
    attributes: List[str]


class DocumentationGenerator:
    """
    自動ドキュメント生成
    
    Features:
    - docstringからMarkdown生成
    - API仕様書生成
    - HTML出力
    
    Example:
        >>> gen = DocumentationGenerator()
        >>> gen.add_module(my_module)
        >>> gen.generate_markdown("docs/api.md")
    """
    
    def __init__(self):
        self.classes: List[ClassDoc] = []
        self.functions: List[FunctionDoc] = []
    
    def add_function(self, func: Callable) -> FunctionDoc:
        """関数を追加"""
        sig = str(inspect.signature(func))
        docstring = inspect.getdoc(func) or ""
        
        # パラメータ解析
        params = {}
        for line in docstring.split('\n'):
            if line.strip().startswith(':param'):
                parts = line.split(':')
                if len(parts) >= 3:
                    param_name = parts[1].replace('param', '').strip()
                    param_desc = ':'.join(parts[2:]).strip()
                    params[param_name] = param_desc
        
        doc = FunctionDoc(
            name=func.__name__,
            signature=sig,
            docstring=docstring,
            parameters=params,
            returns="",
        )
        self.functions.append(doc)
        return doc
    
    def add_class(self, cls: type) -> ClassDoc:
        """クラスを追加"""
        docstring = inspect.getdoc(cls) or ""
        
        methods = []
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_'):
                methods.append(self.add_function(method))
        
        attributes = [
            name for name in dir(cls)
            if not name.startswith('_') and not callable(getattr(cls, name))
        ]
        
        doc = ClassDoc(
            name=cls.__name__,
            docstring=docstring,
            methods=methods,
            attributes=attributes,
        )
        self.classes.append(doc)
        return doc
    
    def add_module(self, module) -> None:
        """モジュールを追加"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                self.add_class(obj)
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                self.add_function(obj)
    
    def generate_markdown(self, output_path: Optional[str] = None) -> str:
        """Markdownを生成"""
        lines = ["# API Reference\n"]
        
        if self.classes:
            lines.append("## Classes\n")
            for cls in self.classes:
                lines.append(f"### {cls.name}\n")
                lines.append(f"{cls.docstring}\n")
                
                if cls.methods:
                    lines.append("#### Methods\n")
                    for method in cls.methods:
                        lines.append(f"##### `{method.name}{method.signature}`\n")
                        lines.append(f"{method.docstring}\n")
        
        if self.functions:
            lines.append("## Functions\n")
            for func in self.functions:
                lines.append(f"### `{func.name}{func.signature}`\n")
                lines.append(f"{func.docstring}\n")
        
        content = '\n'.join(lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return content
    
    def generate_html(self, output_path: Optional[str] = None) -> str:
        """HTMLを生成"""
        md_content = self.generate_markdown()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>API Documentation</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; border-bottom: 2px solid #3498db; }}
        h3 {{ color: #27ae60; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <pre>{md_content}</pre>
</body>
</html>
"""
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html
