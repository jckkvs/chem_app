"""
レポート生成エンジン

Implements: F-REPORT-001
設計思想:
- 実験結果の自動レポート生成
- HTML/PDF/Markdown対応
- 図表の自動埋め込み

機能:
- モデル性能サマリー
- 特徴量重要度
- 予測結果の可視化
- エクスポート機能
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """レポートセクション"""
    title: str
    content: str
    order: int = 0
    section_type: str = "text"  # text, table, figure, code


@dataclass
class ExperimentReport:
    """実験レポート"""
    title: str
    experiment_id: str
    created_at: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(
        self,
        title: str,
        content: str,
        section_type: str = "text",
    ) -> None:
        """セクションを追加"""
        order = len(self.sections)
        self.sections.append(ReportSection(
            title=title,
            content=content,
            order=order,
            section_type=section_type,
        ))


class ReportGenerator:
    """
    レポート生成エンジン
    
    Usage:
        gen = ReportGenerator()
        report = gen.create_experiment_report(
            experiment_name="Tg Prediction",
            y_true=y_true,
            y_pred=y_pred,
            feature_importances=importances,
        )
        gen.export_html(report, "report.html")
    """
    
    def create_experiment_report(
        self,
        experiment_name: str,
        y_true: np.ndarray = None,
        y_pred: np.ndarray = None,
        feature_importances: Dict[str, float] = None,
        smiles_list: List[str] = None,
        model_params: Dict[str, Any] = None,
        additional_metrics: Dict[str, float] = None,
    ) -> ExperimentReport:
        """実験レポートを生成"""
        import uuid
        
        report = ExperimentReport(
            title=f"Experiment Report: {experiment_name}",
            experiment_id=str(uuid.uuid4())[:8],
            created_at=datetime.now().isoformat(),
        )
        
        # 概要セクション
        overview = f"""
**Experiment**: {experiment_name}  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**ID**: {report.experiment_id}
"""
        report.add_section("Overview", overview)
        
        # モデルパラメータ
        if model_params:
            params_content = self._dict_to_markdown_table(model_params)
            report.add_section("Model Parameters", params_content, "table")
        
        # 性能メトリクス
        if y_true is not None and y_pred is not None:
            metrics = self._calculate_metrics(y_true, y_pred)
            if additional_metrics:
                metrics.update(additional_metrics)
            
            metrics_content = self._dict_to_markdown_table(metrics)
            report.add_section("Performance Metrics", metrics_content, "table")
            
            report.metadata['metrics'] = metrics
        
        # 特徴量重要度
        if feature_importances:
            top_features = sorted(
                feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:15]
            
            fi_content = "| Feature | Importance |\n|---|---|\n"
            for name, imp in top_features:
                fi_content += f"| {name} | {imp:.4f} |\n"
            
            report.add_section("Feature Importance (Top 15)", fi_content, "table")
        
        # データセットサマリー
        if smiles_list:
            data_summary = f"""
**Total Samples**: {len(smiles_list)}  
**Example SMILES**:
- `{smiles_list[0]}` 
- `{smiles_list[1] if len(smiles_list) > 1 else 'N/A'}`
- `{smiles_list[2] if len(smiles_list) > 2 else 'N/A'}`
"""
            report.add_section("Dataset Summary", data_summary)
        
        return report
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """メトリクスを計算"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'Max_Error': np.max(np.abs(y_true - y_pred)),
        }
    
    def _dict_to_markdown_table(self, d: Dict[str, Any]) -> str:
        """辞書をMarkdownテーブルに変換"""
        content = "| Parameter | Value |\n|---|---|\n"
        for key, value in d.items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            content += f"| {key} | {value} |\n"
        return content
    
    def export_html(
        self,
        report: ExperimentReport,
        filepath: str,
        include_style: bool = True,
    ) -> None:
        """HTMLにエクスポート"""
        try:
            import markdown
            md_available = True
        except ImportError:
            md_available = False
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{report.title}</title>",
            "<meta charset='utf-8'>",
        ]
        
        if include_style:
            html_parts.append(self._get_css())
        
        html_parts.append("</head><body>")
        html_parts.append(f"<h1>{report.title}</h1>")
        
        for section in sorted(report.sections, key=lambda x: x.order):
            html_parts.append(f"<div class='section'>")
            html_parts.append(f"<h2>{section.title}</h2>")
            
            if md_available:
                content = markdown.markdown(
                    section.content,
                    extensions=['tables']
                )
            else:
                content = f"<pre>{section.content}</pre>"
            
            html_parts.append(content)
            html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_parts))
        
        logger.info(f"Exported report to {filepath}")
    
    def export_markdown(
        self,
        report: ExperimentReport,
        filepath: str,
    ) -> None:
        """Markdownにエクスポート"""
        lines = [f"# {report.title}", ""]
        
        for section in sorted(report.sections, key=lambda x: x.order):
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Exported report to {filepath}")
    
    def export_json(
        self,
        report: ExperimentReport,
        filepath: str,
    ) -> None:
        """JSONにエクスポート"""
        data = {
            'title': report.title,
            'experiment_id': report.experiment_id,
            'created_at': report.created_at,
            'metadata': report.metadata,
            'sections': [
                {
                    'title': s.title,
                    'content': s.content,
                    'type': s.section_type,
                }
                for s in report.sections
            ],
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported report to {filepath}")
    
    def _get_css(self) -> str:
        """スタイルシート"""
        return """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    background: #f5f5f5;
}
h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
h2 { color: #555; margin-top: 30px; }
.section {
    background: white;
    padding: 20px;
    margin: 15px 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
}
th { background: #4CAF50; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
</style>
"""


def generate_report(
    experiment_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "report.html",
    **kwargs,
) -> str:
    """
    便利関数: レポート生成
    """
    gen = ReportGenerator()
    report = gen.create_experiment_report(
        experiment_name=experiment_name,
        y_true=y_true,
        y_pred=y_pred,
        **kwargs,
    )
    
    path = Path(output_path)
    if path.suffix == '.html':
        gen.export_html(report, output_path)
    elif path.suffix == '.md':
        gen.export_markdown(report, output_path)
    elif path.suffix == '.json':
        gen.export_json(report, output_path)
    else:
        gen.export_html(report, output_path)
    
    return output_path
