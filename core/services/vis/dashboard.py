"""
実験ダッシュボード生成

Implements: F-DASHBOARD-001
設計思想:
- 全実験の比較ダッシュボード
- メトリクス推移
- HTMLエクスポート
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ExperimentDashboard:
    """
    実験ダッシュボード生成
    
    Features:
    - 実験結果の一覧表示
    - メトリクス比較グラフ
    - HTML出力
    
    Example:
        >>> dashboard = ExperimentDashboard()
        >>> dashboard.add_experiment("exp1", {"r2": 0.95, "rmse": 0.1})
        >>> html = dashboard.generate_html()
    """
    
    def __init__(self):
        self.experiments: List[Dict[str, Any]] = []
    
    def add_experiment(
        self,
        name: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """実験を追加"""
        self.experiments.append({
            'name': name,
            'metrics': metrics,
            'config': config or {},
            'timestamp': datetime.now().isoformat(),
        })
    
    def generate_html(self) -> str:
        """HTMLダッシュボード生成"""
        if not self.experiments:
            return "<p>No experiments to display</p>"
        
        # メトリクス名を収集
        all_metrics = set()
        for exp in self.experiments:
            all_metrics.update(exp['metrics'].keys())
        
        # テーブル行生成
        rows = []
        for exp in self.experiments:
            cells = [f"<td>{exp['name']}</td>"]
            for m in sorted(all_metrics):
                val = exp['metrics'].get(m, '-')
                if isinstance(val, float):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{val}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        
        headers = ['Experiment'] + sorted(all_metrics)
        header_row = ''.join(f"<th>{h}</th>" for h in headers)
        
        # ベスト実験を特定（最初のメトリクスで）
        if self.experiments and all_metrics:
            first_metric = sorted(all_metrics)[0]
            best_exp = max(
                self.experiments,
                key=lambda x: x['metrics'].get(first_metric, 0)
            )
            best_name = best_exp['name']
        else:
            best_name = "N/A"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #16213e; color: #00d4ff; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #333; }}
        tr:hover {{ background: #16213e; }}
        .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; border-radius: 10px; flex: 1; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Experiment Dashboard</h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(self.experiments)}</div>
                <div class="stat-label">Total Experiments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(all_metrics)}</div>
                <div class="stat-label">Metrics Tracked</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{best_name}</div>
                <div class="stat-label">Best Experiment</div>
            </div>
        </div>
        
        <h2>📊 Experiment Results</h2>
        <table>
            <thead><tr>{header_row}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
        
        <p style="color: #666; font-size: 0.8em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        return html
    
    def save_html(self, filepath: str) -> str:
        """HTMLを保存"""
        html = self.generate_html()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        return filepath
