"""
実験比較可視化（MLflow/W&B inspired）

Implements: F-COMPARE-VIS-001
設計思想:
- 複数実験メトリクス比較
- 学習曲線プロット
- パラメータ可視化
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentComparator:
    """
    実験比較可視化（MLflow/Weights & Biases inspired）
    
    Features:
    - メトリクス比較表
    - パラメータ並列座標
    - 学習曲線オーバーレイ
    
    Example:
        >>> comparator = ExperimentComparator()
        >>> comparator.add_experiment("exp1", metrics, config)
        >>> html = comparator.generate_comparison()
    """
    
    def __init__(self):
        self.experiments: List[Dict[str, Any]] = []
    
    def add_experiment(
        self,
        name: str,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        learning_curve: Optional[List[float]] = None,
    ) -> None:
        """実験を追加"""
        self.experiments.append({
            'name': name,
            'metrics': metrics,
            'config': config or {},
            'learning_curve': learning_curve or [],
        })
    
    def generate_comparison(self) -> str:
        """比較HTMLを生成"""
        if not self.experiments:
            return "<p>No experiments to compare</p>"
        
        # メトリクステーブル
        metrics_table = self._generate_metrics_table()
        
        # パラメータテーブル
        params_table = self._generate_params_table()
        
        # ベスト実験
        best = self._find_best_experiment()
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
        h1 {{ color: #58a6ff; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #161b22; color: #58a6ff; padding: 12px; }}
        td {{ padding: 10px; border-bottom: 1px solid #21262d; }}
        .best {{ color: #3fb950; font-weight: bold; }}
        .card {{ background: #161b22; padding: 20px; border-radius: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>📊 Experiment Comparison</h1>
    
    <div class="card">
        <h2>🏆 Best: {best['name']} (R² = {best['metrics'].get('r2', 0):.4f})</h2>
    </div>
    
    <h2>Metrics</h2>
    {metrics_table}
    
    <h2>Parameters</h2>
    {params_table}
</body>
</html>
"""
    
    def _generate_metrics_table(self) -> str:
        """メトリクステーブル生成"""
        all_metrics = set()
        for exp in self.experiments:
            all_metrics.update(exp['metrics'].keys())
        
        headers = ['Experiment'] + sorted(all_metrics)
        header_row = ''.join(f"<th>{h}</th>" for h in headers)
        
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
        
        return f"<table><tr>{header_row}</tr>{''.join(rows)}</table>"
    
    def _generate_params_table(self) -> str:
        """パラメータテーブル生成"""
        all_params = set()
        for exp in self.experiments:
            all_params.update(exp['config'].keys())
        
        if not all_params:
            return "<p>No parameters to display</p>"
        
        headers = ['Experiment'] + sorted(all_params)
        header_row = ''.join(f"<th>{h}</th>" for h in headers)
        
        rows = []
        for exp in self.experiments:
            cells = [f"<td>{exp['name']}</td>"]
            for p in sorted(all_params):
                val = exp['config'].get(p, '-')
                cells.append(f"<td>{val}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        
        return f"<table><tr>{header_row}</tr>{''.join(rows)}</table>"
    
    def _find_best_experiment(self) -> Dict[str, Any]:
        """ベスト実験を特定"""
        if not self.experiments:
            return {'name': 'N/A', 'metrics': {}}
        
        return max(
            self.experiments,
            key=lambda x: x['metrics'].get('r2', x['metrics'].get('accuracy', 0))
        )
    
    def get_ranking(self, metric: str = 'r2') -> List[Dict[str, Any]]:
        """メトリクスでランキング"""
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: x['metrics'].get(metric, 0),
            reverse=True,
        )
        return sorted_exps
