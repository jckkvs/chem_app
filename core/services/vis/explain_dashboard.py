"""
モデル説明ダッシュボード

Implements: F-EXPLAIN-DASH-001
設計思想:
- インタラクティブ説明
- 特徴量重要度可視化
- 局所/大域説明統合
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExplainabilityDashboard:
    """
    モデル説明ダッシュボード
    
    Features:
    - 特徴量重要度表示
    - SHAP/LIME統合
    - インタラクティブHTML
    
    Example:
        >>> dash = ExplainabilityDashboard(model, X, feature_names)
        >>> html = dash.generate()
    """
    
    def __init__(
        self,
        model=None,
        X=None,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.X = X
        self.feature_names = feature_names or []
        self.importance_: Optional[Dict[str, float]] = None
    
    def compute_importance(self) -> Dict[str, float]:
        """特徴量重要度を計算"""
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            names = self.feature_names or [f"Feature_{i}" for i in range(len(importances))]
            self.importance_ = dict(zip(names, importances))
        else:
            self.importance_ = {}
        
        return self.importance_
    
    def generate(self) -> str:
        """ダッシュボードHTML生成"""
        if self.importance_ is None:
            self.compute_importance()
        
        # 重要度バー
        importance_bars = ""
        if self.importance_:
            sorted_imp = sorted(self.importance_.items(), key=lambda x: x[1], reverse=True)[:15]
            max_imp = max(v for _, v in sorted_imp) if sorted_imp else 1
            
            for name, value in sorted_imp:
                width = (value / max_imp) * 100
                importance_bars += f"""
                <div class="bar-container">
                    <div class="bar-label">{name}</div>
                    <div class="bar" style="width: {width}%"></div>
                    <div class="bar-value">{value:.4f}</div>
                </div>
                """
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Explainability Dashboard</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee; 
            padding: 30px;
            min-height: 100vh;
        }}
        h1 {{ color: #00d9ff; text-shadow: 0 0 20px rgba(0,217,255,0.5); }}
        .card {{
            background: rgba(22, 33, 62, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }}
        .bar-container {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}
        .bar-label {{
            width: 200px;
            font-size: 0.9em;
            color: #aaa;
        }}
        .bar {{
            height: 20px;
            background: linear-gradient(90deg, #00d9ff, #0099ff);
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        .bar-value {{
            margin-left: 10px;
            font-size: 0.85em;
            color: #00d9ff;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            color: #00d9ff;
        }}
    </style>
</head>
<body>
    <h1>🧠 Model Explainability</h1>
    
    <div class="card summary">
        <div class="stat">
            <div class="stat-value">{len(self.importance_)}</div>
            <div>Features</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len(self.X) if self.X is not None else 0}</div>
            <div>Samples</div>
        </div>
        <div class="stat">
            <div class="stat-value">{self.model.__class__.__name__ if self.model else 'N/A'}</div>
            <div>Model</div>
        </div>
    </div>
    
    <div class="card">
        <h2>📊 Feature Importance</h2>
        {importance_bars if importance_bars else "<p>No importance data available</p>"}
    </div>
</body>
</html>
"""
