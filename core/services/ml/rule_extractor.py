"""
解釈可能ルール抽出エンジン

Implements: F-RULES-001
設計思想:
- 決定木からIF-THENルールを抽出
- 化学的に意味のあるルール発見
- 知識発見支援
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """抽出されたルール"""
    conditions: List[str]
    prediction: float
    support: int  # 該当サンプル数
    confidence: float  # 信頼度
    
    def to_string(self) -> str:
        conditions_str = " AND ".join(self.conditions)
        return f"IF {conditions_str} THEN predict = {self.prediction:.4f} (support={self.support}, conf={self.confidence:.2f})"


class RuleExtractor:
    """
    解釈可能ルール抽出エンジン
    
    Features:
    - 決定木からIF-THENルール抽出
    - ルールの重要度ランキング
    - 化学記述子に基づくルール説明
    
    Example:
        >>> extractor = RuleExtractor(max_depth=5)
        >>> extractor.fit(X, y)
        >>> rules = extractor.extract_rules(min_support=10)
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        task_type: str = 'regression',
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.task_type = task_type
        
        self.tree_: Optional[Any] = None
        self.feature_names_: List[str] = []
        self.X_: Optional[pd.DataFrame] = None
        self.y_: Optional[pd.Series] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RuleExtractor':
        """ルール抽出用の決定木を学習"""
        self.feature_names_ = list(X.columns)
        self.X_ = X
        self.y_ = y
        
        if self.task_type == 'regression':
            self.tree_ = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
        else:
            self.tree_ = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )
        
        self.tree_.fit(X, y)
        return self
    
    def extract_rules(
        self,
        min_support: int = 5,
        min_confidence: float = 0.0,
    ) -> List[Rule]:
        """ルールを抽出"""
        if self.tree_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        tree = self.tree_.tree_
        rules = []
        
        def recurse(node: int, conditions: List[str], depth: int):
            if tree.feature[node] == -2:  # リーフノード
                prediction = tree.value[node].flatten()[0]
                if self.task_type == 'classification':
                    prediction = int(np.argmax(tree.value[node]))
                
                support = int(tree.n_node_samples[node])
                
                if support >= min_support:
                    # 信頼度計算
                    if self.task_type == 'regression':
                        # 予測値と実際値の相関
                        indices = self._get_leaf_indices(node)
                        if len(indices) > 0:
                            actual = self.y_.iloc[indices]
                            confidence = 1.0 - (actual.std() / (self.y_.std() + 1e-9))
                        else:
                            confidence = 0.0
                    else:
                        confidence = tree.value[node].max() / tree.n_node_samples[node]
                    
                    if confidence >= min_confidence:
                        rules.append(Rule(
                            conditions=conditions.copy(),
                            prediction=float(prediction),
                            support=support,
                            confidence=float(confidence),
                        ))
                return
            
            feature_name = self.feature_names_[tree.feature[node]]
            threshold = tree.threshold[node]
            
            # 左の子（<=）
            left_condition = f"{feature_name} <= {threshold:.4f}"
            recurse(tree.children_left[node], conditions + [left_condition], depth + 1)
            
            # 右の子（>）
            right_condition = f"{feature_name} > {threshold:.4f}"
            recurse(tree.children_right[node], conditions + [right_condition], depth + 1)
        
        recurse(0, [], 0)
        
        # サポートでソート
        rules.sort(key=lambda r: r.support, reverse=True)
        
        return rules
    
    def _get_leaf_indices(self, leaf_node: int) -> List[int]:
        """リーフノードに該当するサンプルインデックスを取得"""
        leaf_ids = self.tree_.apply(self.X_)
        return list(np.where(leaf_ids == leaf_node)[0])
    
    def get_tree_text(self) -> str:
        """決定木のテキスト表現"""
        if self.tree_ is None:
            return ""
        return export_text(self.tree_, feature_names=self.feature_names_)
    
    def get_top_rules_html(self, rules: List[Rule], top_n: int = 10) -> str:
        """トップルールをHTML表示"""
        html = '<div style="font-family: monospace;">'
        for i, rule in enumerate(rules[:top_n]):
            html += f'<p><b>Rule {i+1}:</b> {rule.to_string()}</p>'
        html += '</div>'
        return html
