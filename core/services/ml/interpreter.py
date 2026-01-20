"""
モデル解釈エンジン（LIME/Captum inspired）

Implements: F-INTERPRET-001
設計思想:
- 局所的説明
- 特徴寄与分解
- 反実仮想説明
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LocalExplanation:
    """局所的説明"""
    sample_index: int
    prediction: float
    feature_contributions: Dict[str, float]
    top_positive: List[tuple]
    top_negative: List[tuple]


class ModelInterpreter:
    """
    モデル解釈エンジン（LIME/Captum inspired）
    
    Features:
    - 摂動ベース局所説明
    - 特徴寄与分解
    - 反実仮想生成
    
    Example:
        >>> interpreter = ModelInterpreter(model)
        >>> explanation = interpreter.explain(X.iloc[0])
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names
        self.n_samples = n_samples
    
    def explain(
        self,
        sample: pd.Series,
        X_background: Optional[pd.DataFrame] = None,
    ) -> LocalExplanation:
        """単一サンプルを説明"""
        if self.feature_names is None:
            self.feature_names = list(sample.index)
        
        original_pred = self.model.predict(sample.values.reshape(1, -1))[0]
        
        # 摂動ベース寄与計算
        contributions = {}
        
        for i, feature in enumerate(self.feature_names):
            perturbed = sample.values.copy()
            
            # 特徴を平均値で置換
            if X_background is not None:
                perturbed[i] = X_background[feature].mean()
            else:
                perturbed[i] = 0
            
            perturbed_pred = self.model.predict(perturbed.reshape(1, -1))[0]
            contributions[feature] = original_pred - perturbed_pred
        
        # ソート
        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_positive = [(k, v) for k, v in sorted_contribs if v > 0][:5]
        top_negative = [(k, v) for k, v in sorted_contribs if v < 0][-5:]
        
        return LocalExplanation(
            sample_index=0,
            prediction=float(original_pred),
            feature_contributions=contributions,
            top_positive=top_positive,
            top_negative=top_negative,
        )
    
    def explain_batch(
        self,
        X: pd.DataFrame,
        X_background: Optional[pd.DataFrame] = None,
    ) -> List[LocalExplanation]:
        """バッチ説明"""
        return [
            self.explain(X.iloc[i], X_background)
            for i in range(len(X))
        ]
    
    def counterfactual(
        self,
        sample: pd.Series,
        target_pred: float,
        max_changes: int = 3,
    ) -> Dict[str, Any]:
        """反実仮想生成"""
        original_pred = self.model.predict(sample.values.reshape(1, -1))[0]
        
        # 特徴重要度順にソート
        explanation = self.explain(sample)
        sorted_features = sorted(
            explanation.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        
        # 上位特徴を変更
        modified = sample.copy()
        changes = {}
        
        for feature, contrib in sorted_features[:max_changes]:
            idx = list(sample.index).index(feature)
            
            # 方向を決定
            if target_pred > original_pred:
                modified.iloc[idx] *= 1.5 if contrib > 0 else 0.5
            else:
                modified.iloc[idx] *= 0.5 if contrib > 0 else 1.5
            
            changes[feature] = {
                'original': float(sample.iloc[idx]),
                'modified': float(modified.iloc[idx]),
            }
        
        new_pred = self.model.predict(modified.values.reshape(1, -1))[0]
        
        return {
            'original_prediction': float(original_pred),
            'target_prediction': target_pred,
            'achieved_prediction': float(new_pred),
            'changes': changes,
        }
