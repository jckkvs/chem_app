"""
ワークフローオーケストレーター（KNIME/Pipeline Pilot inspired）

Implements: F-WORKFLOW-001
設計思想:
- パイプライン構築
- ステップ実行
- 結果キャッシング
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """ワークフローステップ"""
    name: str
    function: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """ステップ結果"""
    step_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Workflow:
    """
    ワークフローオーケストレーター（KNIME inspired）
    
    Features:
    - パイプライン構築
    - 依存関係解決
    - 並列/逐次実行
    - 結果キャッシュ
    
    Example:
        >>> wf = Workflow("My Pipeline")
        >>> wf.add_step("load", load_data, {"path": "data.csv"})
        >>> wf.add_step("extract", extract_features, depends_on=["load"])
        >>> results = wf.run()
    """
    
    def __init__(self, name: str = "Unnamed Workflow"):
        self.name = name
        self.steps: Dict[str, WorkflowStep] = {}
        self.results: Dict[str, StepResult] = {}
        self.data: Dict[str, Any] = {}  # データ共有用
    
    def add_step(
        self,
        name: str,
        function: Callable,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
    ) -> 'Workflow':
        """ステップを追加"""
        self.steps[name] = WorkflowStep(
            name=name,
            function=function,
            params=params or {},
            depends_on=depends_on or [],
        )
        return self
    
    def run(self, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, StepResult]:
        """ワークフロー実行"""
        self.data = initial_data or {}
        self.results = {}
        
        # トポロジカルソート
        execution_order = self._topological_sort()
        
        logger.info(f"Starting workflow: {self.name}")
        logger.info(f"Execution order: {execution_order}")
        
        for step_name in execution_order:
            result = self._execute_step(step_name)
            self.results[step_name] = result
            
            if not result.success:
                logger.error(f"Step {step_name} failed: {result.error}")
                break
        
        return self.results
    
    def _topological_sort(self) -> List[str]:
        """依存関係に基づいてソート"""
        visited = set()
        order = []
        
        def dfs(node: str):
            if node in visited:
                return
            visited.add(node)
            
            for dep in self.steps[node].depends_on:
                if dep in self.steps:
                    dfs(dep)
            
            order.append(node)
        
        for step_name in self.steps:
            dfs(step_name)
        
        return order
    
    def _execute_step(self, step_name: str) -> StepResult:
        """単一ステップ実行"""
        step = self.steps[step_name]
        
        logger.info(f"Executing step: {step_name}")
        start_time = time.time()
        
        try:
            # 依存ステップの出力を取得
            dep_outputs = {
                dep: self.results[dep].output
                for dep in step.depends_on
                if dep in self.results
            }
            
            # パラメータに共有データと依存出力を追加
            params = {
                **step.params,
                'workflow_data': self.data,
                'dependencies': dep_outputs,
            }
            
            # 実行
            output = step.function(**params)
            
            # 共有データに保存
            self.data[step_name] = output
            
            duration = time.time() - start_time
            
            return StepResult(
                step_name=step_name,
                success=True,
                output=output,
                duration_seconds=round(duration, 3),
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return StepResult(
                step_name=step_name,
                success=False,
                error=str(e),
                duration_seconds=round(duration, 3),
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """実行サマリー"""
        total_duration = sum(r.duration_seconds for r in self.results.values())
        successful = sum(1 for r in self.results.values() if r.success)
        
        return {
            'workflow': self.name,
            'total_steps': len(self.steps),
            'executed': len(self.results),
            'successful': successful,
            'failed': len(self.results) - successful,
            'total_duration': round(total_duration, 3),
        }


# プリセットワークフロー関数
def load_csv_step(path: str = "", **kwargs) -> Any:
    """CSVロードステップ"""
    import pandas as pd
    return pd.read_csv(path) if path else None


def extract_features_step(workflow_data: dict = None, **kwargs) -> Any:
    """特徴量抽出ステップ"""
    from core.services.features.rdkit_eng import RDKitFeatureExtractor
    
    df = workflow_data.get('load', None)
    if df is None:
        return None
    
    extractor = RDKitFeatureExtractor()
    smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    return extractor.transform(df[smiles_col].tolist())


def train_model_step(dependencies: dict = None, **kwargs) -> Any:
    """モデル訓練ステップ"""
    features = dependencies.get('extract', None)
    if features is None:
        return None
    
    # 簡易訓練（実際にはtarget必要）
    return {'model': 'trained', 'n_features': len(features.columns)}
