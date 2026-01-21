"""
ハイパーパラメータ最適化（Ray Tune inspired）

Implements: F-HPOPT-001
設計思想:
- グリッド/ランダム/ベイズ最適化
- 早期終了
- 並列探索
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HPOResult:
    """最適化結果"""
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0


class HyperParameterOptimizer:
    """
    ハイパーパラメータ最適化（Ray Tune/Hyperopt inspired）
    
    Features:
    - グリッドサーチ
    - ランダムサーチ
    - ベイズ最適化（簡易版）
    
    Example:
        >>> hpo = HyperParameterOptimizer(method='random', n_trials=50)
        >>> result = hpo.optimize(objective_func, search_space)
    """
    
    def __init__(
        self,
        method: str = 'random',
        n_trials: int = 50,
        random_state: int = 42,
    ):
        self.method = method
        self.n_trials = n_trials
        self.rng = np.random.RandomState(random_state)
    
    def optimize(
        self,
        objective: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Any],
        maximize: bool = True,
    ) -> HPOResult:
        """最適化実行"""
        if self.method == 'grid':
            return self._grid_search(objective, search_space, maximize)
        elif self.method == 'random':
            return self._random_search(objective, search_space, maximize)
        elif self.method == 'bayesian':
            return self._bayesian_search(objective, search_space, maximize)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _grid_search(
        self,
        objective: Callable,
        search_space: Dict[str, Any],
        maximize: bool,
    ) -> HPOResult:
        """グリッドサーチ"""
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        
        best_score = float('-inf') if maximize else float('inf')
        best_params = {}
        all_trials = []
        
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            score = objective(params)
            
            all_trials.append({'params': params, 'score': score})
            
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            n_trials=len(all_trials),
        )
    
    def _random_search(
        self,
        objective: Callable,
        search_space: Dict[str, Any],
        maximize: bool,
    ) -> HPOResult:
        """ランダムサーチ"""
        best_score = float('-inf') if maximize else float('inf')
        best_params = {}
        all_trials = []
        
        for _ in range(self.n_trials):
            params = self._sample_params(search_space)
            score = objective(params)
            
            all_trials.append({'params': params, 'score': score})
            
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            n_trials=self.n_trials,
        )
    
    def _bayesian_search(
        self,
        objective: Callable,
        search_space: Dict[str, Any],
        maximize: bool,
    ) -> HPOResult:
        """簡易ベイズ最適化"""
        # 探索的→活用的に移行
        all_trials = []
        
        for i in range(self.n_trials):
            # 探索率を徐々に下げる
            explore_rate = max(0.1, 1 - i / self.n_trials)
            
            if self.rng.random() < explore_rate or len(all_trials) < 5:
                params = self._sample_params(search_space)
            else:
                # 上位トライアルから摂動
                top_trials = sorted(all_trials, key=lambda x: x['score'], reverse=maximize)[:3]
                base_params = self.rng.choice(top_trials)['params']
                params = self._perturb_params(base_params, search_space)
            
            score = objective(params)
            all_trials.append({'params': params, 'score': score})
        
        best = max(all_trials, key=lambda x: x['score']) if maximize else min(all_trials, key=lambda x: x['score'])
        
        return HPOResult(
            best_params=best['params'],
            best_score=best['score'],
            all_trials=all_trials,
            n_trials=self.n_trials,
        )
    
    def _sample_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータをサンプリング"""
        params = {}
        for key, values in search_space.items():
            if isinstance(values, list):
                params[key] = self.rng.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                low, high = values
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = self.rng.randint(low, high + 1)
                else:
                    params[key] = self.rng.uniform(low, high)
            else:
                params[key] = values
        return params
    
    def _perturb_params(
        self,
        base_params: Dict[str, Any],
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """パラメータを摂動"""
        params = base_params.copy()
        key = self.rng.choice(list(params.keys()))
        params[key] = self._sample_params({key: search_space[key]})[key]
        return params
