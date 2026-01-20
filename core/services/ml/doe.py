"""
実験計画法（DOE）

Implements: F-DOE-001
設計思想:
- 実験設計
- 空間充填
- 最適配置
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentDesign:
    """実験計画"""
    points: np.ndarray
    n_experiments: int
    design_type: str
    coverage: float


class DesignOfExperiments:
    """
    実験計画法
    
    Features:
    - ラテン超方格
    - 中心複合計画
    - 空間充填設計
    
    Example:
        >>> doe = DesignOfExperiments()
        >>> design = doe.latin_hypercube(n_samples=20, n_dims=4)
    """
    
    def latin_hypercube(
        self,
        n_samples: int,
        n_dims: int,
        bounds: List[Tuple[float, float]] = None,
    ) -> ExperimentDesign:
        """ラテン超方格サンプリング"""
        if bounds is None:
            bounds = [(0, 1)] * n_dims
        
        # 各次元でn_samples個の区間に分割
        points = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            low, high = bounds[dim]
            perms = np.random.permutation(n_samples)
            
            for i in range(n_samples):
                u = np.random.uniform(0, 1)
                points[i, dim] = low + (high - low) * (perms[i] + u) / n_samples
        
        coverage = self._calculate_coverage(points)
        
        return ExperimentDesign(
            points=points,
            n_experiments=n_samples,
            design_type='latin_hypercube',
            coverage=coverage,
        )
    
    def central_composite(
        self,
        n_factors: int,
        alpha: float = 1.414,
    ) -> ExperimentDesign:
        """中心複合計画"""
        # 2^k 階乗計画
        factorial_points = np.array(
            [[(-1) ** (i // (2 ** j)) for j in range(n_factors)]
             for i in range(2 ** n_factors)]
        )
        
        # 軸点
        axial_points = []
        for i in range(n_factors):
            point_plus = [0] * n_factors
            point_minus = [0] * n_factors
            point_plus[i] = alpha
            point_minus[i] = -alpha
            axial_points.append(point_plus)
            axial_points.append(point_minus)
        axial_points = np.array(axial_points)
        
        # 中心点
        center_point = np.zeros((1, n_factors))
        
        all_points = np.vstack([factorial_points, axial_points, center_point])
        
        return ExperimentDesign(
            points=all_points,
            n_experiments=len(all_points),
            design_type='central_composite',
            coverage=self._calculate_coverage(all_points),
        )
    
    def space_filling(
        self,
        n_samples: int,
        n_dims: int,
        method: str = 'maximin',
    ) -> ExperimentDesign:
        """空間充填設計"""
        if method == 'maximin':
            return self._maximin_design(n_samples, n_dims)
        else:
            return self.latin_hypercube(n_samples, n_dims)
    
    def _maximin_design(
        self,
        n_samples: int,
        n_dims: int,
        n_iterations: int = 100,
    ) -> ExperimentDesign:
        """Maximin設計（最小距離を最大化）"""
        best_design = np.random.rand(n_samples, n_dims)
        best_min_dist = self._min_distance(best_design)
        
        for _ in range(n_iterations):
            candidate = np.random.rand(n_samples, n_dims)
            min_dist = self._min_distance(candidate)
            
            if min_dist > best_min_dist:
                best_design = candidate
                best_min_dist = min_dist
        
        return ExperimentDesign(
            points=best_design,
            n_experiments=n_samples,
            design_type='maximin',
            coverage=self._calculate_coverage(best_design),
        )
    
    def _min_distance(self, points: np.ndarray) -> float:
        """点間の最小距離"""
        from scipy.spatial.distance import pdist
        distances = pdist(points)
        return distances.min() if len(distances) > 0 else 0
    
    def _calculate_coverage(self, points: np.ndarray) -> float:
        """空間充填率"""
        # 各次元の範囲をカバーしているか
        ranges = points.max(axis=0) - points.min(axis=0)
        return ranges.mean()
