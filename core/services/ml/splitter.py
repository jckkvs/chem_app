"""
高度なデータ分割

Implements: F-SPLIT-001
設計思想:
- 時系列分割
- グループ分割
- 層化分割
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    高度なデータ分割
    
    Features:
    - 複数分割戦略
    - 検証セット作成
    - クロスバリデーション
    
    Example:
        >>> splitter = DataSplitter(strategy='stratified')
        >>> for train, test in splitter.split(X, y):
        ...     model.fit(X.iloc[train], y.iloc[train])
    """
    
    def __init__(
        self,
        strategy: str = 'random',
        n_splits: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
    
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """分割イテレータ"""
        if self.strategy == 'random':
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            yield from splitter.split(X)
            
        elif self.strategy == 'stratified':
            if y is None:
                raise ValueError("y required for stratified split")
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            yield from splitter.split(X, y)
            
        elif self.strategy == 'group':
            if groups is None:
                raise ValueError("groups required for group split")
            splitter = GroupKFold(n_splits=self.n_splits)
            yield from splitter.split(X, y, groups)
            
        elif self.strategy == 'timeseries':
            splitter = TimeSeriesSplit(n_splits=self.n_splits)
            yield from splitter.split(X)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """単純なtrain/test分割"""
        stratify_col = y if stratify else None
        
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col,
        )
    
    def train_val_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        val_size: float = 0.1,
    ) -> Tuple:
        """train/val/test 3分割"""
        # まずtrain+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        
        # train vs val
        val_ratio = val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.random_state,
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
