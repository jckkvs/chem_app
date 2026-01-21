"""
高度クロスバリデーション

Implements: F-CV-001
設計思想:
- 複数CVストラテジー対応
- 分子特化型CV（Scaffold Split）
- 結果の詳細レポート
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """CV結果"""
    train_scores: List[float]
    test_scores: List[float]
    mean_train_score: float
    mean_test_score: float
    std_test_score: float
    fold_details: List[Dict[str, Any]] = field(default_factory=list)


class CrossValidator:
    """
    高度クロスバリデーション
    
    Features:
    - K-Fold, Stratified, Group, Scaffold Split
    - 複数メトリクス対応
    - 詳細レポート生成
    
    Example:
        >>> cv = CrossValidator(strategy='kfold', n_splits=5)
        >>> result = cv.validate(model, X, y)
    """
    
    def __init__(
        self,
        strategy: Literal['kfold', 'stratified', 'group', 'scaffold'] = 'kfold',
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
        self.random_state = random_state
    
    def validate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
        smiles: Optional[List[str]] = None,
        scoring: str = 'r2',
    ) -> CVResult:
        """クロスバリデーション実行"""
        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error, r2_score
        
        splitter = self._get_splitter(groups, smiles)
        
        train_scores = []
        test_scores = []
        fold_details = []
        
        split_args = (X,) if groups is None else (X, y, groups)
        
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 学習
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # 評価
            train_pred = fold_model.predict(X_train)
            test_pred = fold_model.predict(X_test)
            
            if scoring == 'r2':
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
            else:  # rmse
                train_score = -np.sqrt(mean_squared_error(y_train, train_pred))
                test_score = -np.sqrt(mean_squared_error(y_test, test_pred))
            
            train_scores.append(train_score)
            test_scores.append(test_score)
            
            fold_details.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_score': train_score,
                'test_score': test_score,
            })
        
        return CVResult(
            train_scores=train_scores,
            test_scores=test_scores,
            mean_train_score=np.mean(train_scores),
            mean_test_score=np.mean(test_scores),
            std_test_score=np.std(test_scores),
            fold_details=fold_details,
        )
    
    def _get_splitter(
        self,
        groups: Optional[pd.Series] = None,
        smiles: Optional[List[str]] = None,
    ):
        """スプリッターを取得"""
        if self.strategy == 'stratified':
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        
        elif self.strategy == 'group':
            return GroupKFold(n_splits=self.n_splits)
        
        elif self.strategy == 'scaffold' and smiles is not None:
            return ScaffoldSplit(
                n_splits=self.n_splits,
                smiles=smiles,
            )
        
        else:  # kfold
            return KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )


class ScaffoldSplit:
    """Scaffold-based split for molecules"""
    
    def __init__(self, n_splits: int, smiles: List[str]):
        self.n_splits = n_splits
        self.smiles = smiles
    
    def split(self, X, y=None, groups=None):
        """Scaffold split"""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            scaffolds = {}
            for idx, smi in enumerate(self.smiles):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False
                    )
                else:
                    scaffold = smi
                
                if scaffold not in scaffolds:
                    scaffolds[scaffold] = []
                scaffolds[scaffold].append(idx)
            
            # スキャフォルドをサイズでソート
            scaffold_sets = list(scaffolds.values())
            scaffold_sets.sort(key=len, reverse=True)
            
            # フォールドに分配
            folds = [[] for _ in range(self.n_splits)]
            for scaffold_indices in scaffold_sets:
                # 最も小さいフォールドに追加
                min_fold = min(range(self.n_splits), key=lambda i: len(folds[i]))
                folds[min_fold].extend(scaffold_indices)
            
            # 各フォールドをテストにして分割
            all_indices = set(range(len(self.smiles)))
            for test_indices in folds:
                train_indices = list(all_indices - set(test_indices))
                yield train_indices, test_indices
                
        except Exception as e:
            logger.warning(f"Scaffold split failed, using KFold: {e}")
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            yield from kfold.split(X)
