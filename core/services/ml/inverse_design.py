"""
逆解析（インバースデザイン）

目的物性値から最適化合物を自動探索
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """候補化合物"""
    smiles: str
    predicted_value: float
    score: float
    properties: Dict[str, float] = field(default_factory=dict)
    rank: int = 0


class InverseDesign:
    """
    逆解析（インバースデザイン）
    
    目的物性を満たす化合物を自動探索
    
    Methods:
    - exhaustive_search: 既知ライブラリから全探索
    - random_sampling: ランダムサンプリング探索
    - bayesian_optimization: ベイズ最適化探索
    
    Example:
        >>> designer = InverseDesign(model, property_name='solubility')
        >>> candidates = designer.optimize(
        ...     target_value=2.0,
        ...     direction='maximize',
        ...     method='bayesian',
        ...     n_iterations=100
        ... )
        >>> print(candidates[0].smiles, candidates[0].predicted_value)
    """
    
    def __init__(
        self,
        model,
        property_name: str = 'solubility',
        random_state: int = 42
    ):
        """
        Args:
            model: 学習済み予測モデル
            property_name: 物性名
            random_state: 乱数シード
        """
        self.model = model
        self.property_name = property_name
        self.random_state = random_state
        np.random.seed(random_state)
    
    def optimize(
        self,
        target_value: float,
        direction: str = 'maximize',
        method: str = 'exhaustive',
        n_iterations: int = 100,
        constraints: Optional[Dict] = None,
        library: Optional[List[str]] = None
    ) -> List[Candidate]:
        """
        最適化合物を探索
        
        Args:
            target_value: 目標値
            direction: 'maximize', 'minimize', 'target'
            method: 'exhaustive', 'random', 'bayesian'
            n_iterations: 試行回数
            constraints: 制約条件
            library: 化合物ライブラリ（全探索用）
        
        Returns:
            候補化合物のリスト（スコア順）
        """
        logger.info(
            f"Starting inverse design: method={method}, "
            f"target={target_value}, direction={direction}"
        )
        
        if method == 'exhaustive':
            if library is None:
                raise ValueError("Library is required for exhaustive search")
            return self._exhaustive_search(library, target_value, direction, constraints)
        
        elif method == 'random':
            return self._random_sampling(target_value, direction, n_iterations, constraints)
        
        elif method == 'bayesian':
            return self._bayesian_optimization(target_value, direction, n_iterations, constraints)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _exhaustive_search(
        self,
        library: List[str],
        target_value: float,
        direction: str,
        constraints: Optional[Dict]
    ) -> List[Candidate]:
        """全ライブラリを予測して評価"""
        logger.info(f"Exhaustive search: {len(library)} molecules")
        
        candidates = []
        
        for smiles in library:
            try:
                # 制約チェック
                if constraints and not self._check_constraints(smiles, constraints):
                    continue
                
                # 予測
                prediction = self._predict_single(smiles)
                
                # スコア計算
                score = self._calculate_score(prediction, target_value, direction)
                
                # 物性計算
                properties = self._calculate_properties(smiles)
                
                candidates.append(Candidate(
                    smiles=smiles,
                    predicted_value=prediction,
                    score=score,
                    properties=properties
                ))
            except Exception as e:
                logger.debug(f"Failed to process {smiles}: {e}")
                continue
        
        # スコア順にソート
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # ランク付け
        for i, c in enumerate(candidates[:100]):
            c.rank = i + 1
        
        logger.info(f"Found {len(candidates)} valid candidates")
        
        return candidates[:100]
    
    def _random_sampling(
        self,
        target_value: float,
        direction: str,
        n_iterations: int,
        constraints: Optional[Dict]
    ) -> List[Candidate]:
        """ランダムに化合物を生成・評価"""
        logger.info(f"Random sampling: {n_iterations} iterations")
        
        candidates = []
        
        for i in range(n_iterations):
            try:
                # ランダムSMILES生成
                smiles = self._generate_random_smiles()
                
                # 制約チェック
                if constraints and not self._check_constraints(smiles, constraints):
                    continue
                
                # 予測
                prediction = self._predict_single(smiles)
                
                # スコア計算
                score = self._calculate_score(prediction, target_value, direction)
                
                # 物性計算
                properties = self._calculate_properties(smiles)
                
                candidates.append(Candidate(
                    smiles=smiles,
                    predicted_value=prediction,
                    score=score,
                    properties=properties
                ))
                
            except Exception as e:
                logger.debug(f"Iteration {i} failed: {e}")
                continue
        
        # スコア順にソート
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # ランク付け
        for i, c in enumerate(candidates[:100]):
            c.rank = i + 1
        
        logger.info(f"Generated {len(candidates)} valid candidates")
        
        return candidates[:100]
    
    def _bayesian_optimization(
        self,
        target_value: float,
        direction: str,
        n_iterations: int,
        constraints: Optional[Dict]
    ) -> List[Candidate]:
        """ベイズ最適化で効率的に探索"""
        logger.info(f"Bayesian optimization: {n_iterations} iterations")
        
        # 簡易版: ランダムサンプリング + スコアベース選択
        # 本格実装はscikit-optimizeなどを使用
        
        candidates = []
        best_score = -np.inf
        
        for i in range(n_iterations):
            try:
                # 初期はランダム、後半は最良付近を探索
                if i < 20 or len(candidates) == 0:
                    smiles = self._generate_random_smiles()
                else:
                    # 最良候補付近を変異
                    best_candidate = candidates[0]
                    smiles = self._mutate_smiles(best_candidate.smiles)
                
                # 制約チェック
                if constraints and not self._check_constraints(smiles, constraints):
                    continue
                
                # 予測
                prediction = self._predict_single(smiles)
                
                # スコア計算
                score = self._calculate_score(prediction, target_value, direction)
                
                # 物性計算
                properties = self._calculate_properties(smiles)
                
                candidate = Candidate(
                    smiles=smiles,
                    predicted_value=prediction,
                    score=score,
                    properties=properties
                )
                
                candidates.append(candidate)
                
                # 最良スコア更新
                if score > best_score:
                    best_score = score
                    logger.info(f"Iteration {i}: New best score = {score:.4f}")
                
                # スコア順に再ソート
                candidates.sort(key=lambda x: x.score, reverse=True)
                
            except Exception as e:
                logger.debug(f"Iteration {i} failed: {e}")
                continue
        
        # ランク付け
        for i, c in enumerate(candidates[:100]):
            c.rank = i + 1
        
        logger.info(f"Optimization completed: best_score={best_score:.4f}")
        
        return candidates[:100]
    
    def _predict_single(self, smiles: str) -> float:
        """単一化合物の予測"""
        # モデルの入力形式に応じて調整
        try:
            prediction = self.model.predict([smiles])[0]
            return float(prediction)
        except:
            # フィーチャー生成が必要な場合
            from core.services.features.rdkit_eng import RDKitEngine
            engine = RDKitEngine()
            features = engine.extract([smiles])
            prediction = self.model.predict(features)[0]
            return float(prediction)
    
    def _calculate_score(
        self,
        prediction: float,
        target_value: float,
        direction: str
    ) -> float:
        """スコア計算"""
        if direction == 'maximize':
            # 予測値が大きいほど良い
            return prediction
        
        elif direction == 'minimize':
            # 予測値が小さいほど良い
            return -prediction
        
        elif direction == 'target':
            # 目標値に近いほど良い
            diff = abs(prediction - target_value)
            return 1.0 / (1.0 + diff)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def _check_constraints(self, smiles: str, constraints: Dict) -> bool:
        """制約条件をチェック"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # 分子量
        if 'molecular_weight' in constraints:
            mw = Descriptors.MolWt(mol)
            if 'min' in constraints['molecular_weight'] and mw < constraints['molecular_weight']['min']:
                return False
            if 'max' in constraints['molecular_weight'] and mw > constraints['molecular_weight']['max']:
                return False
        
        # logP
        if 'logP' in constraints:
            logp = Descriptors.MolLogP(mol)
            if 'min' in constraints['logP'] and logp < constraints['logP']['min']:
                return False
            if 'max' in constraints['logP'] and logp > constraints['logP']['max']:
                return False
        
        # 回転可能結合数
        if 'num_rotatable_bonds' in constraints:
            n_rot = Descriptors.NumRotatableBonds(mol)
            if 'max' in constraints['num_rotatable_bonds'] and n_rot > constraints['num_rotatable_bonds']['max']:
                return False
        
        # TPSA
        if 'TPSA' in constraints:
            tpsa = Descriptors.TPSA(mol)
            if 'min' in constraints['TPSA'] and tpsa < constraints['TPSA']['min']:
                return False
            if 'max' in constraints['TPSA'] and tpsa > constraints['TPSA']['max']:
                return False
        
        return True
    
    def _calculate_properties(self, smiles: str) -> Dict[str, float]:
        """分子物性を計算"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logP': Descriptors.MolLogP(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'TPSA': Descriptors.TPSA(mol),
            'QED': QED.qed(mol)
        }
    
    def _generate_random_smiles(self) -> str:
        """ランダムSMILESを生成（簡易版）"""
        # 簡易実装: 一般的な構造テンプレートから生成
        templates = [
            "c1ccccc1",  # ベンゼン
            "C1CCCCC1",  # シクロヘキサン
            "c1ccc(O)cc1",  # フェノール
            "c1ccc(N)cc1",  # アニリン
            "CC(=O)O",  # 酢酸
            "CCO",  # エタノール
        ]
        
        base = np.random.choice(templates)
        mol = Chem.MolFromSmiles(base)
        
        # ランダム変異（簡易版）
        # 実際はより高度な生成アルゴリズムを使用
        return Chem.MolToSmiles(mol)
    
    def _mutate_smiles(self, smiles: str) -> str:
        """SMILESを変異（簡易版）"""
        # 簡易実装: わずかに構造を変更
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._generate_random_smiles()
        
        # ランダムに官能基を追加/削除
        # 実際はより高度な変異アルゴリズムを使用
        return Chem.MolToSmiles(mol)
