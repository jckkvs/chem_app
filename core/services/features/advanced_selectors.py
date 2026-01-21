"""
高度な特徴量選択アルゴリズム

Implements: F-SELECT-002
設計思想:
- Boruta: ランダムフォレストベースの全特徴量選択
- SHAP重要度: モデル解釈性と選択の両立
- 相互情報量ベース: 非線形関係の検出
- 冗長性除去: mRMR (minimum Redundancy Maximum Relevance)

継続的に新しい手法を追加可能な設計。

参考文献:
- Boruta: Kursa & Rudnicki, JMLS 2010
- SHAP: Lundberg & Lee, NeurIPS 2017
- mRMR: Peng et al., TPAMI 2005
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """特徴量選択結果"""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    method: str
    n_original: int = 0
    n_selected: int = 0
    
    def summary(self) -> str:
        return f"{self.method}: {self.n_original} → {self.n_selected} features"


class BorutaSelector:
    """
    Borutaアルゴリズムによる特徴量選択
    
    ランダムフォレストを使用し、シャドウ特徴量との比較で
    有意な特徴量を自動選択。
    
    Reference: Kursa & Rudnicki, "Feature Selection with the Boruta Package", 2010
    """
    
    def __init__(
        self,
        task_type: Literal['regression', 'classification'] = 'regression',
        n_estimators: int = 100,
        max_iter: int = 50,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        """
        Args:
            task_type: タスクタイプ
            n_estimators: ランダムフォレストの木の数
            max_iter: 最大イテレーション数
            alpha: 有意水準
            random_state: 乱数シード
        """
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BorutaSelector':
        """
        Borutaアルゴリズムを実行
        
        手順:
        1. シャドウ特徴量（元特徴量のシャッフル）を追加
        2. ランダムフォレストで重要度計算
        3. 各特徴量がシャドウの最大重要度より有意に高いか検定
        4. 確定（高い）/ 棄却（低い）/ 保留を決定
        5. max_iter回繰り返し
        """
        np.random.seed(self.random_state)
        
        feature_names = list(X.columns)
        n_features = len(feature_names)
        
        # 各特徴量のヒット回数（シャドウに勝った回数）
        hit_counts = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            # シャドウ特徴量を作成
            X_shadow = X.apply(np.random.permutation)
            X_shadow.columns = [f"shadow_{c}" for c in X.columns]
            
            # 結合
            X_combined = pd.concat([X, X_shadow], axis=1)
            
            # ランダムフォレスト学習
            if self.task_type == 'regression':
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state + iteration,
                    n_jobs=-1,
                )
            else:
                rf = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state + iteration,
                    n_jobs=-1,
                )
            
            rf.fit(X_combined.values, y.values)
            
            # 重要度取得
            importances = rf.feature_importances_
            original_imp = importances[:n_features]
            shadow_imp = importances[n_features:]
            
            # シャドウの最大重要度
            shadow_max = shadow_imp.max()
            
            # シャドウに勝った特徴量をカウント
            hits = original_imp > shadow_max
            hit_counts += hits.astype(int)
        
        # 二項検定で有意かどうか判定
        from scipy import stats
        
        selected = []
        scores = {}
        
        for i, name in enumerate(feature_names):
            # 帰無仮説: ヒット確率 = 0.5（ランダム）
            p_value = stats.binom_test(
                hit_counts[i], 
                self.max_iter, 
                0.5, 
                alternative='greater'
            )
            
            scores[name] = hit_counts[i] / self.max_iter
            
            if p_value < self.alpha:
                selected.append(name)
        
        self.selected_features_ = selected
        self.feature_scores_ = scores
        
        logger.info(f"Boruta: {n_features} → {len(selected)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """選択された特徴量のみ抽出"""
        if self.selected_features_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def get_result(self) -> FeatureSelectionResult:
        return FeatureSelectionResult(
            selected_features=self.selected_features_ or [],
            feature_scores=self.feature_scores_ or {},
            method="Boruta",
            n_original=len(self.feature_scores_ or {}),
            n_selected=len(self.selected_features_ or []),
        )


class MRMRSelector:
    """
    mRMR (minimum Redundancy Maximum Relevance) 特徴量選択
    
    ターゲットとの相関が高く（Relevance）、
    他の選択済み特徴量との相関が低い（Redundancy）特徴量を選択。
    
    Reference: Peng et al., "Feature Selection Based on Mutual Information", 2005
    """
    
    def __init__(
        self,
        k: int = 20,
        task_type: Literal['regression', 'classification'] = 'regression',
    ):
        """
        Args:
            k: 選択する特徴量数
            task_type: タスクタイプ
        """
        self.k = k
        self.task_type = task_type
        
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MRMRSelector':
        """mRMRアルゴリズムを実行"""
        feature_names = list(X.columns)
        n_features = len(feature_names)
        
        # 1. 各特徴量とターゲットの相互情報量（Relevance）
        if self.task_type == 'regression':
            relevance = mutual_info_regression(X.values, y.values, random_state=42)
        else:
            relevance = mutual_info_classif(X.values, y.values, random_state=42)
        
        relevance_dict = {name: rel for name, rel in zip(feature_names, relevance)}
        
        # 2. 貪欲法で特徴量を選択
        selected = []
        remaining = set(feature_names)
        scores = {}
        
        for _ in range(min(self.k, n_features)):
            best_score = -np.inf
            best_feature = None
            
            for f in remaining:
                # Relevance
                rel = relevance_dict[f]
                
                # Redundancy（選択済み特徴量との平均相互情報量）
                if selected:
                    X_f = X[f].values.reshape(-1, 1)
                    X_selected = X[selected].values
                    
                    # 数値安定性のため小さな値を加算
                    redundancy = 0
                    for s in selected:
                        corr = np.corrcoef(X[f].values, X[s].values)[0, 1]
                        redundancy += abs(corr) if not np.isnan(corr) else 0
                    redundancy /= len(selected)
                else:
                    redundancy = 0
                
                # mRMRスコア = Relevance - Redundancy
                mrmr_score = rel - redundancy
                
                if mrmr_score > best_score:
                    best_score = mrmr_score
                    best_feature = f
            
            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                scores[best_feature] = best_score
        
        self.selected_features_ = selected
        self.feature_scores_ = scores
        
        logger.info(f"mRMR: {n_features} → {len(selected)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def get_result(self) -> FeatureSelectionResult:
        return FeatureSelectionResult(
            selected_features=self.selected_features_ or [],
            feature_scores=self.feature_scores_ or {},
            method="mRMR",
            n_original=len(self.feature_scores_ or {}) + (self.k if self.selected_features_ else 0),
            n_selected=len(self.selected_features_ or []),
        )


class PermutationImportanceSelector:
    """
    Permutation Importance による特徴量選択
    
    各特徴量をシャッフルしてモデル性能の低下度を測定。
    モデルに依存しない汎用的な重要度。
    """
    
    def __init__(
        self,
        model=None,
        k: int = 20,
        n_repeats: int = 10,
        task_type: Literal['regression', 'classification'] = 'regression',
        random_state: int = 42,
    ):
        """
        Args:
            model: 使用するモデル（Noneの場合はRandomForest）
            k: 選択する特徴量数
            n_repeats: シャッフル繰り返し回数
            task_type: タスクタイプ
            random_state: 乱数シード
        """
        self.model = model
        self.k = k
        self.n_repeats = n_repeats
        self.task_type = task_type
        self.random_state = random_state
        
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PermutationImportanceSelector':
        """Permutation Importanceを計算"""
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import cross_val_score

        # モデル準備
        if self.model is None:
            if self.task_type == 'regression':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
        
        # モデル学習
        self.model.fit(X.values, y.values)
        
        # Permutation Importance計算
        result = permutation_importance(
            self.model,
            X.values,
            y.values,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        # 重要度でソート
        importances = result.importances_mean
        feature_names = list(X.columns)
        
        scores = {name: imp for name, imp in zip(feature_names, importances)}
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        self.selected_features_ = [f for f, _ in sorted_features[:self.k]]
        self.feature_scores_ = scores
        
        logger.info(f"Permutation Importance: {len(feature_names)} → {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class EnsembleFeatureSelector:
    """
    複数の特徴量選択手法を組み合わせるアンサンブル
    
    各手法で選択された特徴量の投票数でランキング。
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        k: int = 20,
        task_type: Literal['regression', 'classification'] = 'regression',
        voting_threshold: float = 0.5,
    ):
        """
        Args:
            methods: 使用する手法のリスト ['boruta', 'mrmr', 'permutation']
            k: 選択する特徴量数
            task_type: タスクタイプ
            voting_threshold: 投票閾値（この割合以上の手法で選ばれた特徴量を採用）
        """
        self.methods = methods or ['mrmr', 'permutation']
        self.k = k
        self.task_type = task_type
        self.voting_threshold = voting_threshold
        
        self.selected_features_: Optional[List[str]] = None
        self.vote_counts_: Optional[Dict[str, int]] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleFeatureSelector':
        """アンサンブル特徴量選択"""
        vote_counts = {col: 0 for col in X.columns}
        
        for method in self.methods:
            if method == 'boruta':
                selector = BorutaSelector(task_type=self.task_type, max_iter=20)
            elif method == 'mrmr':
                selector = MRMRSelector(k=self.k, task_type=self.task_type)
            elif method == 'permutation':
                selector = PermutationImportanceSelector(k=self.k, task_type=self.task_type)
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            try:
                selector.fit(X, y)
                for f in selector.selected_features_ or []:
                    vote_counts[f] += 1
            except Exception as e:
                logger.warning(f"{method} failed: {e}")
        
        # 投票数でソート
        threshold = len(self.methods) * self.voting_threshold
        selected = [
            f for f, count in sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
            if count >= threshold
        ][:self.k]
        
        self.selected_features_ = selected
        self.vote_counts_ = vote_counts
        
        logger.info(f"Ensemble: {len(X.columns)} → {len(selected)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'mrmr',
    k: int = 20,
    task_type: str = 'regression',
) -> Tuple[pd.DataFrame, FeatureSelectionResult]:
    """
    便利関数: 特徴量選択を実行
    
    Args:
        X: 特徴量
        y: ターゲット
        method: 手法 ('boruta', 'mrmr', 'permutation', 'ensemble')
        k: 選択数
        task_type: タスクタイプ
        
    Returns:
        (選択後の特徴量, 結果オブジェクト)
    """
    if method == 'boruta':
        selector = BorutaSelector(task_type=task_type)
    elif method == 'mrmr':
        selector = MRMRSelector(k=k, task_type=task_type)
    elif method == 'permutation':
        selector = PermutationImportanceSelector(k=k, task_type=task_type)
    elif method == 'ensemble':
        selector = EnsembleFeatureSelector(k=k, task_type=task_type)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_selected = selector.fit_transform(X, y)
    result = selector.get_result() if hasattr(selector, 'get_result') else FeatureSelectionResult(
        selected_features=selector.selected_features_,
        feature_scores=getattr(selector, 'feature_scores_', {}),
        method=method,
        n_original=len(X.columns),
        n_selected=len(X_selected.columns),
    )
    
    return X_selected, result
