"""
Applicability Domain (AD) 分析

Implements: F-AD-001
設計思想:
- QSARモデルの予測信頼性評価
- トレーニングデータ外の分子を検出
- 予測時の信頼区間推定

モデルはトレーニングデータの化学空間内でのみ信頼できる。
Applicability Domainは「モデルが信頼できる範囲」を定義。

参考文献:
- OECD QSAR Guidance: Applicability Domain 2007
- Tropsha et al., QSAR Comb. Sci. 2003
- Sahigara et al., Molecules 2012

手法:
1. 距離ベース: k-NN距離の閾値
2. 密度ベース: LOF (Local Outlier Factor)
3. 範囲ベース: Bounding Box / Convex Hull
4. レバレッジベース: Williams Plot
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ADResult:
    """Applicability Domain判定結果"""
    in_domain: np.ndarray  # 各サンプルがAD内か
    confidence: np.ndarray  # 信頼度スコア (0-1)
    distance_to_train: np.ndarray  # トレーニングデータへの距離
    
    # 統計
    n_total: int = 0
    n_in_domain: int = 0
    n_out_domain: int = 0
    
    @property
    def coverage(self) -> float:
        """AD内カバレッジ"""
        return self.n_in_domain / self.n_total if self.n_total > 0 else 0.0
    
    def summary(self) -> str:
        return (
            f"Applicability Domain: {self.n_in_domain}/{self.n_total} "
            f"({self.coverage:.1%}) in domain"
        )


class DistanceBasedAD:
    """
    距離ベースのApplicability Domain
    
    k-NN平均距離を使用し、トレーニングデータの閾値と比較。
    
    Reference: 
        Sahigara et al., "Comparison of Different Approaches to 
        Define the Applicability Domain of QSAR Models", 2012
    """
    
    def __init__(
        self,
        k: int = 5,
        threshold_percentile: float = 95,
        metric: str = 'euclidean',
    ):
        """
        Args:
            k: 近傍数
            threshold_percentile: 閾値パーセンタイル
            metric: 距離メトリック
        """
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.metric = metric
        
        self._scaler: Optional[StandardScaler] = None
        self._nn: Optional[NearestNeighbors] = None
        self._threshold: float = 0.0
        self._train_distances: Optional[np.ndarray] = None
    
    def fit(self, X_train: np.ndarray) -> 'DistanceBasedAD':
        """トレーニングデータでADを学習"""
        n_samples = len(X_train)
        
        # 標準化
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        
        # kをサンプル数に合わせて調整
        k_actual = min(self.k, n_samples - 1)
        if k_actual < 1:
            k_actual = 1
        
        # k-NNモデル構築
        self._nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=self.metric)
        self._nn.fit(X_scaled)
        
        # トレーニングデータ間の距離を計算
        distances, _ = self._nn.kneighbors(X_scaled)
        # 自分自身を除く
        self._train_distances = distances[:, 1:].mean(axis=1)
        
        # 閾値を設定
        self._threshold = np.percentile(
            self._train_distances, 
            self.threshold_percentile
        )
        
        logger.info(f"AD threshold set to {self._threshold:.4f} "
                   f"({self.threshold_percentile}th percentile)")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> ADResult:
        """テストデータのAD判定"""
        if self._nn is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        # 標準化
        X_scaled = self._scaler.transform(X_test)
        
        # k-NN距離計算
        distances, _ = self._nn.kneighbors(X_scaled)
        mean_distances = distances.mean(axis=1)
        
        # AD判定
        in_domain = mean_distances <= self._threshold
        
        # 信頼度（距離が閾値の何倍か→信頼度に変換）
        ratio = mean_distances / self._threshold
        confidence = np.clip(1 - (ratio - 1), 0, 1)
        confidence[in_domain] = np.clip(1 - ratio[in_domain] * 0.5, 0.5, 1)
        
        return ADResult(
            in_domain=in_domain,
            confidence=confidence,
            distance_to_train=mean_distances,
            n_total=len(X_test),
            n_in_domain=int(in_domain.sum()),
            n_out_domain=int((~in_domain).sum()),
        )


class LOFBasedAD:
    """
    Local Outlier Factor ベースのApplicability Domain
    
    密度ベースの外れ値検出で、局所的な密度差を考慮。
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
    ):
        """
        Args:
            n_neighbors: 近傍数
            contamination: 外れ値の想定割合
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        self._scaler: Optional[StandardScaler] = None
        self._lof: Optional[LocalOutlierFactor] = None
    
    def fit(self, X_train: np.ndarray) -> 'LOFBasedAD':
        """トレーニングデータでADを学習"""
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)
        
        # LOFは novelty=True で新規データの判定が可能
        self._lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X_train) - 1),
            contamination=self.contamination,
            novelty=True,
        )
        self._lof.fit(X_scaled)
        
        return self
    
    def predict(self, X_test: np.ndarray) -> ADResult:
        """テストデータのAD判定"""
        if self._lof is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        X_scaled = self._scaler.transform(X_test)
        
        # LOF予測（1=正常, -1=異常）
        predictions = self._lof.predict(X_scaled)
        in_domain = predictions == 1
        
        # 負のLOFスコア（小さいほど異常）を信頼度に変換
        scores = -self._lof.decision_function(X_scaled)
        # スコアを0-1に正規化
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        confidence = 1 - scores_normalized
        
        return ADResult(
            in_domain=in_domain,
            confidence=confidence,
            distance_to_train=scores,
            n_total=len(X_test),
            n_in_domain=int(in_domain.sum()),
            n_out_domain=int((~in_domain).sum()),
        )


class LeverageBasedAD:
    """
    レバレッジベースのApplicability Domain (Williams Plot)
    
    ハット行列の対角要素（レバレッジ）でトレーニング空間からの逸脱を検出。
    
    Reference: 
        Williams et al., "The importance of molecular structure in 
        QSAR models", 1986
    """
    
    def __init__(self, threshold_multiplier: float = 3.0):
        """
        Args:
            threshold_multiplier: 閾値 = multiplier * (p+1) / n
                - 標準的には3を使用
        """
        self.threshold_multiplier = threshold_multiplier
        
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._threshold: float = 0.0
        self._n_train: int = 0
        self._n_features: int = 0
    
    def fit(self, X_train: np.ndarray) -> 'LeverageBasedAD':
        """トレーニングデータでADを学習"""
        self._n_train, self._n_features = X_train.shape
        
        # 平均と共分散行列
        self._mean = X_train.mean(axis=0)
        X_centered = X_train - self._mean
        
        cov = np.cov(X_centered.T)
        
        # 共分散行列の逆行列（特異行列対策）
        try:
            self._cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
        except np.linalg.LinAlgError:
            self._cov_inv = np.eye(cov.shape[0])
        
        # 閾値: 3(p+1)/n (一般的な基準)
        self._threshold = self.threshold_multiplier * (self._n_features + 1) / self._n_train
        
        logger.info(f"Leverage threshold: {self._threshold:.4f}")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> ADResult:
        """テストデータのAD判定"""
        if self._cov_inv is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        # 各サンプルのレバレッジ計算
        X_centered = X_test - self._mean
        leverages = np.array([
            x @ self._cov_inv @ x.T for x in X_centered
        ])
        
        in_domain = leverages <= self._threshold
        
        # 信頼度（レバレッジが閾値の何倍か）
        ratio = leverages / self._threshold
        confidence = np.clip(1 - (ratio - 1) * 0.5, 0, 1)
        
        return ADResult(
            in_domain=in_domain,
            confidence=confidence,
            distance_to_train=leverages,
            n_total=len(X_test),
            n_in_domain=int(in_domain.sum()),
            n_out_domain=int((~in_domain).sum()),
        )


class EnsembleAD:
    """
    複数のAD手法を組み合わせたアンサンブル
    
    複数の視点からADを評価し、より堅牢な判定を実現。
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        voting: Literal['all', 'majority', 'any'] = 'majority',
    ):
        """
        Args:
            methods: 使用する手法 ['distance', 'lof', 'leverage']
            voting: 投票方式
                - 'all': すべての手法でAD内
                - 'majority': 過半数の手法でAD内
                - 'any': いずれかの手法でAD内
        """
        self.methods = methods or ['distance', 'lof']
        self.voting = voting
        
        self._estimators: Dict[str, object] = {}
    
    def fit(self, X_train: np.ndarray) -> 'EnsembleAD':
        """トレーニングデータでADを学習"""
        for method in self.methods:
            if method == 'distance':
                est = DistanceBasedAD()
            elif method == 'lof':
                est = LOFBasedAD()
            elif method == 'leverage':
                est = LeverageBasedAD()
            else:
                logger.warning(f"Unknown method: {method}")
                continue
            
            est.fit(X_train)
            self._estimators[method] = est
        
        return self
    
    def predict(self, X_test: np.ndarray) -> ADResult:
        """テストデータのAD判定"""
        if not self._estimators:
            raise RuntimeError("fit()を先に呼び出してください")
        
        results = {}
        for name, est in self._estimators.items():
            results[name] = est.predict(X_test)
        
        # 投票
        in_domain_votes = np.stack([r.in_domain for r in results.values()], axis=0)
        confidence_avg = np.stack([r.confidence for r in results.values()], axis=0).mean(axis=0)
        
        n_methods = len(self._estimators)
        vote_counts = in_domain_votes.sum(axis=0)
        
        if self.voting == 'all':
            in_domain = vote_counts == n_methods
        elif self.voting == 'majority':
            in_domain = vote_counts >= n_methods / 2
        else:  # 'any'
            in_domain = vote_counts >= 1
        
        # 平均距離
        avg_distance = np.stack(
            [r.distance_to_train for r in results.values()], 
            axis=0
        ).mean(axis=0)
        
        return ADResult(
            in_domain=in_domain,
            confidence=confidence_avg,
            distance_to_train=avg_distance,
            n_total=len(X_test),
            n_in_domain=int(in_domain.sum()),
            n_out_domain=int((~in_domain).sum()),
        )


class ApplicabilityDomainAnalyzer:
    """
    Applicability Domain分析の統合インターフェース
    
    Usage:
        # フィンガープリントベースのAD分析
        analyzer = ApplicabilityDomainAnalyzer(method='ensemble')
        analyzer.fit_from_smiles(train_smiles)
        result = analyzer.predict_from_smiles(test_smiles)
        print(result.summary())
    """
    
    def __init__(
        self,
        method: Literal['distance', 'lof', 'leverage', 'ensemble'] = 'ensemble',
        fp_type: str = 'morgan',
        fp_radius: int = 2,
        fp_bits: int = 1024,
    ):
        self.method = method
        self.fp_type = fp_type
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        
        self._estimator = None
        self._fitted = False
    
    def _compute_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """SMILESからフィンガープリントを計算"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            fps = []
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, self.fp_radius, nBits=self.fp_bits
                        )
                        fps.append(list(fp))
                    else:
                        fps.append([0] * self.fp_bits)
                except Exception:
                    fps.append([0] * self.fp_bits)
            
            return np.array(fps)
        except ImportError:
            logger.warning("RDKitが利用不可")
            return np.zeros((len(smiles_list), self.fp_bits))
    
    def fit(self, X_train: np.ndarray) -> 'ApplicabilityDomainAnalyzer':
        """特徴量行列でADを学習"""
        if self.method == 'distance':
            self._estimator = DistanceBasedAD()
        elif self.method == 'lof':
            self._estimator = LOFBasedAD()
        elif self.method == 'leverage':
            self._estimator = LeverageBasedAD()
        else:  # ensemble
            self._estimator = EnsembleAD()
        
        self._estimator.fit(X_train)
        self._fitted = True
        
        return self
    
    def fit_from_smiles(self, train_smiles: List[str]) -> 'ApplicabilityDomainAnalyzer':
        """SMILESリストでADを学習"""
        fps = self._compute_fingerprints(train_smiles)
        return self.fit(fps)
    
    def predict(self, X_test: np.ndarray) -> ADResult:
        """特徴量行列でAD判定"""
        if not self._fitted:
            raise RuntimeError("fit()を先に呼び出してください")
        
        return self._estimator.predict(X_test)
    
    def predict_from_smiles(self, test_smiles: List[str]) -> ADResult:
        """SMILESリストでAD判定"""
        fps = self._compute_fingerprints(test_smiles)
        return self.predict(fps)


def check_applicability_domain(
    train_smiles: List[str],
    test_smiles: List[str],
    method: str = 'ensemble',
) -> ADResult:
    """
    便利関数: Applicability Domain判定
    
    Args:
        train_smiles: トレーニングSMILES
        test_smiles: テストSMILES
        method: 判定手法
        
    Returns:
        ADResult: 判定結果
    """
    analyzer = ApplicabilityDomainAnalyzer(method=method)
    analyzer.fit_from_smiles(train_smiles)
    return analyzer.predict_from_smiles(test_smiles)
