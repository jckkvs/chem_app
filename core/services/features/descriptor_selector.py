"""
記述子選択エンジン - 物性タイプに応じた最適記述子セットの自動選択

Implements: F-DESC-001
設計思想:
- 全RDKit記述子(200+)を投入するのは非効率（多重共線性、計算コスト）
- 物性タイプに応じた事前定義セット + データ駆動型選択を組み合わせ
- VIF（分散膨張係数）による多重共線性除去

参考文献:
- Molecular Descriptors for Chemoinformatics (Todeschini & Consonni, 2009)
- Feature Selection for High-Dimensional Data (Guyon & Elisseeff, 2003)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

logger = logging.getLogger(__name__)


@dataclass
class DescriptorPreset:
    """物性タイプ別の記述子プリセット定義"""
    name: str
    description: str
    rdkit_descriptors: List[str]
    requires_xtb: bool = False
    xtb_descriptors: List[str] = field(default_factory=list)


# 物性タイプ別プリセット定義
DESCRIPTOR_PRESETS: Dict[str, DescriptorPreset] = {
    'solubility': DescriptorPreset(
        name='溶解度・logP',
        description='水溶性、脂溶性、分配係数の予測に最適',
        rdkit_descriptors=[
            'MolLogP', 'MolMR', 'TPSA', 
            'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
            'NumHeteroatoms', 'FractionCSP3', 'LabuteASA',
            'NumAromaticRings', 'NumSaturatedRings',
        ]
    ),
    'reactivity': DescriptorPreset(
        name='反応性・電子的性質',
        description='化学反応性、電荷分布、酸化還元性の予測に最適',
        rdkit_descriptors=[
            'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
            'NumRadicalElectrons', 'NumValenceElectrons',
            'NumAromaticRings', 'NumAliphaticRings',
        ],
        requires_xtb=True,
        xtb_descriptors=['homo_lumo_gap', 'dipole_norm', 'energy']
    ),
    'thermal': DescriptorPreset(
        name='熱的性質',
        description='融点、沸点、熱安定性の予測に最適',
        rdkit_descriptors=[
            'MolWt', 'HeavyAtomMolWt', 'HeavyAtomCount',
            'NumRotatableBonds', 'RingCount', 'NumAromaticRings',
            'BertzCT', 'Chi0', 'Chi1', 'Chi2n', 'Chi3n',
            'Kappa1', 'Kappa2', 'Kappa3',
        ]
    ),
    'admet': DescriptorPreset(
        name='ADMET・薬物動態',
        description='吸収・分布・代謝・排泄・毒性の予測に最適',
        rdkit_descriptors=[
            'MolLogP', 'MolWt', 'TPSA', 
            'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
            'NumAromaticRings', 'FractionCSP3',
            'qed',  # Quantitative Estimate of Drug-likeness
        ]
    ),
    'electronic': DescriptorPreset(
        name='電子的性質（量子化学）',
        description='バンドギャップ、電子状態の予測にはXTB必須',
        rdkit_descriptors=[
            'NumAromaticRings', 'NumAliphaticRings',
            'NumRadicalElectrons', 'NumValenceElectrons',
        ],
        requires_xtb=True,
        xtb_descriptors=['energy', 'homo_lumo_gap', 'dipole_norm']
    ),
    'structural': DescriptorPreset(
        name='構造的性質',
        description='分子形状、立体配置、柔軟性の表現',
        rdkit_descriptors=[
            'NumRotatableBonds', 'NumRings', 'NumAromaticRings',
            'NumSaturatedRings', 'NumAliphaticRings',
            'NumSpiroAtoms', 'NumBridgeheadAtoms',
            'FractionCSP3', 'Asphericity', 'Eccentricity', 'SpherocityIndex',
            'RadiusOfGyration', 'InertialShapeFactor', 'PMI1', 'PMI2', 'PMI3',
        ]
    ),
    'general': DescriptorPreset(
        name='汎用（バランス型）',
        description='特定の物性を指定しない場合の推奨セット',
        rdkit_descriptors=[
            'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
            'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
            'HeavyAtomCount', 'RingCount', 'LabuteASA',
            'Chi0', 'Chi1', 'Kappa1', 'Kappa2',
            'BertzCT', 'HallKierAlpha',
        ]
    ),
}


class DescriptorSelector:
    """
    物性タイプに応じた最適記述子セット選択エンジン
    
    Features:
    - プリセット選択: 物性タイプに応じた事前定義セット
    - 相関ベース選択: ターゲットとの相関係数で上位N件
    - 相互情報量選択: 非線形関係を捉える
    - 特徴量重要度選択: Random Forestベース
    - VIF除去: 多重共線性の高い記述子を除去
    
    Example:
        >>> selector = DescriptorSelector(preset='solubility')
        >>> X_selected = selector.fit_transform(X, y)
        >>> print(selector.selected_features_)
    """
    
    def __init__(
        self,
        preset: Optional[str] = None,
        selection_method: Literal['correlation', 'mutual_info', 'importance', 'all'] = 'correlation',
        top_k: int = 20,
        vif_threshold: float = 10.0,
        correlation_threshold: float = 0.05,
        task_type: Literal['regression', 'classification'] = 'regression',
    ):
        """
        Args:
            preset: 物性タイププリセット名 ('solubility', 'reactivity', 'thermal', etc.)
            selection_method: 選択方法 ('correlation', 'mutual_info', 'importance', 'all')
            top_k: 選択する記述子数
            vif_threshold: VIF閾値（これ以上で除去）
            correlation_threshold: 最低相関係数（これ以下は除外）
            task_type: タスクタイプ
        """
        self.preset = preset
        self.selection_method = selection_method
        self.top_k = top_k
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.task_type = task_type
        
        # 学習後の状態
        self.selected_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        self.vif_removed_: Optional[List[str]] = None
        
    @property
    def preset_info(self) -> Optional[DescriptorPreset]:
        """選択されたプリセットの情報を取得"""
        if self.preset:
            return DESCRIPTOR_PRESETS.get(self.preset)
        return None
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """利用可能なプリセット一覧を取得"""
        return {k: v.description for k, v in DESCRIPTOR_PRESETS.items()}
    
    def get_preset_descriptors(self) -> Tuple[List[str], List[str]]:
        """プリセットからRDKit記述子とXTB記述子を取得"""
        if self.preset and self.preset in DESCRIPTOR_PRESETS:
            preset = DESCRIPTOR_PRESETS[self.preset]
            return preset.rdkit_descriptors, preset.xtb_descriptors
        return [], []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DescriptorSelector':
        """
        データに基づいて最適な記述子を選択
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット変数
            
        Returns:
            self
        """
        # 1. プリセットがある場合、そのカラムのみを対象
        if self.preset:
            rdkit_descs, xtb_descs = self.get_preset_descriptors()
            target_cols = [c for c in rdkit_descs + xtb_descs if c in X.columns]
            if target_cols:
                X = X[target_cols]
            logger.info(f"プリセット '{self.preset}' を適用: {len(target_cols)}記述子")
        
        # 2. 欠損値・定数カラムの除去
        X_clean = self._clean_features(X)
        
        # 3. スコア計算
        if self.selection_method == 'correlation':
            scores = self._correlation_scores(X_clean, y)
        elif self.selection_method == 'mutual_info':
            scores = self._mutual_info_scores(X_clean, y)
        elif self.selection_method == 'importance':
            scores = self._importance_scores(X_clean, y)
        else:  # 'all' - 全手法の平均
            scores = self._combined_scores(X_clean, y)
        
        self.feature_scores_ = scores
        
        # 4. 閾値以上のスコアで上位N件を選択
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [f for f, s in sorted_features if s >= self.correlation_threshold][:self.top_k * 2]
        
        # 5. VIF除去
        if candidates:
            X_candidates = X_clean[candidates]
            selected, removed = self._remove_multicollinear(X_candidates)
            self.vif_removed_ = removed
            self.selected_features_ = selected[:self.top_k]
        else:
            self.selected_features_ = list(X_clean.columns)[:self.top_k]
            self.vif_removed_ = []
        
        logger.info(f"選択された記述子: {len(self.selected_features_)}件")
        logger.info(f"VIFで除去された記述子: {len(self.vif_removed_)}件")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """選択された記述子のみを抽出"""
        if self.selected_features_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """fit + transform"""
        return self.fit(X, y).transform(X)
    
    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """欠損値と定数カラムを除去"""
        # 数値カラムのみ
        X_num = X.select_dtypes(include=[np.number])
        
        # 欠損値が多いカラムを除去
        thresh = len(X_num) * 0.5
        X_num = X_num.dropna(axis=1, thresh=int(thresh))
        
        # 定数カラムを除去
        non_const = X_num.columns[X_num.nunique() > 1]
        X_num = X_num[non_const]
        
        # 残りの欠損値を中央値で補完
        return X_num.fillna(X_num.median())
    
    def _correlation_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """ピアソン相関係数（絶対値）"""
        scores = {}
        for col in X.columns:
            try:
                corr, _ = stats.pearsonr(X[col].values, y.values)
                scores[col] = abs(corr) if not np.isnan(corr) else 0.0
            except Exception:
                scores[col] = 0.0
        return scores
    
    def _mutual_info_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """相互情報量（非線形関係も捉える）"""
        try:
            if self.task_type == 'regression':
                mi = mutual_info_regression(X.values, y.values, random_state=42)
            else:
                mi = mutual_info_classif(X.values, y.values, random_state=42)
            
            # 0-1に正規化
            mi_max = mi.max() if mi.max() > 0 else 1
            return {col: float(mi[i] / mi_max) for i, col in enumerate(X.columns)}
        except Exception as e:
            logger.warning(f"相互情報量計算失敗: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _importance_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Random Forest特徴量重要度"""
        try:
            if self.task_type == 'regression':
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X.values, y.values)
            importances = model.feature_importances_
            
            # 正規化
            imp_max = importances.max() if importances.max() > 0 else 1
            return {col: float(importances[i] / imp_max) for i, col in enumerate(X.columns)}
        except Exception as e:
            logger.warning(f"重要度計算失敗: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _combined_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """全手法の平均スコア"""
        corr = self._correlation_scores(X, y)
        mi = self._mutual_info_scores(X, y)
        imp = self._importance_scores(X, y)
        
        combined = {}
        for col in X.columns:
            combined[col] = (corr.get(col, 0) + mi.get(col, 0) + imp.get(col, 0)) / 3
        return combined
    
    def _remove_multicollinear(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """VIF（分散膨張係数）で多重共線性を除去"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        features = list(X.columns)
        removed = []
        
        while len(features) > 1:
            X_temp = X[features].values
            
            try:
                vifs = [variance_inflation_factor(X_temp, i) for i in range(len(features))]
            except Exception:
                break
            
            max_vif_idx = np.argmax(vifs)
            max_vif = vifs[max_vif_idx]
            
            if max_vif > self.vif_threshold:
                removed_feature = features.pop(max_vif_idx)
                removed.append(removed_feature)
                logger.debug(f"VIF除去: {removed_feature} (VIF={max_vif:.2f})")
            else:
                break
        
        return features, removed
    
    def get_selection_report(self) -> pd.DataFrame:
        """選択結果のレポートを生成"""
        if self.feature_scores_ is None:
            raise RuntimeError("fit()を先に呼び出してください")
        
        data = []
        for feature, score in sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True):
            data.append({
                '記述子': feature,
                'スコア': round(score, 4),
                '選択': '✓' if feature in (self.selected_features_ or []) else '',
                'VIF除去': '✗' if feature in (self.vif_removed_ or []) else '',
            })
        
        return pd.DataFrame(data)
