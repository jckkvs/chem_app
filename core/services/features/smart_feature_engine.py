"""
Smart Feature Engineering Engine - 物性×データセット特性→最適特徴量

Implements: F-SMART-001
設計思想:
- 目的物性に応じた記述子プリセット適用
- データセット分析による記述子推奨
- 事前学習モデル埋め込みの統合
- ユーザーによるカスタマイズ（追加/削除）
- 継続的改善を前提とした拡張性

これは「完成」ではなく、研究者として継続的に改善すべきモジュール。

参考文献:
- Feature Engineering for ML (Zheng & Casari, 2018)
- Molecular Descriptors for Chemoinformatics (Todeschini, 2009)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .dataset_analyzer import DatasetAnalyzer, DatasetProfile
from .descriptor_presets import (
    MATERIAL_PRESETS,
    DescriptorPreset,
    get_preset,
    list_presets,
)
from .pretrained_embeddings import PretrainedEmbeddingEngine
from .rdkit_eng import RDKitFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """特徴量生成設定"""
    # 物性プリセット
    target_property: Optional[str] = None
    
    # 記述子カスタマイズ
    user_additions: List[str] = field(default_factory=list)
    user_removals: List[str] = field(default_factory=list)
    
    # 事前学習モデル
    use_pretrained: List[str] = field(default_factory=list)  # ['unimol', 'chemberta']
    
    # Morganフィンガープリント
    use_morgan_fp: bool = False
    morgan_radius: int = 2
    morgan_bits: int = 1024
    
    # XTB量子化学記述子
    use_xtb: bool = False
    
    # TARTE（表データ用）
    use_tarte: bool = False
    tarte_mode: str = "featurizer"
    
    # データセット分析
    analyze_dataset: bool = True
    
    # 後処理
    remove_low_variance: bool = True
    variance_threshold: float = 0.01
    remove_high_correlation: bool = True
    correlation_threshold: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書に変換"""
        return {
            'target_property': self.target_property,
            'user_additions': self.user_additions,
            'user_removals': self.user_removals,
            'use_pretrained': self.use_pretrained,
            'use_morgan_fp': self.use_morgan_fp,
            'use_xtb': self.use_xtb,
            'use_tarte': self.use_tarte,
        }


@dataclass
class FeatureGenerationResult:
    """特徴量生成結果"""
    features: pd.DataFrame
    
    # メタデータ
    n_samples: int = 0
    n_features: int = 0
    
    # 使用した記述子
    rdkit_descriptors: List[str] = field(default_factory=list)
    morgan_fp_used: bool = False
    xtb_descriptors: List[str] = field(default_factory=list)
    pretrained_models: List[str] = field(default_factory=list)
    
    # データセット分析結果
    dataset_profile: Optional[DatasetProfile] = None
    
    # 削除された特徴量
    removed_low_variance: List[str] = field(default_factory=list)
    removed_high_correlation: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """結果サマリー"""
        lines = [
            f"=== Feature Generation Result ===",
            f"Samples: {self.n_samples}",
            f"Features: {self.n_features}",
            f"RDKit Descriptors: {len(self.rdkit_descriptors)}",
            f"Morgan FP: {'Yes' if self.morgan_fp_used else 'No'}",
            f"XTB Descriptors: {len(self.xtb_descriptors)}",
            f"Pretrained Models: {self.pretrained_models}",
        ]
        
        if self.removed_low_variance:
            lines.append(f"Removed (low variance): {len(self.removed_low_variance)}")
        if self.removed_high_correlation:
            lines.append(f"Removed (high correlation): {len(self.removed_high_correlation)}")
        
        return "\n".join(lines)


class SmartFeatureEngine:
    """
    物性×データセット特性→最適特徴量セットを生成
    
    3つの入力源を統合:
    1. 目的物性プリセット（domain knowledge）
    2. データセット分析（data-driven）
    3. ユーザーカスタマイズ（expert knowledge）
    
    Usage:
        # 基本使用法
        engine = SmartFeatureEngine(target_property='glass_transition')
        result = engine.fit_transform(smiles_list, y)
        
        # カスタマイズ
        engine = SmartFeatureEngine(
            target_property='elastic_modulus',
            user_additions=['CustomDesc1'],
            user_removals=['NumRotatableBonds'],
            use_pretrained=['unimol'],
        )
        result = engine.fit_transform(smiles_list, y)
        
        # 新規データへの適用
        new_features = engine.transform(new_smiles)
    """
    
    def __init__(
        self,
        target_property: Optional[str] = None,
        user_additions: List[str] = None,
        user_removals: List[str] = None,
        use_pretrained: List[str] = None,
        use_morgan_fp: bool = False,
        use_xtb: bool = False,
        use_tarte: bool = False,
        analyze_dataset: bool = True,
        config: Optional[FeatureConfig] = None,
    ):
        """
        Args:
            target_property: 目的物性名 ('glass_transition', 'elastic_modulus', etc.)
            user_additions: 追加する記述子リスト
            user_removals: 削除する記述子リスト
            use_pretrained: 使用する事前学習モデル ['unimol', 'chemberta']
            use_morgan_fp: Morganフィンガープリントを使用
            use_xtb: XTB量子化学記述子を使用
            use_tarte: TARTE表データ埋め込みを使用
            analyze_dataset: データセット分析を実行
            config: FeatureConfig（個別引数より優先）
        """
        if config:
            self.config = config
        else:
            self.config = FeatureConfig(
                target_property=target_property,
                user_additions=user_additions or [],
                user_removals=user_removals or [],
                use_pretrained=use_pretrained or [],
                use_morgan_fp=use_morgan_fp,
                use_xtb=use_xtb,
                use_tarte=use_tarte,
                analyze_dataset=analyze_dataset,
            )
        
        # コンポーネント
        self._dataset_analyzer = DatasetAnalyzer()
        self._pretrained_engine = PretrainedEmbeddingEngine()
        self._rdkit_extractor: Optional[RDKitFeatureExtractor] = None
        
        # 学習後の状態
        self._fitted = False
        self._selected_descriptors: List[str] = []
        self._dataset_profile: Optional[DatasetProfile] = None
        self._feature_columns: List[str] = []
    
    @property
    def preset(self) -> Optional[DescriptorPreset]:
        """現在のプリセット"""
        if self.config.target_property:
            return get_preset(self.config.target_property)
        return None
    
    def _determine_descriptors(self) -> Tuple[List[str], List[str]]:
        """
        使用する記述子を決定
        
        Returns:
            (rdkit_descriptors, xtb_descriptors)
        """
        rdkit_descs = set()
        xtb_descs = set()
        
        # 1. プリセットから取得
        if self.preset:
            rdkit_descs.update(self.preset.rdkit_descriptors)
            xtb_descs.update(self.preset.xtb_descriptors)
            logger.info(f"プリセット '{self.config.target_property}' を適用: {len(self.preset.rdkit_descriptors)}記述子")
        else:
            # プリセットがない場合はgeneral
            general = get_preset('general')
            if general:
                rdkit_descs.update(general.rdkit_descriptors)
        
        # 2. データセット分析からの推奨を追加
        if self._dataset_profile and self._dataset_profile.recommended_descriptor_categories:
            # カテゴリ→具体的記述子のマッピング
            category_mapping = {
                'lipophilicity': ['MolLogP', 'MolMR', 'TPSA', 'LabuteASA'],
                'structural': ['FractionCSP3', 'NumRotatableBonds', 'NumAromaticRings', 'NumAliphaticRings'],
                'topological': ['BertzCT', 'Chi0', 'Chi1', 'Kappa1', 'Kappa2', 'Kappa3'],
                'electronic': ['NumHAcceptors', 'NumHDonors', 'NumAromaticCarbocycles'],
                'complexity': ['BalabanJ', 'HallKierAlpha', 'Ipc'],
                'size': ['MolWt', 'HeavyAtomMolWt', 'NumHeavyAtoms', 'NHOHCount'],
            }
            
            for category in self._dataset_profile.recommended_descriptor_categories:
                if category in category_mapping:
                    rdkit_descs.update(category_mapping[category])
                    logger.info(f"データセット分析: カテゴリ '{category}' から記述子を追加")
        
        # 3. ユーザー追加
        rdkit_descs.update(self.config.user_additions)
        
        # 4. ユーザー削除
        rdkit_descs -= set(self.config.user_removals)
        xtb_descs -= set(self.config.user_removals)
        
        return list(rdkit_descs), list(xtb_descs)
    
    def fit(
        self, 
        smiles_list: List[str], 
        y: Optional[pd.Series] = None,
        tabular_data: Optional[pd.DataFrame] = None,
    ) -> 'SmartFeatureEngine':
        """
        データセットに基づいて特徴量生成を準備
        
        Args:
            smiles_list: SMILESリスト
            y: ターゲット変数（特徴量選択に使用）
            tabular_data: 追加の表データ（TARTE用）
        """
        logger.info(f"SmartFeatureEngine.fit: {len(smiles_list)}サンプル")
        
        # 1. データセット分析
        if self.config.analyze_dataset:
            logger.info("データセット分析中...")
            self._dataset_profile = self._dataset_analyzer.analyze(smiles_list)
            
            for note in self._dataset_profile.analysis_notes:
                logger.info(f"  → {note}")
        
        # 2. 使用する記述子を決定
        rdkit_descs, xtb_descs = self._determine_descriptors()
        self._selected_descriptors = rdkit_descs
        
        # 3. RDKit抽出器を準備
        # カテゴリ指定ではなく、選択された記述子名を直接使用
        self._rdkit_extractor = RDKitFeatureExtractor()
        
        self._fitted = True
        logger.info(f"選択された記述子: {len(rdkit_descs)}件")
        
        return self
    
    def transform(
        self, 
        smiles_list: List[str],
        tabular_data: Optional[pd.DataFrame] = None,
    ) -> FeatureGenerationResult:
        """
        特徴量を生成
        
        Args:
            smiles_list: SMILESリスト
            tabular_data: 追加の表データ
            
        Returns:
            FeatureGenerationResult: 生成された特徴量と メタデータ
        """
        if not self._fitted:
            raise RuntimeError("fit()を先に呼び出してください")
        
        result = FeatureGenerationResult(
            features=pd.DataFrame(),
            n_samples=len(smiles_list),
        )
        
        feature_dfs = []
        
        # 1. RDKit記述子
        if self._selected_descriptors:
            logger.info(f"RDKit記述子を生成中: {len(self._selected_descriptors)}件")
            rdkit_df = self._rdkit_extractor.transform(smiles_list)
            
            # 選択された記述子のみを抽出
            available = [c for c in self._selected_descriptors if c in rdkit_df.columns]
            if available:
                feature_dfs.append(rdkit_df[available])
                result.rdkit_descriptors = available
        
        # 2. Morganフィンガープリント
        if self.config.use_morgan_fp or (self.preset and self.preset.morgan_fp):
            logger.info("Morganフィンガープリントを生成中...")
            morgan_df = self._generate_morgan_fp(smiles_list)
            if morgan_df is not None and not morgan_df.empty:
                feature_dfs.append(morgan_df)
                result.morgan_fp_used = True
        
        # 3. 事前学習モデル埋め込み
        if self.config.use_pretrained:
            for model_name in self.config.use_pretrained:
                if self._pretrained_engine.is_model_available(model_name):
                    logger.info(f"事前学習モデル '{model_name}' で埋め込み生成中...")
                    emb_df = self._pretrained_engine.get_embeddings_df(smiles_list, model_name)
                    if not emb_df.empty:
                        feature_dfs.append(emb_df)
                        result.pretrained_models.append(model_name)
                else:
                    logger.warning(f"事前学習モデル '{model_name}' は利用不可")
        
        # 4. TARTE（表データ）
        if self.config.use_tarte and tabular_data is not None:
            logger.info("TARTE埋め込みを生成中...")
            tarte_df = self._generate_tarte_embeddings(tabular_data)
            if tarte_df is not None and not tarte_df.empty:
                feature_dfs.append(tarte_df)
                result.pretrained_models.append('tarte')
        
        # 5. 結合
        if feature_dfs:
            combined_df = pd.concat(feature_dfs, axis=1)
            
            # 6. 後処理
            if self.config.remove_low_variance:
                combined_df, removed = self._remove_low_variance(combined_df)
                result.removed_low_variance = removed
            
            if self.config.remove_high_correlation:
                combined_df, removed = self._remove_high_correlation(combined_df)
                result.removed_high_correlation = removed
            
            result.features = combined_df
            result.n_features = combined_df.shape[1]
            self._feature_columns = list(combined_df.columns)
        
        result.dataset_profile = self._dataset_profile
        
        logger.info(result.summary())
        
        return result
    
    def fit_transform(
        self, 
        smiles_list: List[str], 
        y: Optional[pd.Series] = None,
        tabular_data: Optional[pd.DataFrame] = None,
    ) -> FeatureGenerationResult:
        """fit + transform"""
        self.fit(smiles_list, y, tabular_data)
        return self.transform(smiles_list, tabular_data)
    
    def _generate_morgan_fp(self, smiles_list: List[str]) -> Optional[pd.DataFrame]:
        """Morganフィンガープリントを生成"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            radius = self.config.morgan_radius
            bits = self.config.morgan_bits
            
            if self.preset and self.preset.morgan_fp:
                radius = self.preset.morgan_radius
                bits = self.preset.morgan_bits
            
            fps = []
            for smi in smiles_list:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
                        fps.append(list(fp))
                    else:
                        fps.append([0] * bits)
                except Exception:
                    fps.append([0] * bits)
            
            columns = [f"morgan_r{radius}_{i}" for i in range(bits)]
            return pd.DataFrame(fps, columns=columns)
            
        except ImportError:
            logger.warning("RDKitが利用不可、Morganフィンガープリントをスキップ")
            return None
    
    def _generate_tarte_embeddings(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """TARTE埋め込みを生成"""
        try:
            from .tarte_eng import TarteFeatureExtractor
            
            extractor = TarteFeatureExtractor(mode=self.config.tarte_mode)
            extractor.fit(df)
            return extractor.transform(df)
            
        except Exception as e:
            logger.warning(f"TARTE埋め込み生成失敗: {e}")
            return None
    
    def _remove_low_variance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """低分散特徴量を削除"""
        variances = df.var()
        low_var = variances[variances < self.config.variance_threshold].index.tolist()
        
        if low_var:
            logger.info(f"低分散特徴量を削除: {len(low_var)}件")
            df = df.drop(columns=low_var)
        
        return df, low_var
    
    def _remove_high_correlation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """高相関特徴量を削除"""
        if df.shape[1] < 2:
            return df, []
        
        # 欠損値を中央値で補完
        df_filled = df.fillna(df.median())
        
        # 相関行列
        corr_matrix = df_filled.corr().abs()
        
        # 上三角行列で閾値以上のペアを探す
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > self.config.correlation_threshold):
                to_drop.append(column)
        
        if to_drop:
            logger.info(f"高相関特徴量を削除: {len(to_drop)}件")
            df = df.drop(columns=to_drop)
        
        return df, to_drop
    
    def get_params(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return self.config.to_dict()
    
    def get_feature_names(self) -> List[str]:
        """生成される特徴量名を取得"""
        return self._feature_columns.copy()
    
    @staticmethod
    def list_target_properties() -> Dict[str, str]:
        """利用可能な目的物性プリセット一覧"""
        return list_presets()


def generate_smart_features(
    smiles_list: List[str],
    target_property: str = 'general',
    y: Optional[pd.Series] = None,
    **kwargs
) -> pd.DataFrame:
    """
    便利関数: スマート特徴量を生成
    
    Args:
        smiles_list: SMILESリスト
        target_property: 目的物性名
        y: ターゲット変数
        **kwargs: SmartFeatureEngineへの追加引数
        
    Returns:
        pd.DataFrame: 特徴量
    """
    engine = SmartFeatureEngine(target_property=target_property, **kwargs)
    result = engine.fit_transform(smiles_list, y)
    return result.features
