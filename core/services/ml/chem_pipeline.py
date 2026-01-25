"""
化学ML特化Pipeline Builder

Implements: F-CHEM-PIPELINE-001
設計思想:
- FeatureUnion/ColumnTransformerの化学ML実務での使い方を明確化
- ユーザーが直感的に使える高レベルAPI
- ユースケース別ヘルパー関数（QSAR/溶解度/反応予測等）

参考:
- chem_ml_pipeline_design.md
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline

logger = logging.getLogger(__name__)


class ChemMLPipelineBuilder:
    """
    化学ML特化のパイプライン構築器
    
    使いやすさを最優先し、実際の化学MLワークフローに対応。
    
    Features:
    - ユースケース別ヘルパー関数
    - 自動特徴量選択・モデル選択
    - ベストプラクティス組み込み
    
    Example - QSAR予測（3行で完結）:
        >>> builder = ChemMLPipelineBuilder()
        >>> pipeline = builder.build_qsar_pipeline(
        ...     smiles_column='SMILES', features='auto', model='auto'
        ... )
        >>> pipeline.fit(train_df, train_df['pIC50'])
        >>> predictions = pipeline.predict(test_df)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Args:
            random_state: 再現性のための乱数シード
        """
        self.random_state = random_state
    
    def build_qsar_pipeline(
        self,
        smiles_column: str,
        features: Union[List[str], Literal['auto']] = 'auto',
        model: Union[str, BaseEstimator, Literal['auto']] = 'auto',
        feature_selection: Optional[str] = None,
        cv_strategy: str = 'kfold:n_splits=5',
        optimize_hyperparams: bool = False,
        n_jobs: int = -1,
    ) -> Pipeline:
        """
        QSAR（定量的構造活性相関）予測用パイプライン自動構築
        
        複数種類の化学記述子を並列に計算→統合し、予測モデルを構築。
        
        Args:
            smiles_column: SMILESカラム名
            features: 使用する特徴量リスト or 'auto'（自動選択）
                例: ['morgan_fp', 'rdkit', 'maccs']
                    'auto' → データサイズに応じて最適な組み合わせ
            model: モデル or 'auto'（自動選択）
            feature_selection: 特徴選択手法（オプション）
                例: 'rfecv', 'k_best:k=100'
            cv_strategy: CV戦略
            optimize_hyperparams: Optunaでハイパーパラメータ最適化
            n_jobs: 並列実行数
        
        Returns:
            完全なMLパイプライン
        
        Example:
            >>> # 最もシンプル（全自動）
            >>> pipeline = builder.build_qsar_pipeline('SMILES')
            >>> 
            >>> # カスタマイズ
            >>> pipeline = builder.build_qsar_pipeline(
            ...     smiles_column='SMILES',
            ...     features=['morgan_fp:n_bits=2048', 'rdkit'],
            ...     model='random_forest',
            ...     feature_selection='rfecv',
            ...     optimize_hyperparams=True
            ... )
        """
        from core.services.ml.chem_pipeline_helpers import AutoFeatureUnion
        
        logger.info(f"QSAR Pipeline構築開始: features={features}, model={model}")
        
        # 1. 特徴量選択
        if features == 'auto':
            # データサイズに応じて自動選択（後で実装）
            features = ['morgan_fp', 'rdkit', 'maccs']
        
        # 2. FeatureUnion構築
        feature_union_builder = AutoFeatureUnion(random_state=self.random_state)
        feature_union = feature_union_builder.from_feature_list(
            features=features,
            smiles_column=smiles_column,
            auto_scale=True,
            n_jobs=n_jobs
        )
        
        # 3. モデル選択
        if model == 'auto':
            # タスクタイプに応じて自動選択
            from sklearn.ensemble import RandomForestRegressor
            model_estimator = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=n_jobs
            )
        elif isinstance(model, str):
            model_estimator = self._create_model(model)
        else:
            model_estimator = model
        
        # 4. パイプライン構築
        steps = [('features', feature_union)]
        
        # 特徴選択（オプション）
        if feature_selection:
            selector = self._create_feature_selector(feature_selection)
            steps.append(('feature_selection', selector))
        
        steps.append(('model', model_estimator))
        
        pipeline = Pipeline(steps)
        
        logger.info(f"QSAR Pipeline構築完了: {len(steps)} steps")
        
        return pipeline
    
    def build_solubility_prediction_pipeline(
        self,
        smiles_column: str,
        structure_features: List[str] = ['morgan_fp', 'rdkit'],
        physicochemical: Optional[List[str]] = None,
        quantum_chem: Optional[List[str]] = None,
        model: str = 'auto',
    ) -> Pipeline:
        """
        溶解度予測用パイプライン構築
        
        構造特徴 + 物性値 + 量子化学計算結果を統合。
        
        Args:
            smiles_column: SMILESカラム名
            structure_features: 構造特徴量
            physicochemical: 物性値カラム（MW, LogP等）
            quantum_chem: 量子化学計算カラム（HOMO, LUMO等）
            model: モデル
        
        Returns:
            Pipeline
        
        Example:
            >>> pipeline = builder.build_solubility_prediction_pipeline(
            ...     smiles_column='SMILES',
            ...     physicochemical=['MW', 'LogP', 'TPSA'],
            ...     quantum_chem=['HOMO', 'LUMO']
            ... )
        """
        logger.info("Solubility Prediction Pipeline構築")
        
        # 構造特徴のFeatureUnion
        from core.services.ml.chem_pipeline_helpers import AutoFeatureUnion
        structure_union = AutoFeatureUnion(
            random_state=self.random_state
        ).from_feature_list(
            features=structure_features,
            smiles_column=smiles_column,
            auto_scale=True
        )
        
        # ColumnTransformerで異なるタイプ統合
        transformers = [('structure', structure_union, [smiles_column])]
        
        if physicochemical:
            from sklearn.preprocessing import StandardScaler
            transformers.append(
                ('physicochemical', StandardScaler(), physicochemical)
            )
        
        if quantum_chem:
            from sklearn.preprocessing import RobustScaler
            transformers.append(
                ('quantum', RobustScaler(), quantum_chem)
            )
        
        from sklearn.compose import ColumnTransformer
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # モデル
        if model == 'auto':
            from sklearn.ensemble import GradientBoostingRegressor
            model_estimator = GradientBoostingRegressor(
                random_state=self.random_state
            )
        else:
            model_estimator = self._create_model(model)
        
        pipeline = Pipeline([
            ('features', column_transformer),
            ('model', model_estimator)
        ])
        
        return pipeline
    
    def _create_model(self, model_spec: str) -> BaseEstimator:
        """モデル仕様文字列からモデル作成"""
        from core.services.ml.model_factory import ModelFactory
        
        # 'random_forest:n_estimators=100' 形式をパース
        if ':' in model_spec:
            model_name, params_str = model_spec.split(':', 1)
            params = self._parse_params(params_str)
        else:
            model_name = model_spec
            params = {}
        
        factory = ModelFactory(task_type='regression')
        return factory.create_model(model_name, **params)
    
    def _create_feature_selector(self, selector_spec: str) -> BaseEstimator:
        """特徴選択仕様文字列からSelector作成"""
        from core.services.ml.feature_selector import FeatureSelector
        
        if ':' in selector_spec:
            method, params_str = selector_spec.split(':', 1)
            params = self._parse_params(params_str)
        else:
            method = selector_spec
            params = {}
        
        return FeatureSelector(method=method, **params)
    
    def _parse_params(self, params_str: str) -> Dict[str, Any]:
        """パラメータ文字列をパース"""
        params = {}
        for item in params_str.split(','):
            key, value = item.split('=')
            # 型推論
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
            params[key.strip()] = value
        return params


# ヘルパー用の別ファイルに分割予定（ここでは同じファイルに記載）
# 次のイテレーションでchem_pipeline_helpers.pyに移動
