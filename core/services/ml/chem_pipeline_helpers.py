"""
化学ML Pipeline ヘルパークラス

AutoFeatureUnion: 化学特徴量の自動統合
SmartColumnTransformer: データ型自動判定
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AutoFeatureUnion:
    """
    化学特徴量を自動的にFeatureUnionに統合
    
    複数の化学記述子を並列計算し、自動的にスケーリング・次元削減可能。
    
    Example:
        >>> union_builder = AutoFeatureUnion()
        >>> feature_union = union_builder.from_feature_list(
        ...     features=['morgan_fp', 'rdkit', 'maccs'],
        ...     smiles_column='SMILES',
        ...     auto_scale=True
        ... )
        >>> X_transformed = feature_union.fit_transform(df)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def from_feature_list(
        self,
        features: List[str],
        smiles_column: str,
        auto_scale: bool = True,
        auto_select: Union[bool, int] = False,
        n_jobs: int = -1,
    ) -> FeatureUnion:
        """
        特徴量リストからFeatureUnion自動構築
        
        Args:
            features: 特徴量リスト
                例: ['morgan_fp', 'rdkit', 'maccs']
                    ['morgan_fp:n_bits=2048,radius=3', 'rdkit']
            smiles_column: SMILESカラム名
            auto_scale: 自動スケーリング（各特徴量をスケーリング）
            auto_select: 自動特徴選択（True or 特徴数）
            n_jobs: 並列実行数
        
        Returns:
            FeatureUnion
        """
        transformers = []
        
        for feature_spec in features:
            # 特徴量仕様をパース
            transformer = self._create_feature_transformer(
                feature_spec, smiles_column, auto_scale
            )
            
            # 特徴名を生成
            feature_name = feature_spec.split(':')[0]
            
            transformers.append((feature_name, transformer))
        
        feature_union = FeatureUnion(
            transformer_list=transformers,
            n_jobs=n_jobs
        )
        
        # 自動特徴選択（オプション）
        if auto_select:
            if isinstance(auto_select, bool):
                k = 100  # デフォルト
            else:
                k = auto_select
            
            from sklearn.feature_selection import SelectKBest, f_regression
            from sklearn.pipeline import Pipeline
            
            return Pipeline([
                ('union', feature_union),
                ('select', SelectKBest(f_regression, k=k))
            ])
        
        return feature_union
    
    def _create_feature_transformer(
        self,
        feature_spec: str,
        smiles_column: str,
        auto_scale: bool
    ) -> Pipeline:
        """特徴量仕様からTransformer作成"""
        
        # 'morgan_fp:n_bits=2048,radius=3' をパース
        if ':' in feature_spec:
            feature_type, params_str = feature_spec.split(':', 1)
            params = self._parse_params(params_str)
        else:
            feature_type = feature_spec
            params = {}
        
        # 特徴抽出器を作成
        extractor = self._create_feature_extractor(feature_type, params)
        
        # SMILESカラム選択 + 特徴抽出
        steps = [
            ('select_smiles', SmilesColumnSelector(smiles_column)),
            ('extract', extractor)
        ]
        
        # 自動スケーリング
        if auto_scale:
            steps.append(('scale', StandardScaler()))
        
        return Pipeline(steps)
    
    def _create_feature_extractor(
        self,
        feature_type: str,
        params: Dict[str, Any]
    ) -> BaseEstimator:
        """特徴抽出器を作成"""
        
        if feature_type == 'morgan_fp':
            from core.services.features.fingerprint import FingerprintGenerator
            return FingerprintGenerator(fp_type='morgan', **params)
        
        elif feature_type == 'maccs':
            from core.services.features.fingerprint import FingerprintGenerator
            return FingerprintGenerator(fp_type='maccs', **params)
        
        elif feature_type == 'rdkit':
            from core.services.features.rdkit_eng import RDKitFeatureExtractor
            return RDKitFeatureExtractor(**params)
        
        elif feature_type == 'rdkit_fp':
            from core.services.features.fingerprint import FingerprintGenerator
            return FingerprintGenerator(fp_type='rdkit', **params)
        
        elif feature_type == 'avalon':
            from core.services.features.fingerprint import FingerprintGenerator
            return FingerprintGenerator(fp_type='avalon', **params)
        
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
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


class SmartColumnTransformer:
    """
    データ型を自動判定してColumnTransformer構築
    
    SMILESカラム、連続値、カテゴリ変数を自動検出し、
    それぞれに最適な前処理を適用。
    
    Example:
        >>> transformer = SmartColumnTransformer()
        >>> transformer.fit_auto(
        ...     df,
        ...     smiles_columns='auto',  # 自動検出
        ...     target_column='Solubility'
        ... )
        >>> X_transformed = transformer.transform(df)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.smiles_columns_: List[str] = []
        self.continuous_columns_: List[str] = []
        self.categorical_columns_: List[str] = []
    
    def fit_auto(
        self,
        df: pd.DataFrame,
        smiles_columns: Union[str, List[str], Literal['auto']] = 'auto',
        continuous_columns: Union[List[str], Literal['auto']] = 'auto',
        categorical_columns: Union[List[str], Literal['auto']] = 'auto',
        target_column: Optional[str] = None,
    ):
        """
        データフレームから自動でColumnTransformer構築
        
        Args:
            df: DataFrame
            smiles_columns: SMILESカラム or 'auto'
            continuous_columns: 連続値カラム or 'auto'
            categorical_columns: カテゴリカラム or 'auto'
            target_column: ターゲットカラム（除外）
        """
        # 除外カラム
        exclude_cols = [target_column] if target_column else []
        
        # SMILESカラム検出
        if smiles_columns == 'auto':
            self.smiles_columns_ = self._detect_smiles_columns(df, exclude_cols)
        elif isinstance(smiles_columns, str):
            self.smiles_columns_ = [smiles_columns]
        else:
            self.smiles_columns_ = smiles_columns
        
        exclude_cols.extend(self.smiles_columns_)
        
        # 連続値カラム検出
        if continuous_columns == 'auto':
            self.continuous_columns_ = self._detect_continuous_columns(
                df, exclude_cols
            )
        else:
            self.continuous_columns_ = continuous_columns
        
        exclude_cols.extend(self.continuous_columns_)
        
        # カテゴリカラム検出
        if categorical_columns == 'auto':
            self.categorical_columns_ = self._detect_categorical_columns(
                df, exclude_cols
            )
        else:
            self.categorical_columns_ = categorical_columns
        
        # ColumnTransformer構築
        transformers = []
        
        # SMILES → 化学特徴化（Morgan FP）
        if self.smiles_columns_:
            from core.services.features.fingerprint import FingerprintGenerator
            for smiles_col in self.smiles_columns_:
                transformers.append((
                    f'smiles_{smiles_col}',
                    Pipeline([
                        ('select', SmilesColumnSelector(smiles_col)),
                        ('fp', FingerprintGenerator(fp_type='morgan')),
                        ('scale', StandardScaler())
                    ]),
                    [smiles_col]
                ))
        
        # 連続値 → StandardScaler
        if self.continuous_columns_:
            transformers.append((
                'continuous',
                StandardScaler(),
                self.continuous_columns_
            ))
        
        # カテゴリ → OneHotEncoder
        if self.categorical_columns_:
            from sklearn.preprocessing import OneHotEncoder
            transformers.append((
                'categorical',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                self.categorical_columns_
            ))
        
        self.column_transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """変換"""
        if self.column_transformer_ is None:
            raise ValueError("Not fitted. Call fit_auto() first.")
        return self.column_transformer_.transform(df)
    
    def fit_transform(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """フィット＆変換"""
        self.fit_auto(df, **kwargs)
        return self.transform(df)
    
    def _detect_smiles_columns(
        self,
        df: pd.DataFrame,
        exclude: List[str]
    ) -> List[str]:
        """SMILESカラムを自動検出"""
        smiles_pattern = re.compile(r'^[A-Za-z0-9@+\-\[\]\(\)=#$]+$')
        smiles_cols = []
        
        for col in df.columns:
            if col in exclude:
                continue
            
            # カラム名にSMILESが含まれる
            if 'smiles' in col.lower():
                smiles_cols.append(col)
                continue
            
            # データの形式チェック（サンプル）
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    # RDKitで検証
                    try:
                        from rdkit import Chem
                        valid_count = sum(
                            1 for s in sample if Chem.MolFromSmiles(str(s)) is not None
                        )
                        if valid_count / len(sample) > 0.8:  # 80%以上が有効
                            smiles_cols.append(col)
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"SMILES validation failed for column '{col}': {e}")
        
        return smiles_cols
    
    def _detect_continuous_columns(
        self,
        df: pd.DataFrame,
        exclude: List[str]
    ) -> List[str]:
        """連続値カラムを自動検出"""
        continuous_cols = []
        
        for col in df.columns:
            if col in exclude:
                continue
            
            # 数値型 and ユニーク値が多い
            if df[col].dtype in ['float64', 'int64']:
                n_unique = df[col].nunique()
                if n_unique > 10:  # 閾値
                    continuous_cols.append(col)
        
        return continuous_cols
    
    def _detect_categorical_columns(
        self,
        df: pd.DataFrame,
        exclude: List[str]
    ) -> List[str]:
        """カテゴリカラムを自動検出"""
        categorical_cols = []
        
        for col in df.columns:
            if col in exclude:
                continue
            
            # object型 or ユニーク値が少ない
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_cols.append(col)
        
        return categorical_cols


class SmilesColumnSelector(BaseEstimator, TransformerMixin):
    """SMILESカラムを選択するTransformer"""
    
    def __init__(self, column_name: str):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[[self.column_name]]
        return X
