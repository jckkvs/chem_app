"""
特徴量抽出器テンプレート

新しい特徴量抽出器を作成する際のテンプレートです。
このファイルをコピーして、必要箇所を実装してください。

使い方:
1. このファイルを core/services/features/my_extractor.py にコピー
2. クラス名を変更（例: MyFeatureExtractor）
3. __init__、transform、descriptor_names を実装
4. （オプション）fit メソッドをオーバーライド（Statefulの場合）
5. core/services/features/__init__.py でエクスポート
"""

from .base import BaseFeatureExtractor
import pandas as pd
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TemplateFeatureExtractor(BaseFeatureExtractor):
    """
    【TODO】ここに特徴量抽出器の説明を記載
    
    Implements: F-XXX-001
    
    Example:
        >>> extractor = TemplateFeatureExtractor(param1=value1)
        >>> features = extractor.transform(['CCO', 'c1ccccc1'])
        >>> print(features.shape)
        (2, 10)
    """
    
    def __init__(
        self,
        param1: str = 'default_value',
        param2: int = 100,
        **kwargs
    ):
        """
        【TODO】パラメータの説明を記載
        
        Args:
            param1: パラメータ1の説明
            param2: パラメータ2の説明
            **kwargs: 基底クラスに渡す追加パラメータ
        """
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
        
        # 【TODO】必要な初期化処理を追加
        # 例: self.calculator = SomeCalculator()
    
    def fit(
        self,
        smiles_list: List[str],
        y: Optional[Any] = None
    ) -> 'TemplateFeatureExtractor':
        """
        【オプション】Stateful抽出器の場合のみオーバーライド
        
        Stateless抽出器（RDKit, XTB等）の場合は不要。
        Stateful抽出器（UMAP等）の場合に実装。
        
        Args:
            smiles_list: SMILESのリスト
            y: ターゲット変数（Supervised学習用）
            
        Returns:
            self
        """
        # 【TODO】学習処理を実装（必要な場合のみ）
        # 例:
        # features = self.transform(smiles_list)
        # self.reducer = UMAP()
        # self.reducer.fit(features, y)
        
        self._is_fitted = True
        return self
    
    def transform(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        【必須】SMILESを特徴量に変換
        
        Args:
            smiles_list: SMILESのリスト
            
        Returns:
            pd.DataFrame: 特徴量DataFrame（行=分子、列=記述子）
        """
        # 【TODO】ここに変換ロジックを実装
        
        features = []
        
        for smiles in smiles_list:
            try:
                # 例: RDKitで分子オブジェクト作成
                # mol = Chem.MolFromSmiles(smiles)
                # if mol is None:
                #     features.append([None] * self.n_descriptors)
                #     continue
                
                # 特徴量計算
                # feature_vector = self._calculate_features(mol)
                # features.append(feature_vector)
                
                # プレースホルダー（実装時に削除）
                features.append([0.0] * 10)
                
            except Exception as e:
                logger.warning(f"Failed to process {smiles}: {e}")
                features.append([None] * self.n_descriptors)
        
        # DataFrameに変換
        df = pd.DataFrame(
            features,
            columns=self.descriptor_names
        )
        
        return df
    
    def _calculate_features(self, mol) -> List[float]:
        """
        【内部メソッド】個別分子の特徴量計算
        
        Args:
            mol: 分子オブジェクト
            
        Returns:
            List[float]: 特徴量ベクトル
        """
        # 【TODO】ここに計算ロジックを実装
        # 例:
        # return [
        #     Descriptors.MolWt(mol),
        #     Descriptors.LogP(mol),
        #     Descriptors.TPSA(mol),
        # ]
        
        return [0.0] * self.n_descriptors
    
    @property
    def descriptor_names(self) -> List[str]:
        """
        【必須】記述子名のリスト
        
        Returns:
            List[str]: 記述子名
        """
        # 【TODO】実際の記述子名を返す
        # 例:
        # return ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
        
        return [f'feature_{i}' for i in range(10)]
    
    def save(self, path: str) -> None:
        """
        【オプション】Stateful抽出器の場合のみオーバーライド
        
        Args:
            path: 保存先パス
        """
        import joblib
        
        save_data = {
            'config': self.config,
            'param1': self.param1,
            'param2': self.param2,
            'is_fitted': self._is_fitted,
            # 【TODO】保存が必要な他の状態を追加
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Saved extractor to {path}")
    
    def load(self, path: str) -> 'TemplateFeatureExtractor':
        """
        【オプション】Stateful抽出器の場合のみオーバーライド
        
        Args:
            path: 読み込み元パス
            
        Returns:
            self
        """
        import joblib
        
        data = joblib.load(path)
        self.config = data.get('config', {})
        self.param1 = data.get('param1')
        self.param2 = data.get('param2')
        self._is_fitted = data.get('is_fitted', False)
        # 【TODO】他の状態を復元
        
        logger.info(f"Loaded extractor from {path}")
        return self


# 【TODO】テストコード例（core/tests/test_my_extractor.py に作成）
"""
import pytest
from core.services.features.my_extractor import TemplateFeatureExtractor

def test_init():
    extractor = TemplateFeatureExtractor(param1='test')
    assert extractor.param1 == 'test'

def test_transform():
    extractor = TemplateFeatureExtractor()
    smiles = ['CCO', 'c1ccccc1']
    
    result = extractor.transform(smiles)
    
    assert result.shape[0] == 2
    assert result.shape[1] == extractor.n_descriptors
    assert list(result.columns) == extractor.descriptor_names

def test_invalid_smiles():
    extractor = TemplateFeatureExtractor()
    result = extractor.transform(['INVALID'])
    
    # 無効なSMILESはNaNになるべき
    assert result.isna().all().all()
"""
