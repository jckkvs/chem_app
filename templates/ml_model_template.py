"""
MLモデルテンプレート

新しいMLモデルを作成する際のテンプレートです。
このファイルをコピーして、必要箇所を実装してください。

使い方:
1. このファイルを core/services/ml/my_model.py にコピー
2. クラス名を変更（例: MyModel）
3. __init__、fit、predict を実装
4. （オプション）predict_proba をオーバーライド（分類の場合）
"""

from .base import BaseMLModel
import numpy as np
import pandas as pd
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class TemplateMLModel(BaseMLModel):
    """
    【TODO】ここにモデルの説明を記載
    
    Implements: F-ML-XXX-001
    
    Example:
        >>> model = TemplateMLModel(param1=100)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        param1: int = 100,
        param2: float = 0.1,
        **kwargs
    ):
        """
        【TODO】パラメータの説明を記載
        
        Args:
            param1: パラメータ1の説明
            param2: パラメータ2の説明
            **kwargs: 基底クラスに渡す追加パラメータ
        """
        super().__init__(
            param1=param1,
            param2=param2,
            **kwargs
        )
        
        # 【TODO】モデルを初期化
        # 例: self.model = SomeModel(param1=param1, param2=param2)
        self.model = None
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        eval_set: Optional[tuple] = None,
        verbose: bool = True,
        **kwargs
    ) -> 'TemplateMLModel':
        """
        【必須】モデルを学習
        
        Args:
            X: 特徴量（N x M）
            y: ターゲット変数（N,）
            eval_set: 検証セット（(X_val, y_val)）
            verbose: ログ出力するか
            **kwargs: その他の学習パラメータ
            
        Returns:
            self
        """
        # 【TODO】入力検証
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have the same length: {X.shape[0]} != {len(y)}")
        
        # 【TODO】学習処理を実装
        # 例:
        # self.model = SomeModel(**self.params)
        # self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        
        # メタデータ保存
        self.set_metadata('n_features', X.shape[1])
        self.set_metadata('n_samples', X.shape[0])
        
        self._is_fitted = True
        logger.info(f"Model training completed: {X.shape[0]} samples, {X.shape[1]} features")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """
        【必須】予測を実行
        
        Args:
            X: 特徴量（N x M）
            
        Returns:
            np.ndarray: 予測値（N,）
            
        Raises:
            RuntimeError: モデルが未学習の場合
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        
        # 【TODO】予測処理を実装
        # 例:
        # predictions = self.model.predict(X)
        # return predictions
        
        # プレースホルダー（実装時に削除）
        return np.zeros(X.shape[0])
    
    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray
    ) -> Optional[np.ndarray]:
        """
        【オプション】クラス確率を予測（分類モデルの場合のみ）
        
        Args:
            X: 特徴量（N x M）
            
        Returns:
            np.ndarray: クラス確率（N x C）
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet")
        
        # 【TODO】分類モデルの場合のみ実装
        # if hasattr(self.model, 'predict_proba'):
        #     return self.model.predict_proba(X)
        
        return None  # 回帰モデルの場合
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        【オプション】特徴量重要度を取得
        
        Returns:
            np.ndarray: 特徴量重要度、None（サポートされていない場合）
        """
        if not self.is_fitted:
            return None
        
        # 【TODO】特徴量重要度を取得（サポートされている場合）
        # if hasattr(self.model, 'feature_importances_'):
        #     return self.model.feature_importances_
        
        return None


# 【TODO】テストコード例（core/tests/test_my_model.py に作成）
"""
import pytest
import numpy as np
import pandas as pd
from core.services.ml.my_model import TemplateMLModel

@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f'f{i}' for i in range(5)])
    y = np.random.rand(100)
    return X, y

def test_init():
    model = TemplateMLModel(param1=200)
    assert model.params['param1'] == 200
    assert not model.is_fitted

def test_fit(sample_data):
    X, y = sample_data
    model = TemplateMLModel()
    
    model.fit(X, y)
    
    assert model.is_fitted
    assert model.metadata['n_features'] == 5
    assert model.metadata['n_samples'] == 100

def test_predict(sample_data):
    X, y = sample_data
    model = TemplateMLModel()
    model.fit(X[:80], y[:80])
    
    predictions = model.predict(X[80:])
    
    assert len(predictions) == 20
    assert isinstance(predictions, np.ndarray)

def test_predict_before_fit(sample_data):
    X, y = sample_data
    model = TemplateMLModel()
    
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(X)

def test_save_load(sample_data, tmp_path):
    X, y = sample_data
    model = TemplateMLModel(param1=150)
    model.fit(X, y)
    
    # 保存
    save_path = tmp_path / "model.pkl"
    model.save(save_path)
    
    # 読み込み
    model2 = TemplateMLModel()
    model2.load(save_path)
    
    assert model2.is_fitted
    assert model2.params['param1'] == 150
    
    # 予測が同じか
    pred1 = model.predict(X)
    pred2 = model2.predict(X)
    np.testing.assert_array_almost_equal(pred1, pred2)
"""
