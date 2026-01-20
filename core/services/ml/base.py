"""
ML モデル 基底クラス

新しいMLモデルを追加する際は、このクラスを継承してください。

Implements: F-ML-BASE-001
設計思想:
- Strategy Patternによるモデルの切り替え
- fit/predict/save/loadの統一インターフェース
- メタデータ管理
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING
from pathlib import Path
import logging

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class BaseMLModel(ABC):
    """
    機械学習モデルの抽象基底クラス
    
    統一インターフェースで異なるML実装を扱う。
    
    Subclasses:
    - XGBoostModel: XGBoost実装
    - LightGBMModel: LightGBM実装
    - RandomForestModel: RandomForest実装
    - NeuralNetworkModel: 深層学習実装
    
    Example:
        >>> model = XGBoostModel(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> model.save('model.pkl')
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: モデル固有のハイパーパラメータ
        """
        self.params = kwargs
        self._is_fitted = False
        self._metadata: Dict[str, Any] = {}
        self.model = None
    
    @abstractmethod
    def fit(
        self,
        X: 'pd.DataFrame | np.ndarray',
        y: 'pd.Series | np.ndarray',
        **kwargs
    ) -> 'BaseMLModel':
        """
        モデルを学習
        
        Args:
            X: 特徴量（N x M）
            y: ターゲット変数（N,）
            **kwargs: 追加パラメータ（eval_set, early_stoppingなど）
            
        Returns:
            self（メソッドチェーン用）
            
        Raises:
            ValueError: 入力データが無効な場合
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: 'pd.DataFrame | np.ndarray'
    ) -> 'np.ndarray':
        """
        予測を実行
        
        Args:
            X: 特徴量（N x M）
            
        Returns:
            np.ndarray: 予測値（N,）
            
        Raises:
            RuntimeError: モデルが未学習の場合
        """
        pass
    
    def predict_proba(
        self,
        X: 'pd.DataFrame | np.ndarray'
    ) -> Optional['np.ndarray']:
        """
        クラス確率を予測（分類のみ）
        
        Args:
            X: 特徴量
            
        Returns:
            np.ndarray: クラス確率（N x C）、回帰の場合はNone
        """
        return None  # 分類モデルでオーバーライド
    
    def save(self, path: str | Path) -> None:
        """
        モデルを保存
        
        Args:
            path: 保存先パス
        """
        import joblib
        
        save_data = {
            'model': self.model,
            'params': self.params,
            'metadata': self._metadata,
            'is_fitted': self._is_fitted,
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str | Path) -> 'BaseMLModel':
        """
        モデルを読み込み
        
        Args:
            path: 読み込み元パス
            
        Returns:
            self
        """
        import joblib
        
        data = joblib.load(path)
        self.model = data['model']
        self.params = data.get('params', {})
        self._metadata = data.get('metadata', {})
        self._is_fitted = data.get('is_fitted', False)
        
        logger.info(f"Model loaded from {path}")
        return self
    
    @property
    def is_fitted(self) -> bool:
        """モデルが学習済みか"""
        return self._is_fitted
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """モデルのメタデータ"""
        return self._metadata
    
    def set_metadata(self, key: str, value: Any) -> None:
        """メタデータを設定"""
        self._metadata[key] = value
    
    def get_params(self) -> Dict[str, Any]:
        """ハイパーパラメータを取得"""
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseMLModel':
        """
        ハイパーパラメータを設定
        
        Args:
            **params: 更新するパラメータ
            
        Returns:
            self
        """
        self.params.update(params)
        return self
    
    def __repr__(self) -> str:
        params_str = ', '.join(f'{k}={v}' for k, v in list(self.params.items())[:3])
        return f"{self.__class__.__name__}({params_str}...)"
