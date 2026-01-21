"""
TARTE (Transformer Augmented Representation of Table Entries) 特徴量抽出器

Implements: F-TARTE-001
設計思想:
- オプショナル依存関係（tarte-aiがなくても動作）
- 3つのモード: Featurizer, Finetuning, Boosting
- 遅延インポートによる依存関係分離

参考文献:
- Table Foundation Models: on knowledge pre-training for tabular learning
  (Kim et al., 2025)
- arXiv: 2505.14415
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# tarte-ai利用可能フラグ（遅延チェック）
_TARTE_AVAILABLE: Optional[bool] = None


def _check_tarte_available() -> bool:
    """tarte-aiがインストールされているかチェック（遅延評価）"""
    global _TARTE_AVAILABLE
    if _TARTE_AVAILABLE is None:
        try:
            import tarte_ai
            _TARTE_AVAILABLE = True
            logger.info(f"tarte-ai v{getattr(tarte_ai, '__version__', 'unknown')} が利用可能")
        except ImportError:
            _TARTE_AVAILABLE = False
            logger.debug("tarte-aiがインストールされていません")
    return _TARTE_AVAILABLE


def is_tarte_available() -> bool:
    """公開API: tarte-aiが利用可能かどうか"""
    return _check_tarte_available()


class TarteFeatureExtractor(BaseFeatureExtractor):
    """
    TARTE (Transformer Augmented Representation of Table Entries) 特徴量抽出器
    
    表形式データに対するTransformer事前学習モデルを使用して
    高品質な埋め込みベクトルを生成する。
    
    Modes:
    - featurizer: 事前学習モデルで埋め込みベクトル生成（デフォルト、高速）
    - finetuning: 下流タスク向けにモデルファインチューン（高精度）
    - boosting: 残差に対して逐次学習（最高精度、低速）
    
    Example:
        >>> extractor = TarteFeatureExtractor(mode="featurizer")
        >>> df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        >>> embeddings = extractor.fit_transform(df)
    
    Note:
        tarte-aiがインストールされていない場合はNaN埋めのDataFrameを返します。
        インストール: pip install tarte-ai
    """
    
    # サポートするモード
    SUPPORTED_MODES = ("featurizer", "finetuning", "boosting")
    
    # デフォルトの埋め込み次元数
    DEFAULT_EMBEDDING_DIM = 768
    
    def __init__(
        self,
        mode: str = "featurizer",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        n_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        target_column: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs
    ):
        """
        Args:
            mode: 動作モード ("featurizer", "finetuning", "boosting")
            embedding_dim: 埋め込みの次元数（featurizerモードでは無視）
            n_epochs: ファインチューニング/ブースティングのエポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            target_column: ターゲット列名（finetuning/boostingで必要）
            random_state: 乱数シード
            verbose: 詳細ログを出力するか
        """
        super().__init__(**kwargs)
        
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Supported modes: {self.SUPPORTED_MODES}"
            )
        
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_column = target_column
        self.random_state = random_state
        self.verbose = verbose
        
        # 内部状態
        self._model = None
        self._feature_names: List[str] = []
        self._actual_embedding_dim: Optional[int] = None
        
        # 利用可能性チェック
        self._tarte_available = _check_tarte_available()
        if not self._tarte_available:
            logger.warning(
                "tarte-aiがインストールされていません。"
                "NaN埋めのDataFrameを返します。インストール: pip install tarte-ai"
            )
    
    @property
    def is_available(self) -> bool:
        """tarte-aiが利用可能か"""
        return self._tarte_available
    
    def _get_tarte_model(self):
        """tarte-aiモデルを遅延ロード"""
        if not self._tarte_available:
            return None
        
        try:
            import tarte_ai
            
            if self.mode == "featurizer":
                # Featurizerモード: 事前学習モデルで埋め込み生成
                model = tarte_ai.TarteFeaturizer()
            elif self.mode == "finetuning":
                # Finetuningモード: 下流タスク向けにファインチューン
                model = tarte_ai.TarteFinetuner(
                    n_epochs=self.n_epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
            elif self.mode == "boosting":
                # Boostingモード: 残差に対して逐次学習
                model = tarte_ai.TarteBooster(
                    n_epochs=self.n_epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            
            return model
            
        except Exception as e:
            logger.error(f"tarte-aiモデルの初期化に失敗: {e}")
            return None
    
    def fit(
        self,
        data: pd.DataFrame,
        y: Optional[Any] = None
    ) -> 'TarteFeatureExtractor':
        """
        表形式データにフィット
        
        Args:
            data: 入力DataFrame（SMILESリストではなくDataFrame）
            y: ターゲット変数（finetuning/boostingで使用）
            
        Returns:
            self
        """
        if not self._tarte_available:
            self._is_fitted = True
            self._feature_names = [f"tarte_{i}" for i in range(self.embedding_dim)]
            self._actual_embedding_dim = self.embedding_dim
            return self
        
        # DataFrameでない場合はエラー
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"TarteFeatureExtractorはDataFrameを期待しますが、"
                f"{type(data).__name__}が渡されました"
            )
        
        self._model = self._get_tarte_model()
        if self._model is None:
            self._is_fitted = True
            return self
        
        try:
            if self.verbose:
                logger.info(f"TARTE {self.mode}モードでフィット開始...")
            
            # ターゲット列の処理
            if self.mode in ("finetuning", "boosting"):
                if y is None and self.target_column and self.target_column in data.columns:
                    y = data[self.target_column]
                    data = data.drop(columns=[self.target_column])
                
                if y is None:
                    logger.warning(
                        f"{self.mode}モードではターゲット変数が推奨されます"
                    )
            
            # フィット実行
            if self.mode == "featurizer":
                # Featurizerはstatelessなのでfitはno-op
                pass
            else:
                self._model.fit(data, y)
            
            # 埋め込み次元数を取得
            sample_emb = self._model.transform(data.head(1))
            self._actual_embedding_dim = sample_emb.shape[1]
            self._feature_names = [
                f"tarte_{i}" for i in range(self._actual_embedding_dim)
            ]
            
            if self.verbose:
                logger.info(
                    f"TARTE フィット完了: 埋め込み次元={self._actual_embedding_dim}"
                )
            
            self._is_fitted = True
            
        except Exception as e:
            logger.error(f"TARTE フィット失敗: {e}")
            self._is_fitted = True  # フォールバック用
            self._actual_embedding_dim = self.embedding_dim
            self._feature_names = [f"tarte_{i}" for i in range(self.embedding_dim)]
        
        return self
    
    def transform(self, data: Any) -> pd.DataFrame:
        """
        表形式データを埋め込み表現に変換
        
        Args:
            data: 入力DataFrame または SMILESリスト（互換性のため）
            
        Returns:
            pd.DataFrame: 埋め込みベクトルのDataFrame
        """
        # SMILESリストが渡された場合（互換性のため）
        if isinstance(data, list):
            logger.warning(
                "TarteFeatureExtractorはDataFrameを期待しますが、"
                "リストが渡されました。空のDataFrameを返します。"
            )
            return self._return_empty(len(data))
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"TarteFeatureExtractorはDataFrameを期待しますが、"
                f"{type(data).__name__}が渡されました"
            )
        
        n_samples = len(data)
        
        if not self._tarte_available or self._model is None:
            return self._return_empty(n_samples)
        
        try:
            if self.verbose:
                logger.info(f"TARTE transform: {n_samples}サンプル処理中...")
            
            # ターゲット列がある場合は除外
            transform_data = data
            if self.target_column and self.target_column in data.columns:
                transform_data = data.drop(columns=[self.target_column])
            
            # 変換実行
            embeddings = self._model.transform(transform_data)
            
            # numpy配列をDataFrameに変換
            if isinstance(embeddings, np.ndarray):
                df = pd.DataFrame(
                    embeddings,
                    columns=self._feature_names[:embeddings.shape[1]],
                    index=data.index
                )
            else:
                df = embeddings
            
            if self.verbose:
                logger.info(f"TARTE transform完了: shape={df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"TARTE transform失敗: {e}")
            return self._return_empty(n_samples)
    
    def _return_empty(self, n_samples: int) -> pd.DataFrame:
        """tarte-ai利用不可時の空DataFrame"""
        dim = self._actual_embedding_dim or self.embedding_dim
        cols = [f"tarte_{i}" for i in range(dim)]
        return pd.DataFrame(
            np.nan,
            index=range(n_samples),
            columns=cols
        )
    
    @property
    def descriptor_names(self) -> List[str]:
        """記述子名のリスト"""
        if self._feature_names:
            return self._feature_names
        return [f"tarte_{i}" for i in range(self.embedding_dim)]
    
    def get_params(self) -> Dict[str, Any]:
        """パラメータを取得"""
        return {
            "mode": self.mode,
            "embedding_dim": self.embedding_dim,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "target_column": self.target_column,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
    
    def __repr__(self) -> str:
        return (
            f"TarteFeatureExtractor("
            f"mode='{self.mode}', "
            f"available={self._tarte_available}, "
            f"fitted={self._is_fitted})"
        )
