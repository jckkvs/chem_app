"""
事前学習済み分子表現モデル統合

Implements: F-PRETRAINED-001
設計思想:
- 複数の事前学習モデル（Uni-Mol, ChemBERTa, TARTE等）への統一インターフェース
- すべてオプショナル依存（インストールされていなくても動作）
- 遅延インポートによるメモリ効率化
- 継続的に新しいモデルを追加可能な設計

利用可能なモデル:
- Uni-Mol: 3D構造を考慮した分子表現 (pip install unimol-tools)
- ChemBERTa: SMILES→Transformer埋め込み (pip install transformers)
- TARTE: 表データ向けTransformer (pip install tarte-ai)

参考文献:
- Uni-Mol: Zhou et al., Nat. Mach. Intell. 2023
- ChemBERTa: Chithrananda et al., arXiv 2020
- TARTE: Kim et al., arXiv 2025
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """モデルの利用可能状態"""
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    LOAD_ERROR = "load_error"
    DISABLED = "disabled"


@dataclass
class ModelInfo:
    """モデル情報"""
    name: str
    display_name: str
    description: str
    install_command: str
    embedding_dim: int
    requires_gpu: bool = False
    status: ModelStatus = ModelStatus.NOT_INSTALLED


class BaseEmbeddingModel(ABC):
    """埋め込みモデルの基底クラス"""
    
    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """モデル情報を返す"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """モデルが利用可能か"""
        pass
    
    @abstractmethod
    def get_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """SMILESリスト → 埋め込みベクトル"""
        pass
    
    def get_embeddings_df(self, smiles_list: List[str], prefix: str = None) -> pd.DataFrame:
        """SMILESリスト → DataFrame形式の埋め込み"""
        embeddings = self.get_embeddings(smiles_list)
        prefix = prefix or self.model_info.name
        columns = [f"{prefix}_{i}" for i in range(embeddings.shape[1])]
        return pd.DataFrame(embeddings, columns=columns)


class UniMolWrapper(BaseEmbeddingModel):
    """
    Uni-Mol分子表現モデル
    
    3D構造を考慮した高精度な分子埋め込み。
    GPU推奨だがCPUでも動作。
    
    Install: pip install unimol-tools
    """
    
    _instance = None
    _model = None
    
    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="unimol",
            display_name="Uni-Mol",
            description="3D構造を考慮した分子表現モデル",
            install_command="pip install unimol-tools",
            embedding_dim=512,
            requires_gpu=False,  # CPUでも動作
        )
    
    def is_available(self) -> bool:
        try:
            import unimol_tools
            return True
        except ImportError:
            return False
    
    def _load_model(self):
        """モデルを遅延ロード"""
        if self._model is None:
            try:
                from unimol_tools import UniMolRepr
                self._model = UniMolRepr(data_type='molecule')
                logger.info("Uni-Molモデルをロードしました")
            except Exception as e:
                logger.error(f"Uni-Molモデルのロード失敗: {e}")
                raise
        return self._model
    
    def get_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """SMILESリスト → Uni-Mol埋め込み"""
        if not self.is_available():
            raise RuntimeError("unimol-toolsがインストールされていません")
        
        model = self._load_model()
        
        try:
            # Uni-Molは辞書形式で入力
            data = [{"smiles": smi} for smi in smiles_list]
            result = model.get_repr(data)
            
            # cls_repr または mol_repr を使用
            if 'cls_repr' in result:
                return np.array(result['cls_repr'])
            elif 'mol_repr' in result:
                return np.array(result['mol_repr'])
            else:
                raise ValueError("Uni-Mol出力形式が不明")
                
        except Exception as e:
            logger.error(f"Uni-Mol埋め込み生成失敗: {e}")
            # フォールバック: ゼロベクトル
            return np.zeros((len(smiles_list), self.model_info.embedding_dim))


class ChemBERTaWrapper(BaseEmbeddingModel):
    """
    ChemBERTa分子表現モデル
    
    SMILESをTransformerで処理。軽量でADMET予測に強い。
    
    Install: pip install transformers torch
    """
    
    DEFAULT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._tokenizer = None
        self._model = None
    
    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="chemberta",
            display_name="ChemBERTa",
            description="SMILES向けTransformerモデル（軽量）",
            install_command="pip install transformers torch",
            embedding_dim=768,
            requires_gpu=False,
        )
    
    def is_available(self) -> bool:
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False
    
    def _load_model(self):
        """モデルを遅延ロード"""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                
                # GPU利用可能ならGPUへ
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    logger.info(f"ChemBERTaをGPUにロード: {self.model_name}")
                else:
                    logger.info(f"ChemBERTaをCPUにロード: {self.model_name}")
                    
            except Exception as e:
                logger.error(f"ChemBERTaロード失敗: {e}")
                raise
        
        return self._tokenizer, self._model
    
    def get_embeddings(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """SMILESリスト → ChemBERTa埋め込み"""
        if not self.is_available():
            raise RuntimeError("transformersまたはtorchがインストールされていません")
        
        import torch
        
        tokenizer, model = self._load_model()
        
        all_embeddings = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            
            try:
                # トークナイズ
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                # GPU移動
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # 推論
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # [CLS]トークンの埋め込みを使用
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.warning(f"ChemBERTaバッチ処理失敗: {e}")
                # フォールバック
                all_embeddings.append(
                    np.zeros((len(batch), self.model_info.embedding_dim))
                )
        
        return np.vstack(all_embeddings)


class TARTEWrapper(BaseEmbeddingModel):
    """
    TARTE表データ表現モデル
    
    表形式データ向けのTransformer事前学習モデル。
    SMILESではなくDataFrameを入力とする。
    
    Install: pip install tarte-ai
    """
    
    def __init__(self, mode: str = "featurizer"):
        """
        Args:
            mode: "featurizer" | "finetuning" | "boosting"
        """
        self.mode = mode
        self._model = None
    
    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="tarte",
            display_name="TARTE",
            description="表形式データ向けTransformerモデル",
            install_command="pip install tarte-ai",
            embedding_dim=768,
            requires_gpu=False,
        )
    
    def is_available(self) -> bool:
        try:
            import tarte_ai
            return True
        except ImportError:
            return False
    
    def _load_model(self):
        """モデルを遅延ロード"""
        if self._model is None:
            try:
                import tarte_ai
                
                if self.mode == "featurizer":
                    self._model = tarte_ai.TarteFeaturizer()
                elif self.mode == "finetuning":
                    self._model = tarte_ai.TarteFinetuner()
                elif self.mode == "boosting":
                    self._model = tarte_ai.TarteBooster()
                else:
                    raise ValueError(f"Unknown mode: {self.mode}")
                    
                logger.info(f"TARTEモデルをロード: mode={self.mode}")
                
            except Exception as e:
                logger.error(f"TARTEロード失敗: {e}")
                raise
        
        return self._model
    
    def get_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        注意: TARTEはSMILESではなくDataFrameを想定
        SMILESリストが渡された場合は空の埋め込みを返す
        """
        logger.warning("TARTEはDataFrame入力を想定。get_embeddings_from_dfを使用してください")
        return np.zeros((len(smiles_list), self.model_info.embedding_dim))
    
    def get_embeddings_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame → TARTE埋め込み"""
        if not self.is_available():
            raise RuntimeError("tarte-aiがインストールされていません")
        
        model = self._load_model()
        
        try:
            embeddings = model.transform(df)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"TARTE埋め込み生成失敗: {e}")
            return np.zeros((len(df), self.model_info.embedding_dim))


# =============================================================================
# 統合エンジン
# =============================================================================

class PretrainedEmbeddingEngine:
    """
    事前学習済み分子表現モデルの統合エンジン
    
    Usage:
        engine = PretrainedEmbeddingEngine()
        
        # 利用可能なモデル確認
        print(engine.list_available_models())
        
        # 埋め込み取得
        embeddings = engine.get_embeddings(smiles_list, model='unimol')
        
        # 複数モデルの結合
        combined = engine.get_combined_embeddings(
            smiles_list, 
            models=['unimol', 'chemberta']
        )
    """
    
    AVAILABLE_MODELS: Dict[str, Type[BaseEmbeddingModel]] = {
        'unimol': UniMolWrapper,
        'chemberta': ChemBERTaWrapper,
        'tarte': TARTEWrapper,
    }
    
    def __init__(self):
        self._model_instances: Dict[str, BaseEmbeddingModel] = {}
    
    def _get_model(self, name: str) -> BaseEmbeddingModel:
        """モデルインスタンスを取得（キャッシュ）"""
        if name not in self._model_instances:
            if name not in self.AVAILABLE_MODELS:
                raise ValueError(f"Unknown model: {name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
            self._model_instances[name] = self.AVAILABLE_MODELS[name]()
        return self._model_instances[name]
    
    def list_all_models(self) -> Dict[str, ModelInfo]:
        """全モデル情報を取得"""
        result = {}
        for name, cls in self.AVAILABLE_MODELS.items():
            instance = cls()
            info = instance.model_info
            info.status = ModelStatus.AVAILABLE if instance.is_available() else ModelStatus.NOT_INSTALLED
            result[name] = info
        return result
    
    def list_available_models(self) -> List[str]:
        """利用可能なモデル名リスト"""
        available = []
        for name, cls in self.AVAILABLE_MODELS.items():
            if cls().is_available():
                available.append(name)
        return available
    
    def is_model_available(self, name: str) -> bool:
        """指定モデルが利用可能か"""
        if name not in self.AVAILABLE_MODELS:
            return False
        return self.AVAILABLE_MODELS[name]().is_available()
    
    def get_embeddings(
        self, 
        smiles_list: List[str], 
        model: str = 'unimol'
    ) -> np.ndarray:
        """
        指定モデルで埋め込みを取得
        
        Args:
            smiles_list: SMILESリスト
            model: モデル名
            
        Returns:
            np.ndarray: (n_samples, embedding_dim)
        """
        model_instance = self._get_model(model)
        
        if not model_instance.is_available():
            info = model_instance.model_info
            logger.warning(
                f"{info.display_name}が利用不可。インストール: {info.install_command}"
            )
            return np.zeros((len(smiles_list), info.embedding_dim))
        
        return model_instance.get_embeddings(smiles_list)
    
    def get_embeddings_df(
        self, 
        smiles_list: List[str], 
        model: str = 'unimol'
    ) -> pd.DataFrame:
        """DataFrame形式で埋め込みを取得"""
        model_instance = self._get_model(model)
        return model_instance.get_embeddings_df(smiles_list)
    
    def get_combined_embeddings(
        self, 
        smiles_list: List[str], 
        models: List[str] = None
    ) -> pd.DataFrame:
        """
        複数モデルの埋め込みを結合
        
        Args:
            smiles_list: SMILESリスト
            models: 使用するモデルリスト（Noneで利用可能な全て）
            
        Returns:
            pd.DataFrame: 結合された埋め込み
        """
        if models is None:
            models = self.list_available_models()
        
        if not models:
            logger.warning("利用可能な事前学習モデルがありません")
            return pd.DataFrame()
        
        dfs = []
        for model_name in models:
            if self.is_model_available(model_name):
                df = self.get_embeddings_df(smiles_list, model_name)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, axis=1)


def get_pretrained_embeddings(
    smiles_list: List[str],
    models: List[str] = None
) -> pd.DataFrame:
    """便利関数: 事前学習埋め込みを取得"""
    engine = PretrainedEmbeddingEngine()
    return engine.get_combined_embeddings(smiles_list, models)
