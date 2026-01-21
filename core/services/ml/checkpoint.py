"""
チェックポイント＆復元エンジン

Implements: F-CKPT-001
設計思想:
- 学習途中の状態保存
- 途中から再開
- モデル・前処理器・オプティマイザー状態の一括管理
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """チェックポイントのメタデータ"""
    checkpoint_id: str
    experiment_name: str
    epoch: int
    step: int
    metrics: Dict[str, float]
    config: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)


class CheckpointManager:
    """
    学習チェックポイント管理
    
    Features:
    - 学習状態の保存（モデル、前処理器、メトリクス）
    - 中断からの再開
    - 複数チェックポイントの管理
    - 自動クリーンアップ
    
    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> manager.save(experiment_name="exp1", epoch=5, model=model, preprocessor=prep)
        >>> loaded = manager.load_latest("exp1")
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
    ):
        """
        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ
            max_checkpoints: 保持する最大チェックポイント数
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(
        self,
        experiment_name: str,
        epoch: int,
        model: Any,
        preprocessor: Optional[Any] = None,
        feature_selector: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> str:
        """
        チェックポイントを保存
        
        Args:
            experiment_name: 実験名
            epoch: エポック番号
            model: 学習済みモデル
            preprocessor: 前処理器
            feature_selector: 特徴量選択器
            metrics: 現在のメトリクス
            config: 設定
            step: ステップ番号
            
        Returns:
            チェックポイントID
        """
        checkpoint_id = f"{experiment_name}_epoch{epoch:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # メタデータ作成
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            experiment_name=experiment_name,
            epoch=epoch,
            step=step,
            metrics=metrics or {},
            config=config or {},
            created_at=datetime.now().isoformat(),
        )
        
        # ファイル保存
        with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        joblib.dump(model, os.path.join(checkpoint_path, "model.joblib"))
        
        if preprocessor:
            joblib.dump(preprocessor, os.path.join(checkpoint_path, "preprocessor.joblib"))
        
        if feature_selector:
            joblib.dump(feature_selector, os.path.join(checkpoint_path, "feature_selector.joblib"))
        
        logger.info(f"Checkpoint saved: {checkpoint_id}")
        
        # 古いチェックポイントをクリーンアップ
        self._cleanup(experiment_name)
        
        return checkpoint_id
    
    def load(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        チェックポイントを読み込み
        
        Args:
            checkpoint_id: チェックポイントID
            
        Returns:
            Dict containing model, preprocessor, metadata, etc.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        result = {}
        
        # メタデータ
        with open(os.path.join(checkpoint_path, "metadata.json"), "r") as f:
            result["metadata"] = CheckpointMetadata.from_dict(json.load(f))
        
        # モデル
        result["model"] = joblib.load(os.path.join(checkpoint_path, "model.joblib"))
        
        # 前処理器（存在する場合）
        prep_path = os.path.join(checkpoint_path, "preprocessor.joblib")
        if os.path.exists(prep_path):
            result["preprocessor"] = joblib.load(prep_path)
        
        # 特徴量選択器（存在する場合）
        selector_path = os.path.join(checkpoint_path, "feature_selector.joblib")
        if os.path.exists(selector_path):
            result["feature_selector"] = joblib.load(selector_path)
        
        logger.info(f"Checkpoint loaded: {checkpoint_id}")
        return result
    
    def load_latest(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        最新のチェックポイントを読み込み
        
        Args:
            experiment_name: 実験名
            
        Returns:
            チェックポイントデータ（なければNone）
        """
        checkpoints = self.list_checkpoints(experiment_name)
        if not checkpoints:
            return None
        
        latest = checkpoints[0]  # 最新が先頭
        return self.load(latest.checkpoint_id)
    
    def list_checkpoints(self, experiment_name: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        チェックポイント一覧を取得
        
        Args:
            experiment_name: フィルタする実験名（Noneで全件）
            
        Returns:
            メタデータリスト（新しい順）
        """
        checkpoints = []
        
        for name in os.listdir(self.checkpoint_dir):
            meta_path = os.path.join(self.checkpoint_dir, name, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = CheckpointMetadata.from_dict(json.load(f))
                        if experiment_name is None or meta.experiment_name == experiment_name:
                            checkpoints.append(meta)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint metadata: {name} - {e}")
        
        # 新しい順にソート
        checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        return checkpoints
    
    def delete(self, checkpoint_id: str) -> bool:
        """チェックポイントを削除"""
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        
        if not os.path.exists(checkpoint_path):
            return False
        
        import shutil
        shutil.rmtree(checkpoint_path)
        logger.info(f"Checkpoint deleted: {checkpoint_id}")
        return True
    
    def _cleanup(self, experiment_name: str) -> None:
        """古いチェックポイントを削除"""
        checkpoints = self.list_checkpoints(experiment_name)
        
        if len(checkpoints) > self.max_checkpoints:
            for old_ckpt in checkpoints[self.max_checkpoints:]:
                self.delete(old_ckpt.checkpoint_id)
