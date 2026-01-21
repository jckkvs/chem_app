"""
バッチ予測＆履歴管理エンジン

Implements: F-BATCH-001
設計思想:
- 大量分子の一括予測
- 予測履歴の保存・検索
- 進捗追跡
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """個別予測レコード"""
    smiles: str
    prediction: float
    confidence: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BatchPredictionJob:
    """バッチ予測ジョブ"""
    job_id: str
    experiment_id: int
    created_at: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    total_count: int
    completed_count: int
    results: List[PredictionRecord] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'job_id': self.job_id,
            'experiment_id': self.experiment_id,
            'created_at': self.created_at,
            'status': self.status,
            'total_count': self.total_count,
            'completed_count': self.completed_count,
            'error_message': self.error_message,
            'results': [asdict(r) for r in self.results],
        }


class BatchPredictor:
    """
    バッチ予測エンジン
    
    Features:
    - 大量SMILESの一括予測
    - 進捗コールバック
    - 結果のCSV/JSON出力
    - エラー耐性（失敗時もスキップして継続）
    
    Example:
        >>> predictor = BatchPredictor(model, extractor)
        >>> results = predictor.predict_batch(smiles_list, progress_callback=print)
        >>> predictor.save_csv(results, "predictions.csv")
    """
    
    def __init__(
        self,
        model: Any,
        feature_extractor: Any,
        preprocessor: Optional[Any] = None,
        batch_size: int = 100,
    ):
        """
        Args:
            model: 予測モデル
            feature_extractor: 特徴量抽出器
            preprocessor: 前処理器
            batch_size: 一度に処理するバッチサイズ
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.batch_size = batch_size
    
    def predict_batch(
        self,
        smiles_list: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PredictionRecord]:
        """
        バッチ予測を実行
        
        Args:
            smiles_list: SMILESリスト
            progress_callback: 進捗コールバック (completed, total) -> None
            
        Returns:
            予測レコードリスト
        """
        results = []
        total = len(smiles_list)
        
        for i in range(0, total, self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
            
            try:
                # 特徴量抽出
                features_df = self.feature_extractor.transform(batch)
                X = features_df.drop(columns=['SMILES'], errors='ignore')
                
                # 前処理
                if self.preprocessor:
                    X = self.preprocessor.transform(X)
                
                # 予測
                predictions = self.model.predict(X)
                
                # 信頼度（あれば）
                confidences = None
                if hasattr(self.model, 'predict_proba'):
                    try:
                        proba = self.model.predict_proba(X)
                        confidences = proba.max(axis=1)
                    except Exception:
                        pass
                
                # レコード作成
                for j, smi in enumerate(batch):
                    record = PredictionRecord(
                        smiles=smi,
                        prediction=float(predictions[j]),
                        confidence=float(confidences[j]) if confidences is not None else None,
                    )
                    results.append(record)
                    
            except Exception as e:
                logger.warning(f"Batch prediction error at {i}: {e}")
                # エラー時もレコード作成（NaN）
                for smi in batch:
                    results.append(PredictionRecord(
                        smiles=smi,
                        prediction=np.nan,
                    ))
            
            # 進捗コールバック
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)
        
        return results
    
    def predict_single(self, smiles: str) -> PredictionRecord:
        """単一予測"""
        results = self.predict_batch([smiles])
        return results[0] if results else PredictionRecord(smiles=smiles, prediction=np.nan)
    
    def save_csv(self, results: List[PredictionRecord], filepath: str) -> str:
        """CSV形式で保存"""
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")
        return filepath
    
    def save_json(self, results: List[PredictionRecord], filepath: str) -> str:
        """JSON形式で保存"""
        data = [asdict(r) for r in results]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Predictions saved to {filepath}")
        return filepath


class PredictionHistory:
    """
    予測履歴管理
    
    Features:
    - 予測結果の永続化
    - 履歴検索
    - 統計サマリー
    
    Example:
        >>> history = PredictionHistory()
        >>> history.save(job)
        >>> past_jobs = history.list_jobs()
    """
    
    def __init__(self, storage_dir: str = "./prediction_history"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def create_job(
        self,
        experiment_id: int,
        smiles_list: List[str],
    ) -> BatchPredictionJob:
        """新規ジョブを作成"""
        job = BatchPredictionJob(
            job_id=str(uuid.uuid4())[:8],
            experiment_id=experiment_id,
            created_at=datetime.now().isoformat(),
            status='pending',
            total_count=len(smiles_list),
            completed_count=0,
        )
        return job
    
    def save_job(self, job: BatchPredictionJob) -> str:
        """ジョブを保存"""
        filepath = os.path.join(self.storage_dir, f"{job.job_id}.json")
        with open(filepath, 'w') as f:
            json.dump(job.to_dict(), f, indent=2)
        return filepath
    
    def load_job(self, job_id: str) -> Optional[BatchPredictionJob]:
        """ジョブを読み込み"""
        filepath = os.path.join(self.storage_dir, f"{job_id}.json")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = [
            PredictionRecord(**r) for r in data.pop('results', [])
        ]
        return BatchPredictionJob(**data, results=results)
    
    def list_jobs(
        self,
        experiment_id: Optional[int] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """ジョブ一覧を取得"""
        jobs = []
        
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # フィルタ
                if experiment_id and data.get('experiment_id') != experiment_id:
                    continue
                if status and data.get('status') != status:
                    continue
                
                # 結果は除外（サマリーのみ）
                data.pop('results', None)
                jobs.append(data)
            except Exception:
                pass
        
        # 新しい順
        jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return jobs
    
    def delete_job(self, job_id: str) -> bool:
        """ジョブを削除"""
        filepath = os.path.join(self.storage_dir, f"{job_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    
    def get_statistics(self, experiment_id: Optional[int] = None) -> Dict[str, Any]:
        """統計サマリー"""
        jobs = self.list_jobs(experiment_id=experiment_id)
        
        if not jobs:
            return {'total_jobs': 0, 'total_predictions': 0}
        
        return {
            'total_jobs': len(jobs),
            'total_predictions': sum(j.get('total_count', 0) for j in jobs),
            'completed_jobs': len([j for j in jobs if j.get('status') == 'completed']),
            'failed_jobs': len([j for j in jobs if j.get('status') == 'failed']),
        }
