"""
MLflow追跡ラッパー

Implements: F-005
設計思想:
- MLflowの統一インターフェース
- モデルタイプ自動検出（型安全）
- 最新モデルの読み込み
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import mlflow
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class MLTracker:
    """
    MLflow追跡ラッパー
    
    Features:
    - 実験管理
    - パラメータ・メトリクス・アーティファクトのログ
    - モデルの保存・読み込み
    
    Example:
        >>> tracker = MLTracker(experiment_name="my_experiment")
        >>> with tracker.start_run(run_name="run_1"):
        ...     tracker.log_params({"lr": 0.01})
        ...     tracker.log_metrics({"accuracy": 0.95})
        ...     tracker.log_model(model, "model")
    """
    
    def __init__(self, experiment_name: str = "chem_ml_experiment"):
        """
        Args:
            experiment_name: MLflow実験名
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.debug(f"MLflow実験設定: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """
        MLflow runを開始
        
        Args:
            run_name: run名
            
        Returns:
            MLflow run context manager
        """
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        パラメータをログ
        
        Args:
            params: パラメータ辞書
        """
        # 値をMLflow互換形式に変換
        safe_params = {}
        for k, v in params.items():
            if isinstance(v, (list, dict)):
                safe_params[k] = str(v)
            else:
                safe_params[k] = v
        
        mlflow.log_params(safe_params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        メトリクスをログ
        
        Args:
            metrics: メトリクス辞書
        """
        mlflow.log_metrics(metrics)
    
    def log_model(self, model: BaseEstimator, artifact_path: str) -> None:
        """
        モデルをログ
        
        Args:
            model: モデルオブジェクト
            artifact_path: アーティファクトパス
        """
        # 型に基づいてログ方法を選択
        model_type = type(model).__module__
        
        if 'sklearn' in model_type:
            mlflow.sklearn.log_model(model, artifact_path)
        elif 'xgboost' in model_type:
            mlflow.xgboost.log_model(model, artifact_path)
        elif 'lightgbm' in model_type:
            mlflow.lightgbm.log_model(model, artifact_path)
        else:
            # フォールバック
            mlflow.sklearn.log_model(model, artifact_path)
        
        logger.debug(f"モデルログ: {artifact_path}")
    
    def log_artifact(self, local_path: str) -> None:
        """
        アーティファクトをログ
        
        Args:
            local_path: ローカルファイルパス
        """
        mlflow.log_artifact(local_path)
        logger.debug(f"アーティファクトログ: {local_path}")
    
    def load_latest_model(self) -> Optional[Any]:
        """
        最新のモデルを読み込み
        
        Returns:
            モデルオブジェクト（失敗時はNone）
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(f"実験が見つかりません: {self.experiment_name}")
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            
            if runs.empty:
                logger.warning("runが見つかりません")
                return None
            
            run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{run_id}/model"
            
            return mlflow.pyfunc.load_model(model_uri)
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return None
    
    def load_model_by_run_id(self, run_id: str) -> Optional[Any]:
        """
        指定したrun_idからモデルを読み込み
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            モデルオブジェクト
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"モデル読み込みエラー (run_id={run_id}): {e}")
            return None
