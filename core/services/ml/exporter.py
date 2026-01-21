"""
モデルエクスポートエンジン

Implements: F-EXPORT-001
設計思想:
- 複数フォーマット対応（ONNX, Pickle, PMML）
- デプロイメント容易性
- メタデータ付きエクスポート
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Literal, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    学習済みモデルエクスポートエンジン
    
    Features:
    - Pickle/Joblib形式（Python用）
    - ONNX形式（クロスプラットフォーム）
    - メタデータ付きエクスポート
    
    Example:
        >>> exporter = ModelExporter()
        >>> exporter.export_pickle(model, "model.pkl", metadata={"version": "1.0"})
        >>> exporter.export_onnx(model, X_sample, "model.onnx")
    """
    
    def __init__(self, output_dir: str = "./exports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_pickle(
        self,
        model: Any,
        filename: str,
        preprocessor: Optional[Any] = None,
        feature_names: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Pickle/Joblib形式でエクスポート
        
        Args:
            model: 学習済みモデル
            filename: ファイル名
            preprocessor: 前処理器
            feature_names: 特徴量名リスト
            metadata: 追加メタデータ
            
        Returns:
            出力ファイルパス
        """
        output_path = os.path.join(self.output_dir, filename)
        
        export_data = {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "metadata": {
                "export_format": "pickle",
                "exported_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                **(metadata or {}),
            },
        }
        
        joblib.dump(export_data, output_path)
        logger.info(f"Model exported (pickle): {output_path}")
        
        # メタデータJSONも保存
        meta_path = output_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(export_data["metadata"], f, indent=2)
        
        return output_path
    
    def export_onnx(
        self,
        model: Any,
        X_sample: np.ndarray,
        filename: str,
        feature_names: Optional[list] = None,
    ) -> Optional[str]:
        """
        ONNX形式でエクスポート
        
        Args:
            model: 学習済みモデル
            X_sample: サンプル入力（形状推論用）
            filename: ファイル名
            feature_names: 特徴量名リスト
            
        Returns:
            出力ファイルパス（失敗時None）
        """
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            logger.error("skl2onnx is required for ONNX export. Run: pip install skl2onnx")
            return None
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # 入力形状を定義
            n_features = X_sample.shape[1] if len(X_sample.shape) > 1 else X_sample.shape[0]
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            
            # 変換
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12,
            )
            
            # 保存
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"Model exported (ONNX): {output_path}")
            
            # メタデータ
            meta = {
                "export_format": "onnx",
                "exported_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "n_features": n_features,
                "feature_names": feature_names,
            }
            with open(output_path + ".meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    def export_lightgbm_native(
        self,
        model: Any,
        filename: str,
    ) -> Optional[str]:
        """
        LightGBM native format でエクスポート
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            model.booster_.save_model(output_path)
            logger.info(f"Model exported (LightGBM native): {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"LightGBM export failed: {e}")
            return None
    
    def export_xgboost_native(
        self,
        model: Any,
        filename: str,
        format: Literal["json", "ubj"] = "json",
    ) -> Optional[str]:
        """
        XGBoost native format でエクスポート
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            model.save_model(output_path)
            logger.info(f"Model exported (XGBoost native): {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"XGBoost export failed: {e}")
            return None
    
    def load_pickle(self, filepath: str) -> Dict[str, Any]:
        """Pickle形式を読み込み"""
        return joblib.load(filepath)
    
    def load_onnx(self, filepath: str):
        """ONNX形式を読み込みして推論セッション作成"""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(filepath)
        except ImportError:
            logger.error("onnxruntime is required. Run: pip install onnxruntime")
            return None
    
    def list_exports(self) -> list:
        """エクスポート済みファイル一覧"""
        files = []
        for name in os.listdir(self.output_dir):
            if name.endswith(('.pkl', '.joblib', '.onnx', '.txt', '.json')):
                path = os.path.join(self.output_dir, name)
                files.append({
                    "filename": name,
                    "path": path,
                    "size_mb": os.path.getsize(path) / (1024 * 1024),
                    "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
                })
        return files
