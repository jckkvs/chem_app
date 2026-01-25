"""
Chemical ML Platform - Django Ninja API

Implements: F-API-001
設計思想:
- RESTful APIエンドポイント
- 適切なエラーハンドリング
- 型安全なスキーマ
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from django.conf import settings
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404
from ninja import File, Form, NinjaAPI, Schema, UploadedFile
from ninja.security import HttpBearer

from .models import Dataset, Experiment

logger = logging.getLogger(__name__)


# ==============================================================================
# Authentication
# ==============================================================================

class APITokenAuth(HttpBearer):
    """
    Simple token-based authentication
    
    For production, consider more robust solutions:
    - JWT (JSON Web Tokens)
    - OAuth2
    - Django REST Framework TokenAuthentication
    """
    def authenticate(self, request, token: str):
        from django.conf import settings
        
        expected_token = os.environ.get('API_SECRET_TOKEN')
        
        if not expected_token:
            # No token configured - warn and allow (dev mode only)
            if settings.DEBUG:
                warnings.warn(
                    "API_SECRET_TOKEN not set. API is unprotected! "
                    "Set API_SECRET_TOKEN environment variable.",
                    RuntimeWarning,
                    stacklevel=2
                )
                return token  # Allow in dev mode
            else:
                logger.error("API_SECRET_TOKEN not configured in production")
                return None  # Deny in production
        
        # Validate token
        if token == expected_token:
            return token
        
        return None  # Authentication failed


# ==============================================================================
# API Initialization
# ==============================================================================

# Protected API (requires authentication)
api = NinjaAPI(
    title="ChemML API",
    version="2.0",
    urls_namespace="chem_api",
    auth=APITokenAuth(),  # ✅ Authentication required
    description="Chemical ML Platform API - Authentication required"
)

# Public API (no authentication)
public_api = NinjaAPI(
    title="ChemML Public API", 
    version="2.0",
    urls_namespace="chem_public_api",
    description="Public endpoints (health checks, etc.)"
)


# ==============================================================================
# Public Endpoints (No Authentication Required)
# ==============================================================================

@public_api.get("/health")
def health_check(request):
    """ヘルスチェック（運用監視用）"""
    import sys

    from django.db import connection
    
    status = {"status": "healthy", "version": "2.0"}
    
    # DB接続チェック
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        status["database"] = "connected"
    except Exception as e:
        status["database"] = f"error: {e}"
        status["status"] = "degraded"
    
    # Python/Django情報
    status["python"] = sys.version.split()[0]
    
    return status


@public_api.get("/health/rdkit")
def health_rdkit(request):
    """RDKit動作確認"""
    try:
        from rdkit import Chem, rdBase
        mol = Chem.MolFromSmiles("CCO")
        return {
            "status": "ok",
            "rdkit_version": rdBase.rdkitVersion,
            "test_smiles": "CCO",
            "test_valid": mol is not None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ==============================================================================
# Protected Endpoints (Authentication Required)
# ==============================================================================


# --- Schemas ---

class DatasetOut(Schema):
    id: int
    name: str
    file_path: str
    smiles_col: str
    target_col: str


class ExperimentIn(Schema):
    dataset_id: int
    name: str
    features: List[str] = ['rdkit']
    model_type: str = 'lightgbm'


class ExperimentOut(Schema):
    id: int
    name: str
    status: str
    config: Dict
    metrics: Optional[Dict] = None
    created_at: datetime


class PredictionIn(Schema):
    smiles: str


class PredictionOut(Schema):
    prediction: float
    shap_image: Optional[str] = None


class ErrorOut(Schema):
    detail: str


class MoleculeValidation(Schema):
    smiles: str
    valid: bool
    canonical_smiles: Optional[str] = None
    error: Optional[str] = None


class MoleculeProperties(Schema):
    smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    num_rings: int
    num_atoms: int


class BatchPredictionIn(Schema):
    smiles_list: List[str]


class BatchPredictionOut(Schema):
    predictions: List[Dict]


class SuccessOut(Schema):
    success: bool
    message: str


# --- Molecule Endpoints ---

@api.post("/molecules/validate", response=MoleculeValidation)
def validate_molecule(request, payload: PredictionIn):
    """
    SMILESの検証と正規化
    """
    from rdkit import Chem
    
    mol = Chem.MolFromSmiles(payload.smiles)
    if mol is None:
        return {
            "smiles": payload.smiles,
            "valid": False,
            "canonical_smiles": None,
            "error": "Invalid SMILES structure",
        }
    
    return {
        "smiles": payload.smiles,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "error": None,
    }


@api.get("/molecules/{smiles}/properties", response={200: MoleculeProperties, 400: ErrorOut})
def get_molecule_properties(request, smiles: str):
    """
    分子の物性情報を取得
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 400, {"detail": "Invalid SMILES"}
        
        return {
            "smiles": smiles,
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "num_rings": Descriptors.RingCount(mol),
            "num_atoms": mol.GetNumAtoms(),
        }
    except Exception as e:
        logger.error(f"Property calculation failed: {e}")
        return 400, {"detail": str(e)}


@api.get("/molecules/{smiles}/svg")
def get_molecule_svg(request, smiles: str, width: int = 300, height: int = 200):
    """
    分子のSVG画像を取得
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return HttpResponse("<svg></svg>", content_type="image/svg+xml")
        
        svg = Draw.MolToImage(mol, size=(width, height))
        
        # SVG形式で生成
        from rdkit.Chem.Draw import rdMolDraw2D
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()
        
        return HttpResponse(svg_text, content_type="image/svg+xml")
        
    except Exception as e:
        logger.error(f"SVG generation failed: {e}")
        return HttpResponse(f"<svg><text>{e}</text></svg>", content_type="image/svg+xml")


# --- Similarity Search ---

class SimilaritySearchIn(Schema):
    query_smiles: str
    target_smiles_list: List[str]
    threshold: float = 0.7  # Tanimoto類似度閾値
    top_k: int = 10  # 上位k件


class SimilarityResult(Schema):
    smiles: str
    similarity: float


class SimilaritySearchOut(Schema):
    query_smiles: str
    results: List[SimilarityResult]
    threshold: float
    total_searched: int


@api.post("/molecules/similarity", response={200: SimilaritySearchOut, 400: ErrorOut})
def similarity_search(request, payload: SimilaritySearchIn):
    """
    分子類似度検索（Tanimoto係数）
    
    クエリSMILESに対して、ターゲットリストから類似度が閾値以上の分子を検索
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        # クエリ分子のフィンガープリント生成
        query_mol = Chem.MolFromSmiles(payload.query_smiles)
        if query_mol is None:
            return 400, {"detail": "Invalid query SMILES"}
        
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=2, nBits=2048)
        
        # 各ターゲットとの類似度計算
        results = []
        for target_smiles in payload.target_smiles_list:
            try:
                target_mol = Chem.MolFromSmiles(target_smiles)
                if target_mol is None:
                    continue
                
                target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=2, nBits=2048)
                similarity = DataStructs.TanimotoSimilarity(query_fp, target_fp)
                
                if similarity >= payload.threshold:
                    results.append({
                        "smiles": target_smiles,
                        "similarity": round(similarity, 4)
                    })
            except Exception:
                continue
        
        # 類似度で降順ソート、上位k件を返す
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:payload.top_k]
        
        return {
            "query_smiles": payload.query_smiles,
            "results": results,
            "threshold": payload.threshold,
            "total_searched": len(payload.target_smiles_list)
        }
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return 400, {"detail": str(e)}


@api.get("/molecules/{smiles}/fingerprint")
def get_fingerprint(request, smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    分子のフィンガープリントを取得（Morgan/ECFP）
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        
        # ビット位置のリストとして返す
        on_bits = list(fp.GetOnBits())
        
        return {
            "smiles": smiles,
            "fingerprint_type": f"Morgan (radius={radius})",
            "n_bits": n_bits,
            "on_bits": on_bits,
            "bit_count": len(on_bits),
            "density": round(len(on_bits) / n_bits, 4)
        }
        
    except Exception as e:
        return {"error": str(e)}


# --- Dataset Endpoints ---

@api.post("/datasets", response={200: DatasetOut, 400: ErrorOut})
def upload_dataset(
    request,
    file: UploadedFile = File(...),
    name: str = Form(...),
    smiles_col: str = Form("SMILES"),
    target_col: str = Form("target"),
):
    """
    データセットをアップロード
    """
    try:
        # アップロードディレクトリ
        upload_dir = os.path.join(settings.BASE_DIR, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # ファイル保存
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        # データベース登録
        dataset = Dataset.objects.create(
            name=name,
            file_path=file_path,
            smiles_col=smiles_col,
            target_col=target_col,
        )
        
        logger.info(f"Dataset uploaded: {name} ({file_path})")
        return dataset
        
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        return 400, {"detail": str(e)}


@api.get("/datasets", response=List[DatasetOut])
def list_datasets(request):
    """
    データセット一覧を取得
    """
    return Dataset.objects.all()


@api.delete("/datasets/{dataset_id}", response={200: SuccessOut, 404: ErrorOut})
def delete_dataset(request, dataset_id: int):
    """
    データセットを削除
    """
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        name = dataset.name
        file_path = dataset.file_path
        
        # ファイル削除
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # DB削除
        dataset.delete()
        
        logger.info(f"Dataset deleted: {name}")
        return {"success": True, "message": f"Dataset '{name}' deleted"}
        
    except Http404:
        return 404, {"detail": "Dataset not found"}
    except Exception as e:
        logger.error(f"Dataset deletion failed: {e}")
        return 404, {"detail": str(e)}


# --- Experiment Endpoints ---

@api.post("/experiments", response={200: ExperimentOut, 400: ErrorOut})
def create_experiment(request, payload: ExperimentIn):
    """
    新しい実験を作成し、バックグラウンドで訓練を開始
    """
    try:
        dataset = get_object_or_404(Dataset, id=payload.dataset_id)
        
        config = {
            "features": payload.features,
            "model_type": payload.model_type,
        }
        
        experiment = Experiment.objects.create(
            dataset=dataset,
            name=payload.name,
            status='PENDING',
            config=config,
        )
        
        # バックグラウンドタスク起動
        from core.tasks import run_training_task
        run_training_task(experiment.id)
        
        logger.info(f"Experiment created: {experiment.name} (ID={experiment.id})")
        return experiment
        
    except Exception as e:
        logger.error(f"Experiment creation failed: {e}")
        return 400, {"detail": str(e)}


@api.get("/experiments", response=List[ExperimentOut])
def list_experiments(request):
    """
    実験一覧を取得
    """
    return Experiment.objects.all().order_by('-created_at')


@api.get("/experiments/{experiment_id}", response={200: ExperimentOut, 404: ErrorOut})
def get_experiment(request, experiment_id: int):
    """
    実験詳細を取得
    """
    experiment = get_object_or_404(Experiment, id=experiment_id)
    return experiment


@api.delete("/experiments/{experiment_id}", response={200: SuccessOut, 404: ErrorOut})
def delete_experiment(request, experiment_id: int):
    """
    実験を削除
    """
    try:
        experiment = get_object_or_404(Experiment, id=experiment_id)
        name = experiment.name
        experiment.delete()
        
        logger.info(f"Experiment deleted: {name}")
        return {"success": True, "message": f"Experiment '{name}' deleted"}
        
    except Http404:
        return 404, {"detail": "Experiment not found"}
    except Exception as e:
        logger.error(f"Experiment deletion failed: {e}")
        return 404, {"detail": str(e)}


@api.post("/experiments/{experiment_id}/batch_predict", response={200: BatchPredictionOut, 400: ErrorOut})
def batch_predict_experiment(request, experiment_id: int, payload: BatchPredictionIn):
    """
    複数SMILESに対してバッチ予測を実行
    """
    experiment = get_object_or_404(Experiment, id=experiment_id)
    
    if experiment.status != 'COMPLETED':
        return 400, {"detail": "Experiment not completed"}
    
    try:
        from core.services.features.rdkit_eng import RDKitFeatureExtractor
        from core.services.ml.tracking import MLTracker

        # 特徴量抽出
        extractor = RDKitFeatureExtractor()
        X = extractor.transform(payload.smiles_list)
        X = X.drop(columns=['SMILES'], errors='ignore')
        
        # モデルロード
        tracker = MLTracker(experiment_name=experiment.name)
        model = tracker.load_latest_model()
        
        if model is None:
            return 400, {"detail": "Model not found"}
        
        # 予測
        predictions = model.predict(X)
        
        results = [
            {"smiles": smi, "prediction": float(pred)}
            for smi, pred in zip(payload.smiles_list, predictions)
        ]
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return 400, {"detail": str(e)}


@api.get("/experiments/{experiment_id}/artifacts/{filename}")
def get_experiment_artifact(request, experiment_id: int, filename: str):
    """
    実験のアーティファクト（画像等）を取得
    """
    experiment = get_object_or_404(Experiment, id=experiment_id)
    
    try:
        # MLflowから最新のrunを取得
        current_exp = mlflow.get_experiment_by_name(experiment.name)
        if not current_exp:
            raise Http404("Experiment not found in MLflow")
        
        runs = mlflow.search_runs(
            experiment_ids=[current_exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        
        if runs.empty:
            raise Http404("No runs found")
        
        run_id = runs.iloc[0].run_id
        
        # アーティファクトをダウンロード
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=filename,
        )
        
        content_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
        with open(local_path, "rb") as f:
            return HttpResponse(f.read(), content_type=content_type)
            
    except Http404:
        raise
    except Exception as e:
        logger.warning(f"Artifact not found: {filename} - {e}")
        raise Http404(f"Artifact not found: {filename}")


@api.post("/experiments/{experiment_id}/predict", response={200: PredictionOut, 400: ErrorOut})
def predict_experiment(request, experiment_id: int, payload: PredictionIn):
    """
    訓練済みモデルで予測を実行
    """
    experiment = get_object_or_404(Experiment, id=experiment_id)
    
    if experiment.status != 'COMPLETED':
        return 400, {"detail": "Experiment not completed"}
    
    try:
        from core.services.features.rdkit_eng import RDKitFeatureExtractor
        from core.services.features.uma_eng import UMAFeatureExtractor
        from core.services.features.xtb_eng import XTBFeatureExtractor
        from core.services.ml.tracking import MLTracker
        from core.services.vis.shap_eng import SHAPEngine
        
        smiles_list = [payload.smiles]
        features_df_list = []
        
        # 特徴量抽出
        if 'rdkit' in experiment.config['features']:
            features_df_list.append(RDKitFeatureExtractor().transform(smiles_list))
        
        if 'xtb' in experiment.config['features']:
            features_df_list.append(XTBFeatureExtractor().transform(smiles_list))
        
        if 'uma' in experiment.config['features']:
            # UMAモデルをMLflowからロード
            uma_eng = _load_uma_model(experiment.name)
            if uma_eng:
                features_df_list.append(uma_eng.transform(smiles_list))
        
        if not features_df_list:
            return 400, {"detail": "Feature extraction failed"}
        
        # 特徴量結合
        X_pred = pd.concat(
            [f.drop(columns=['SMILES'], errors='ignore') for f in features_df_list],
            axis=1,
        )
        
        # モデルロード
        tracker = MLTracker(experiment_name=experiment.name)
        model = tracker.load_latest_model()
        
        if model is None:
            return 400, {"detail": "Model not found"}
        
        # 予測
        prediction = model.predict(X_pred)[0]
        
        # SHAP説明
        shap_image = _generate_shap_image(model, X_pred)
        
        return {
            "prediction": float(prediction),
            "shap_image": shap_image,
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 400, {"detail": str(e)}


def _load_uma_model(experiment_name: str):
    """UMAモデルをMLflowからロード"""
    try:
        from core.services.features.uma_eng import UMAFeatureExtractor
        
        current_exp = mlflow.get_experiment_by_name(experiment_name)
        if not current_exp:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[current_exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        
        if runs.empty:
            return None
        
        run_id = runs.iloc[0].run_id
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="uma_reducer.joblib",
        )
        
        uma_eng = UMAFeatureExtractor()
        uma_eng.load(local_path)
        return uma_eng
        
    except Exception as e:
        logger.warning(f"UMA model load failed: {e}")
        return None


def _generate_shap_image(model, X: pd.DataFrame) -> Optional[str]:
    """SHAP Force Plotを生成"""
    try:
        import shap

        from core.services.vis.shap_eng import SHAPEngine
        
        shap_eng = SHAPEngine(max_samples=1)
        shap_values, explainer = shap_eng.explain(model, X)
        
        # Force plot生成
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[0]
        
        shap.force_plot(
            expected_value,
            shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            X.iloc[0],
            matplotlib=True,
            show=False,
        )
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode("utf-8")
        
    except Exception as e:
        logger.warning(f"SHAP image generation failed: {e}")
        return None
