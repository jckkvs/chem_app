# API設計ガイドライン

Chemical ML PlatformのREST API設計における標準と原則を定義します。

## 🎯 設計原則

### 1. **RESTful設計**
標準的なHTTPメソッドとステータスコードを使用

| メソッド | 用途 | 例 |
|---------|------|---|
| `GET` | リソースの取得 | `GET /api/experiments` |
| `POST` | リソースの作成 | `POST /api/experiments` |
| `PUT` | リソースの完全更新 | `PUT /api/experiments/1` |
| `PATCH` | リソースの部分更新 | `PATCH /api/experiments/1` |
| `DELETE` | リソースの削除 | `DELETE /api/experiments/1` |

### 2. **リソース指向**
エンドポイントはリソース（名詞）で表現

✅ **Good**: `/api/experiments`  
❌ **Bad**: `/api/createExperiment`

### 3. **一貫性**
命名規則、レスポンス形式、エラー処理を統一

---

## 📐 URL設計

### 基本構造

```
https://api.example.com/api/{version}/{resource}/{id}/{sub-resource}
```

### 命名規則

- **小文字使用**: `experiments`（not `Experiments`）
- **複数形**: `/api/datasets`（not `/api/dataset`）
- **ケバブケース**: `/api/molecule-properties`（パスの場合）
- **スネークケース**: `target_column`（JSONフィールドの場合）

### 階層構造

```
/api/datasets           # データセット一覧
/api/datasets/1         # 特定のデータセット
/api/datasets/1/experiments  # データセットに紐づく実験一覧
```

---

## 📊 リクエスト/レスポンス設計

### リクエストスキーマ

```python
# core/api.py

from ninja import Schema
from typing import Optional

class CreateExperimentRequest(Schema):
    """実験作成リクエスト"""
    dataset_id: int
    name: str
    target_column: str
    feature_type: str = 'rdkit'  # デフォルト値
    model_type: str = 'xgboost'
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": 1,
                "name": "Solubility Prediction",
                "target_column": "logS",
                "feature_type": "rdkit",
                "model_type": "xgboost"
            }
        }
```

### レスポンススキーマ

```python
class ExperimentResponse(Schema):
    """実験レスポンス"""
    id: int
    name: str
    status: str
    created_at: datetime
    metrics: Optional[dict] = None
    
    @staticmethod
    def from_orm(experiment: Experiment):
        """ORMモデルから変換"""
        return ExperimentResponse(
            id=experiment.id,
            name=experiment.name,
            status=experiment.status,
            created_at=experiment.created_at,
            metrics=experiment.result.metrics if hasattr(experiment, 'result') else None
        )
```

---

## 🔢 HTTPステータスコード

適切なステータスコードを返す。

| コード | 意味 | 使用例 |
|-------|------|--------|
| `200 OK` | 成功 | `GET /api/experiments/1` |
| `201 Created` | 作成成功 | `POST /api/experiments` |
| `204 No Content` | 成功（レスポンスなし） | `DELETE /api/experiments/1` |
| `400 Bad Request` | 不正なリクエスト | バリデーションエラー |
| `401 Unauthorized` | 認証失敗 | トークン無効 |
| `403 Forbidden` | 権限なし | 他人のリソースへのアクセス |
| `404 Not Found` | リソースなし | 存在しない実験ID |
| `422 Unprocessable Entity` | バリデーションエラー | 無効なSMILES |
| `500 Internal Server Error` | サーバーエラー | 予期しない例外 |

---

## ❌ エラーレスポンス設計

### 標準エラーフォーマット

```json
{
  "error": {
    "code": "INVALID_SMILES",
    "message": "提供されたSMILESが無効です",
    "details": {
      "smiles": "INVALID",
      "position": 3
    },
    "timestamp": "2026-01-20T23:45:00Z"
  }
}
```

### エラーコード体系

```python
class ErrorCode:
    """標準エラーコード"""
    # 一般エラー (1000番台)
    VALIDATION_ERROR = "ERR_1001"
    MISSING_FIELD = "ERR_1002"
    
    # リソースエラー (2000番台)
    RESOURCE_NOT_FOUND = "ERR_2001"
    RESOURCE_ALREADY_EXISTS = "ERR_2002"
    
    # 化学エラー (3000番台)
    INVALID_SMILES = "ERR_3001"
    MOLECULE_TOO_LARGE = "ERR_3002"
    
    # MLエラー (4000番台)
    MODEL_NOT_TRAINED = "ERR_4001"
    PREDICTION_FAILED = "ERR_4002"
```

### 実装例

```python
from ninja import NinjaAPI
from ninja.errors import HttpError

api = NinjaAPI()

@api.post("/experiments")
def create_experiment(request, data: CreateExperimentRequest):
    try:
        # 処理
        return {"id": 1, "status": "created"}
    
    except Dataset.DoesNotExist:
        raise HttpError(404, {
            "error": {
                "code": "ERR_2001",
                "message": f"Dataset with id {data.dataset_id} not found"
            }
        })
    
    except ValueError as e:
        raise HttpError(422, {
            "error": {
                "code": "ERR_1001",
                "message": str(e)
            }
        })
```

---

## 🔐 認証・認可（将来拡張用）

### APIキー認証（計画中）

```python
from ninja.security import HttpBearer

class AuthBearer(HttpBearer):
    def authenticate(self, request, token):
        # トークン検証
        if validate_token(token):
            return token
        return None

api = NinjaAPI(auth=AuthBearer())
```

### リクエスト例

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.example.com/api/experiments
```

---

## 📄 ページネーション

大量データの取得時はページネーション必須。

### レスポンスフォーマット

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 156,
    "pages": 8
  },
  "links": {
    "first": "/api/experiments?page=1",
    "prev": null,
    "next": "/api/experiments?page=2",
    "last": "/api/experiments?page=8"
  }
}
```

### 実装例

```python
from ninja import Query

class PaginationParams(Schema):
    page: int = 1
    per_page: int = 20

@api.get("/experiments")
def list_experiments(request, pagination: PaginationParams = Query(...)):
    offset = (pagination.page - 1) * pagination.per_page
    experiments = Experiment.objects.all()[offset:offset + pagination.per_page]
    total = Experiment.objects.count()
    
    return {
        "data": [exp.to_dict() for exp in experiments],
        "pagination": {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": total,
            "pages": (total + pagination.per_page - 1) // pagination.per_page
        }
    }
```

---

## 🔍 フィルタリング・ソート・検索

### クエリパラメータ設計

```
GET /api/experiments?status=completed&model_type=xgboost&sort=-created_at&search=solubility
```

| パラメータ | 意味 | 例 |
|----------|------|---|
| `status=completed` | フィルタ | `status=completed` |
| `model_type=xgboost` | フィルタ | `model_type=xgboost` |
| `sort=-created_at` | ソート（降順） | `-created_at` |
| `search=solubility` | 全文検索 | `search=solubility` |

### 実装例

```python
@api.get("/experiments")
def list_experiments(
    request,
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    sort: str = "-created_at",
    search: Optional[str] = None
):
    qs = Experiment.objects.all()
    
    # フィルタ
    if status:
        qs = qs.filter(status=status)
    if model_type:
        qs = qs.filter(model_type=model_type)
    
    # 検索
    if search:
        qs = qs.filter(name__icontains=search)
    
    # ソート
    order_field = sort.lstrip('-')
    if sort.startswith('-'):
        qs = qs.order_by(f'-{order_field}')
    else:
        qs = qs.order_by(order_field)
    
    return {"data": list(qs.values())}
```

---

## 🚀 非同期処理

時間のかかる処理（学習、バッチ予測）は非同期で実行。

### パターン1: ジョブIDを返す

```python
@api.post("/experiments/{id}/train")
def start_training(request, id: int):
    # バックグラウンドタスクをキュー
    job_id = enqueue_training(id)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/jobs/{job_id}"
    }

@api.get("/jobs/{job_id}")
def get_job_status(request, job_id: str):
    job = get_job(job_id)
    
    return {
        "job_id": job_id,
        "status": job.status,  # queued, running, completed, failed
        "progress": job.progress,  # 0-100
        "result": job.result if job.status == "completed" else None
    }
```

### パターン2: WebSocket通知（将来拡張）

```python
# リアルタイム進捗通知
ws://api.example.com/ws/jobs/{job_id}
```

---

## 📦 バッチ操作

複数リソースの一括処理。

### バッチ予測

```python
@api.post("/experiments/{id}/batch_predict")
def batch_predict(request, id: int, data: BatchPredictRequest):
    """
    複数SMILESの一括予測
    """
    predictions = []
    for smiles in data.smiles_list:
        pred = predict_single(id, smiles)
        predictions.append({"smiles": smiles, "prediction": pred})
    
    return {"predictions": predictions}
```

### リクエスト例

```json
{
  "smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]
}
```

---

## 📝 ドキュメント生成

Django NinjaはOpenAPI（Swagger）を自動生成。

### アクセス

```
http://localhost:8000/api/docs
```

### カスタマイズ

```python
api = NinjaAPI(
    title="Chemical ML API",
    version="1.0.0",
    description="分子物性予測プラットフォームのREST API",
)

@api.get(
    "/molecules/{smiles}/properties",
    summary="分子物性取得",
    description="SMILESから分子物性を計算します",
    response={200: MoleculePropertiesResponse},
    tags=["Molecules"]
)
def get_molecule_properties(request, smiles: str):
    pass
```

---

## 🧪 APIテスト

### テストテンプレート

```python
# core/tests/test_api.py

from ninja.testing import TestClient
from core.api import api

client = TestClient(api)

def test_create_experiment():
    """実験作成APIのテスト"""
    response = client.post("/experiments", json={
        "dataset_id": 1,
        "name": "Test Experiment",
        "target_column": "target",
        "feature_type": "rdkit",
        "model_type": "xgboost"
    })
    
    assert response.status_code == 201
    assert "id" in response.json()

def test_invalid_dataset():
    """存在しないデータセットでエラー"""
    response = client.post("/experiments", json={
        "dataset_id": 9999,  # 存在しないID
        "name": "Test",
        "target_column": "target"
    })
    
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ERR_2001"
```

---

## 📈 バージョニング（将来拡張）

API変更時の互換性維持。

### URL バージョニング

```
/api/v1/experiments
/api/v2/experiments
```

### ヘッダー バージョニング

```
Accept: application/vnd.chemml.v2+json
```

---

## ✅ チェックリスト

新しいAPIエンドポイントを追加する際のチェックリスト：

- [ ] RESTful原則に従っているか
- [ ] リクエスト/レスポンススキーマを定義したか
- [ ] 適切なHTTPステータスコードを返すか
- [ ] エラーハンドリングを実装したか
- [ ] バリデーションを追加したか
- [ ] ドキュメントコメント（docstring）を書いたか
- [ ] テストを追加したか（成功/失敗ケース）
- [ ] 認証・認可を考慮したか（必要な場合）

---

## 📚 参考資料

- [Django Ninja Documentation](https://django-ninja.rest-framework.com/)
- [REST API Design Best Practices](https://restfulapi.net/)
- [ARCHITECTURE.md](../ARCHITECTURE.md)

---

良いAPI設計で、使いやすいプラットフォームを作りましょう！🚀
