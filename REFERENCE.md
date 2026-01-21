# Chemical ML Platform - 完全リファレンス

**Version**: 1.0.0  
**Last Updated**: 2026-01-21

このドキュメントは、Chemical ML Platformの全機能、全メソッド、全コマンド、全引数を網羅した包括的なリファレンスです。生成AIでの利用を想定し、1ファイルで完結しています。

---

## 📑 目次

1. [概要](#概要)
2. [インストール](#インストール)
3. [クイックスタート](#クイックスタート)
4. [コマンドラインインターフェース](#コマンドラインインターフェース)
5. [REST API](#rest-api)
6. [特徴量抽出API](#特徴量抽出api)
7. [機械学習API](#機械学習api)
8. [可視化API](#可視化api)
9. [プラグインAPI](#プラグインapi)
10. [設定](#設定)

---

## 概要

Chemical ML Platformは、分子物性予測のための包括的な機械学習プラットフォームです。

### 主要機能
- **特徴量抽出**: RDKit、XTB、UMAP、Transformer（TARTE）
- **機械学習**: XGBoost、LightGBM、RandomForest、AutoML
- **可視化**: SHAP、PDP、化学空間マップ
- **API**: REST API（16エンドポイント）
- **フロントエンド**: Django、Streamlit、PWA

### アーキテクチャ
```
Frontend (Django/Streamlit/PWA)
    ↓
API Layer (Django Ninja)
    ↓
Service Layer (Features/ML/Vis)
    ↓
Data Layer (SQLite/MLflow/Huey)
```

---

## インストール

### 基本インストール
```bash
# 仮想環境作成
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 必須パッケージ
pip install -r requirements.txt

# データベースマイグレーション
python manage.py migrate
```

### 依存パッケージ（requirements.txt）
```
# Core
Django>=4.2
django-ninja>=1.0
gunicorn>=21.0

# Data Science
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0

# Visualization
matplotlib>=3.7
seaborn>=0.12
shap>=0.44

# ML Tracking
mlflow>=2.10

# Chemistry
rdkit>=2023.09
umap-learn>=0.5

# Frontend
streamlit>=1.30
requests>=2.31

# Task Queue
huey>=2.4

# Testing
pytest>=7.4
pytest-django>=4.5
pytest-cov>=4.1
```

### オプショナルパッケージ
```bash
# XTB（量子化学）
conda install -c conda-forge xtb

# TARTE（Transformer）
pip install tarte-ai

# Uni-Mol（3D埋め込み）
pip install unimol-tools

# ChemBERTa
pip install transformers torch
```

---

## クイックスタート

### 1. サーバー起動
```bash
# Django開発サーバー
python manage.py runserver
# → http://localhost:8000

# Streamlit
cd frontend_streamlit
streamlit run app.py
# → http://localhost:8501
```

### 2. Pythonスクリプトで使用
```python
from core.services.features import RDKitFeatureExtractor
from core.services.ml.pipeline import MLPipeline

# 特徴量抽出
extractor = RDKitFeatureExtractor()
X = extractor.transform(['CCO', 'c1ccccc1', 'CC(=O)O'])

# モデル学習
pipeline = MLPipeline(
    feature_extractor=extractor,
    model_type='xgboost'
)
pipeline.fit(smiles_list, y_target)

# 予測
predictions = pipeline.predict(['CN1C=NC2=C1C(=O)N(C(=O)N2C)C'])
```

---

## コマンドラインインターフェース

### Django管理コマンド

#### `python manage.py runserver [port]`
開発サーバーを起動

**引数**:
- `port` (optional): ポート番号（デフォルト: 8000）

**例**:
```bash
python manage.py runserver 8080
```

---

#### `python manage.py migrate`
データベースマイグレーションを実行

**オプション**:
- `--fake`: 実際にはマイグレーションせず、適用済みとしてマーク
- `--fake-initial`: 初回マイグレーションのみfake
- `app_label`: 特定アプリのみマイグレーション

**例**:
```bash
python manage.py migrate core
python manage.py migrate --fake-initial
```

---

#### `python manage.py makemigrations [app_label]`
モデル変更からマイグレーションファイルを生成

**引数**:
- `app_label` (optional): 特定アプリのみ

**オプション**:
- `--dry-run`: 実際には作成せず、変更内容を表示
- `--name NAME`: マイグレーション名を指定

**例**:
```bash
python manage.py makemigrations core
python manage.py makemigrations --dry-run
```

---

#### `python manage.py createsuperuser`
管理者ユーザーを作成

**対話的入力**:
- Username
- Email
- Password

---

#### `python manage.py shell`
Djangoシェルを起動（IPythonが優先）

**例**:
```bash
python manage.py shell
>>> from core.models import Dataset
>>> Dataset.objects.all()
```

---

#### `python manage.py test [path]`
テストを実行

**引数**:
- `path` (optional): テストパス

**オプション**:
- `--keepdb`: テストDB削除をスキップ
- `--parallel N`: N並列でテスト実行

**例**:
```bash
python manage.py test core.tests
python manage.py test --parallel 4
```

---

### CLIツール（cli.py）

#### `python cli.py extract [OPTIONS]`
特徴量抽出

**引数**:
- `--input PATH`: 入力CSVファイル
- `--output PATH`: 出力CSVファイル
- `--smiles-col STR`: SMILES列名（デフォルト: 'smiles'）
- `--type STR`: 特徴量タイプ（rdkit/xtb/uma、デフォルト: rdkit）
- `--verbose`: 詳細ログ

**例**:
```bash
python cli.py extract --input data.csv --output features.csv --type rdkit
```

---

#### `python cli.py train [OPTIONS]`
モデル学習

**引数**:
- `--data PATH`: 学習データCSV
- `--target STR`: ターゲット列名
- `--model STR`: モデルタイプ（xgboost/lightgbm/rf、デフォルト: xgboost）
- `--output PATH`: モデル保存先
- `--cv INT`: 交差検証fold数（デフォルト: 5）

**例**:
```bash
python cli.py train --data features.csv --target logS --model xgboost --output model.pkl
```

---

#### `python cli.py predict [OPTIONS]`
予測実行

**引数**:
- `--model PATH`: モデルファイル
- `--input PATH`: 入力CSV
- `--output PATH`: 出力CSV
- `--smiles-col STR`: SMILES列名

**例**:
```bash
python cli.py predict --model model.pkl --input test.csv --output predictions.csv
```

---

## REST API

### ヘルスチェック

#### `GET /api/health`
システムヘルスチェック

**レスポンス**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-21T10:00:00Z"
}
```

---

#### `GET /api/health/rdkit`
RDKit動作確認

**レスポンス**:
```json
{
  "rdkit_available": true,
  "rdkit_version": "2023.09.1"
}
```

---

### 分子API

#### `POST /api/molecules/validate`
SMILES検証

**リクエスト**:
```json
{
  "smiles": "CCO"
}
```

**レスポンス**:
```json
{
  "valid": true,
  "canonical_smiles": "CCO",
  "errors": []
}
```

**エラー**:
```json
{
  "valid": false,
  "canonical_smiles": null,
  "errors": ["Invalid SMILES syntax"]
}
```

---

#### `GET /api/molecules/{smiles}/properties`
分子物性取得

**パスパラメータ**:
- `smiles`: SMILES文字列（URLエンコード必要）

**レスポンス**:
```json
{
  "smiles": "CCO",
  "properties": {
    "molecular_weight": 46.07,
    "logP": -0.07,
    "tpsa": 20.23,
    "num_h_donors": 1,
    "num_h_acceptors": 1,
    "num_rotatable_bonds": 0,
    "num_aromatic_rings": 0
  }
}
```

---

#### `GET /api/molecules/{smiles}/svg`
分子構造SVG取得

**パスパラメータ**:
- `smiles`: SMILES文字列

**クエリパラメータ**:
- `width` (optional): 幅（デフォルト: 300）
- `height` (optional): 高さ（デフォルト: 300）

**レスポンス**: SVG画像（image/svg+xml）

---

### データセットAPI

#### `GET /api/datasets`
データセット一覧取得

**クエリパラメータ**:
- `page` (optional): ページ番号（デフォルト: 1）
- `per_page` (optional): 1ページあたり件数（デフォルト: 20）

**レスポンス**:
```json
{
  "data": [
    {
      "id": 1,
      "name": "Solubility Dataset",
      "row_count": 1000,
      "uploaded_at": "2026-01-20T10:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 5,
    "pages": 1
  }
}
```

---

#### `POST /api/datasets`
データセットアップロード

**リクエスト**: multipart/form-data
- `name`: データセット名
- `file`: CSVファイル

**レスポンス**:
```json
{
  "id": 1,
  "name": "My Dataset",
  "row_count": 500,
  "uploaded_at": "2026-01-21T10:00:00Z"
}
```

---

#### `DELETE /api/datasets/{id}`
データセット削除

**パスパラメータ**:
- `id`: データセットID

**レスポンス**: 204 No Content

---

### 実験API

#### `GET /api/experiments`
実験一覧取得

**クエリパラメータ**:
- `status` (optional): ステータスフィルタ（pending/running/completed/failed）
- `model_type` (optional): モデルタイプフィルタ
- `page`, `per_page`: ページネーション

**レスポンス**:
```json
{
  "data": [
    {
      "id": 1,
      "name": "XGBoost Solubility",
      "status": "completed",
      "model_type": "xgboost",
      "feature_type": "rdkit",
      "created_at": "2026-01-20T10:00:00Z"
    }
  ]
}
```

---

#### `POST /api/experiments`
実験作成・開始

**リクエスト**:
```json
{
  "dataset_id": 1,
  "name": "My Experiment",
  "target_column": "logS",
  "feature_type": "rdkit",
  "model_type": "xgboost",
  "config": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
  }
}
```

**レスポンス**:
```json
{
  "id": 1,
  "name": "My Experiment",
  "status": "pending",
  "created_at": "2026-01-21T10:00:00Z"
}
```

---

#### `GET /api/experiments/{id}`
実験詳細取得

**パスパラメータ**:
- `id`: 実験ID

**レスポンス**:
```json
{
  "id": 1,
  "name": "My Experiment",
  "status": "completed",
  "dataset": {
    "id": 1,
    "name": "Solubility Dataset"
  },
  "target_column": "logS",
  "feature_type": "rdkit",
  "model_type": "xgboost",
  "config": {...},
  "result": {
    "metrics": {
      "r2": 0.85,
      "mae": 0.23,
      "rmse": 0.31
    },
    "completed_at": "2026-01-21T10:05:00Z"
  }
}
```

---

#### `DELETE /api/experiments/{id}`
実験削除

**パスパラメータ**:
- `id`: 実験ID

**レスポンス**: 204 No Content

---

#### `POST /api/experiments/{id}/predict`
単一予測

**パスパラメータ**:
- `id`: 実験ID

**リクエスト**:
```json
{
  "smiles": "CCO"
}
```

**レスポンス**:
```json
{
  "smiles": "CCO",
  "prediction": 1.23,
  "uncertainty": 0.15
}
```

---

#### `POST /api/experiments/{id}/batch_predict`
バッチ予測

**パスパラメータ**:
- `id`: 実験ID

**リクエスト**:
```json
{
  "smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]
}
```

**レスポンス**:
```json
{
  "predictions": [
    {"smiles": "CCO", "prediction": 1.23, "uncertainty": 0.15},
    {"smiles": "c1ccccc1", "prediction": 2.45, "uncertainty": 0.18},
    {"smiles": "CC(=O)O", "prediction": 0.87, "uncertainty": 0.12}
  ]
}
```

---

## 特徴量抽出API

### BaseFeatureExtractor

全特徴量抽出器の基底クラス

#### メソッド

##### `__init__(**kwargs)`
初期化

**引数**:
- `**kwargs`: 抽出器固有の設定

---

##### `fit(smiles_list, y=None)`
抽出器をデータにフィット（Statefulの場合）

**引数**:
- `smiles_list` (List[str]): SMILESリスト
- `y` (Optional[Any]): ターゲット変数

**戻り値**: `Self`

---

##### `transform(smiles_list)`
SMILESを特徴量に変換

**引数**:
- `smiles_list` (List[str]): SMILESリスト

**戻り値**: `pd.DataFrame` - 特徴量DataFrame

---

##### `fit_transform(smiles_list, y=None)`
fit + transform を一度に実行

**引数**:
- `smiles_list` (List[str]): SMILESリスト
- `y` (Optional[Any]): ターゲット変数

**戻り値**: `pd.DataFrame`

---

##### `save(path)`
抽出器の状態を保存

**引数**:
- `path` (str): 保存先パス

---

##### `load(path)`
抽出器の状態を読み込み

**引数**:
- `path` (str): 読み込み元パス

**戻り値**: `Self`

---

#### プロパティ

##### `is_fitted`
フィット済みか

**戻り値**: `bool`

---

##### `descriptor_names`
記述子名のリスト

**戻り値**: `List[str]`

---

##### `n_descriptors`
記述子の数

**戻り値**: `int`

---

### RDKitFeatureExtractor

RDKit分子記述子抽出

#### 初期化
```python
from core.services.features import RDKitFeatureExtractor

extractor = RDKitFeatureExtractor(
    descriptor_types=['basic', 'topological', 'electronic'],
    use_3d=False
)
```

**引数**:
- `descriptor_types` (List[str]): 記述子タイプ
  - `'basic'`: 基本記述子（MW, LogP, TPSA等）
  - `'topological'`: トポロジカル記述子
  - `'electronic'`: 電子的記述子
  - `'fingerprint'`: フィンガープリント
- `use_3d` (bool): 3D記述子を使用（デフォルト: False）

#### 使用例
```python
extractor = RDKitFeatureExtractor()
features = extractor.transform(['CCO', 'c1ccccc1'])
print(features.shape)  # (2, N)
print(extractor.descriptor_names)  # ['MolWt', 'LogP', ...]
```

---

### XTBFeatureExtractor

XTB量子化学記述子抽出

#### 初期化
```python
from core.services.features import XTBFeatureExtractor

extractor = XTBFeatureExtractor(
    method='GFN2-xTB',
    optimize=True,
    charge=0,
    multiplicity=1
)
```

**引数**:
- `method` (str): 計算手法（'GFN1-xTB', 'GFN2-xTB'、デフォルト: 'GFN2-xTB'）
- `optimize` (bool): 構造最適化を実行（デフォルト: True）
- `charge` (int): 電荷（デフォルト: 0）
- `multiplicity` (int): 多重度（デフォルト: 1）

#### 計算される記述子
- `total_energy`: 全エネルギー（Hartree）
- `homo`: HOMO軌道エネルギー（eV）
- `lumo`: LUMO軌道エネルギー（eV）
- `gap`: HOMO-LUMOギャップ（eV）
- `dipole`: 双極子モーメント（Debye）
- `polarizability`: 分極率

---

### UMAFeatureExtractor

UMAP次元削減による特徴量抽出

#### 初期化
```python
from core.services.features import UMAFeatureExtractor

extractor = UMAFeatureExtractor(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    base_features='rdkit'
)
```

**引数**:
- `n_components` (int): 削減後の次元数（デフォルト: 10）
- `n_neighbors` (int): 近傍数（デフォルト: 15）
- `min_dist` (float): 最小距離（デフォルト: 0.1）
- `metric` (str): 距離メトリック（'euclidean', 'manhattan'等、デフォルト: 'euclidean'）
- `base_features` (str): 元特徴量タイプ（'rdkit', 'fingerprint'、デフォルト: 'rdkit'）

#### 使用例（要fit）
```python
extractor = UMAFeatureExtractor(n_components=5)
extractor.fit(smiles_train, y_train)  # 学習が必要
features = extractor.transform(smiles_test)
```

---

### TarteFeatureExtractor

Transformer（TARTE）による特徴量抽出

#### 初期化
```python
from core.services.features import TarteFeatureExtractor

extractor = TarteFeatureExtractor(
    mode='featurizer',
    model_name='default',
    n_features=128
)
```

**引数**:
- `mode` (str): 動作モード
  - `'featurizer'`: 特徴量抽出のみ
  - `'finetuning'`: Finetuning後の特徴量
  - `'boosting'`: Boosting統合
- `model_name` (str): モデル名（デフォルト: 'default'）
- `n_features` (int): 特徴量次元（デフォルト: 128）

**注意**: `tarte-ai`パッケージが必要

---

### SmartFeatureEngine

物性別最適化特徴量エンジン

#### 初期化
```python
from core.services.features import SmartFeatureEngine

engine = SmartFeatureEngine(
    target_property='glass_transition',
    auto_select=True,
    selection_method='boruta',
    n_features=50
)
```

**引数**:
- `target_property` (str): 物性タイプ
  - 光学: `'refractive_index'`, `'absorption'`, `'fluorescence'`
  - 機械: `'glass_transition'`, `'tensile_strength'`, `'hardness'`
  - 熱: `'melting_point'`, `'thermal_conductivity'`, `'heat_capacity'`
  - 電気: `'conductivity'`, `'dielectric_constant'`
  - 化学: `'solubility'`, `'reactivity'`, `'stability'`
  - 薬理: `'bioavailability'`, `'toxicity'`, `'binding_affinity'`
- `auto_select` (bool): 自動特徴量選択（デフォルト: True）
- `selection_method` (str): 選択手法（'boruta', 'mrmr', 'rfe'、デフォルト: 'boruta'）
- `n_features` (int): 選択する特徴量数（デフォルト: 50）

#### 使用例
```python
engine = SmartFeatureEngine(target_property='solubility')
result = engine.fit_transform(smiles_list, y_solubility)
print(result.keys())  # ['features', 'selected_indices', 'importances']
```

---

## 機械学習API

### BaseMLModel

全MLモデルの基底クラス

#### メソッド

##### `__init__(**kwargs)`
初期化

**引数**:
- `**kwargs`: モデル固有のハイパーパラメータ

---

##### `fit(X, y, **kwargs)`
モデル学習

**引数**:
- `X` (pd.DataFrame | np.ndarray): 特徴量（N x M）
- `y` (pd.Series | np.ndarray): ターゲット変数（N,）
- `**kwargs`: 追加パラメータ（eval_set, early_stopping等）

**戻り値**: `Self`

---

##### `predict(X)`
予測実行

**引数**:
- `X` (pd.DataFrame | np.ndarray): 特徴量（N x M）

**戻り値**: `np.ndarray` - 予測値（N,）

---

##### `predict_proba(X)`
クラス確率予測（分類のみ）

**引数**:
- `X` (pd.DataFrame | np.ndarray): 特徴量

**戻り値**: `Optional[np.ndarray]` - クラス確率（N x C）、回帰の場合None

---

##### `save(path)`
モデル保存

**引数**:
- `path` (str | Path): 保存先パス

---

##### `load(path)`
モデル読み込み

**引数**:
- `path` (str | Path): 読み込み元パス

**戻り値**: `Self`

---

##### `get_params()`
ハイパーパラメータ取得

**戻り値**: `Dict[str, Any]`

---

##### `set_params(**params)`
ハイパーパラメータ設定

**引数**:
- `**params`: 更新するパラメータ

**戻り値**: `Self`

---

### MLPipeline

機械学習パイプライン

#### 初期化
```python
from core.services.ml.pipeline import MLPipeline
from core.services.features import RDKitFeatureExtractor

pipeline = MLPipeline(
    feature_extractor=RDKitFeatureExtractor(),
    model_type='xgboost',
    model_params={'n_estimators': 100, 'max_depth': 6},
    use_uncertainty=True,
    cv_folds=5
)
```

**引数**:
- `feature_extractor` (BaseFeatureExtractor): 特徴量抽出器
- `model_type` (str): モデルタイプ（'xgboost', 'lightgbm', 'randomforest'）
- `model_params` (Dict): モデルパラメータ
- `use_uncertainty` (bool): 不確実性定量化を使用（デフォルト: False）
- `cv_folds` (int): 交差検証fold数（デフォルト: 5）

#### メソッド

##### `fit(smiles_list, y, validation_split=0.2)`
パイプライン学習

**引数**:
- `smiles_list` (List[str]): SMILES リスト
- `y` (np.ndarray): ターゲット変数
- `validation_split` (float): 検証データ比率（デフォルト: 0.2）

**戻り値**: `Self`

---

##### `predict(smiles_list, return_uncertainty=False)`
予測実行

**引数**:
- `smiles_list` (List[str]): SMILESリスト
- `return_uncertainty` (bool): 不確実性も返すか（デフォルト: False）

**戻り値**: 
- `return_uncertainty=False`: `np.ndarray` - 予測値
- `return_uncertainty=True`: `Tuple[np.ndarray, np.ndarray]` - (予測値, 不確実性)

---

##### `evaluate(smiles_test, y_test)`
モデル評価

**引数**:
- `smiles_test` (List[str]): テストSMILES
- `y_test` (np.ndarray): テストターゲット

**戻り値**: `Dict[str, float]` - メトリクス（r2, mae, rmse等）

---

##### `save(path)`
パイプライン保存

**引数**:
- `path` (str): 保存先パス

---

##### `load(path)`
パイプライン読み込み

**引数**:
- `path` (str): 読み込み元パス

**戻り値**: `MLPipeline`

---

### AutoMLOptimizer

Optunaによる自動ハイパーパラメータ最適化

#### 初期化
```python
from core.services.ml.automl import AutoMLOptimizer

optimizer = AutoMLOptimizer(
    model_type='xgboost',
    n_trials=100,
    cv_folds=5,
    direction='maximize',
    metric='r2'
)
```

**引数**:
- `model_type` (str): モデルタイプ
- `n_trials` (int): 試行回数（デフォルト: 100）
- `cv_folds` (int): 交差検証fold数（デフォルト: 5）
- `direction` (str): 最適化方向（'maximize', 'minimize'、デフォルト: 'maximize'）
- `metric` (str): 最適化メトリック（'r2', 'mae', 'rmse'、デフォルト: 'r2'）

#### メソッド

##### `optimize(X, y, timeout=None)`
最適化実行

**引数**:
- `X` (pd.DataFrame): 特徴量
- `y` (np.ndarray): ターゲット
- `timeout` (Optional[int]): タイムアウト（秒）

**戻り値**: `Dict[str, Any]` - 最適パラメータ

---

##### `get_best_params()`
最適パラメータ取得

**戻り値**: `Dict[str, Any]`

---

##### `get_study_summary()`
最適化履歴サマリー

**戻り値**: `pd.DataFrame`

---

## 可視化API

### BaseVisualizer

全可視化の基底クラス

#### メソッド

##### `__init__(**kwargs)`
初期化

**引数**:
- `**kwargs`: 可視化固有の設定

---

##### `plot(*args, **kwargs)`
プロット生成

**引数**:
- `*args`: プロット用データ
- `**kwargs`: プロット設定

**戻り値**: 図オブジェクト（matplotlib.Figure, plotly.Figure等）

---

##### `save(fig, path, format=None, **kwargs)`
図を保存

**引数**:
- `fig`: 図オブジェクト
- `path` (str | Path): 保存先パス
- `format` (Optional[str]): 出力形式（'png', 'svg', 'html', 'json'、自動判定可）
- `**kwargs`: 保存オプション（dpi, width, height等）

---

##### `to_base64(fig, format='png')`
Base64エンコード文字列を生成

**引数**:
- `fig`: 図オブジェクト
- `format` (str): 出力形式（デフォルト: 'png'）

**戻り値**: `str` - Base64エンコード文字列

---

### SHAPVisualizer

SHAP説明可視化

#### 初期化
```python
from core.services.vis.shap_eng import SHAPVisualizer

viz = SHAPVisualizer(
    plot_type='waterfall',
    max_display=20
)
```

**引数**:
- `plot_type` (str): プロットタイプ
  - `'waterfall'`: ウォーターフォール図
  - `'summary'`: サマリープロット
  - `'dependence'`: 依存性プロット
  - `'force'`: フォースプロット
- `max_display` (int): 最大表示特徴量数（デフォルト: 20）

#### メソッド

##### `plot(model, X, feature_names=None, sample_index=None)`
SHAP図生成

**引数**:
- `model`: 学習済みモデル
- `X` (pd.DataFrame): 特徴量
- `feature_names` (Optional[List[str]]): 特徴量名
- `sample_index` (Optional[int]): サンプルインデックス（waterfall/force用）

**戻り値**: matplotlib.Figure

---

### PDPVisualizer

Partial Dependence Plot可視化

#### 初期化
```python
from core.services.vis.pdp_eng import PDPVisualizer

viz = PDPVisualizer(
    feature_names=['MolWt', 'LogP'],
    kind='average'
)
```

**引数**:
- `feature_names` (List[str]): 表示する特徴量名
- `kind` (str): プロットタイプ（'average', 'individual', 'both'、デフォルト: 'average'）

#### メソッド

##### `plot(model, X, feature_idx, grid_resolution=100)`
PDP生成

**引数**:
- `model`: 学習済みモデル
- `X` (pd.DataFrame): 特徴量
- `feature_idx` (int | str): 特徴量インデックスまたは名前
- `grid_resolution` (int): グリッド解像度（デフォルト: 100）

**戻り値**: matplotlib.Figure

---

### ChemSpaceVisualizer

化学空間マップ可視化

#### 初期化
```python
from core.services.vis.chem_space import ChemSpaceVisualizer

viz = ChemSpaceVisualizer(
    method='umap',
    n_components=2,
    color_by='target'
)
```

**引数**:
- `method` (str): 次元削減手法（'umap', 'tsne', 'pca'、デフォルト: 'umap'）
- `n_components` (int): 削減後次元（2 or 3、デフォルト: 2）
- `color_by` (str): 色分け基準（'target', 'cluster', 'none'）

#### メソッド

##### `plot(features, y=None, smiles=None)`
化学空間プロット生成

**引数**:
- `features` (pd.DataFrame): 特徴量
- `y` (Optional[np.ndarray]): ターゲット変数（色分け用）
- `smiles` (Optional[List[str]]): SMILES（ツールチップ用）

**戻り値**: plotly.Figure（インタラクティブ）

---

### MoleculeVisualizer

分子構造可視化

#### 使用例
```python
from core.services.vis.mol_viewer import MoleculeVisualizer

viz = MoleculeVisualizer()
fig = viz.plot('CCO', size=(300, 300), highlight_atoms=[0, 1])
viz.save(fig, 'ethanol.png)
```

**plot()引数**:
- `smiles` (str): SMILES
- `size` (Tuple[int, int]): 画像サイズ（デフォルト: (300, 300)）
- `highlight_atoms` (Optional[List[int]]): ハイライトする原子インデックス

---

## プラグインAPI

### Plugin

プラグイン定義クラス

#### 初期化
```python
from core.services.plugin import Plugin

plugin = Plugin(
    name="my_plugin",
    version="1.0.0",
    description="カスタムプラグイン",
    hooks={
        "on_prediction": my_prediction_hook,
        "on_training": my_training_hook
    },
    author="Your Name",
    license="MIT",
    requires=["rdkit>=2023.09"],
    config={"threshold": 0.8}
)
```

**引数**:
- `name` (str): プラグイン名
- `version` (str): バージョン
- `description` (str): 説明
- `hooks` (Dict[str, Callable]): フック関数
- `enabled` (bool): 有効/無効（デフォルト: True）
- `author` (Optional[str]): 作者
- `license` (Optional[str]): ライセンス
- `requires` (List[str]): 依存パッケージ
- `config` (Dict[str, Any]): 設定

---

### PluginManager

プラグイン管理

#### 初期化
```python
from core.services.plugin import PluginManager

pm = PluginManager(
    auto_discover=True,
    plugin_dir='plugins'
)
```

**引数**:
- `auto_discover` (bool): 自動検出を有効化（デフォルト: False）
- `plugin_dir` (str): プラグインディレクトリ（デフォルト: 'plugins'）

#### メソッド

##### `register(plugin)`
プラグイン登録

**引数**:
- `plugin` (Plugin): プラグインインスタンス

---

##### `unregister(name)`
プラグイン解除

**引数**:
- `name` (str): プラグイン名

**戻り値**: `bool` - 成功時True

---

##### `execute_hook(hook_name, *args, **kwargs)`
フック実行

**引数**:
- `hook_name` (str): フック名
- `*args`, `**kwargs`: フック関数に渡す引数

**戻り値**: `List[Any]` - 各フック関数の戻り値リスト

---

##### `discover_plugins(plugin_dir=None)`
プラグイン自動検出

**引数**:
- `plugin_dir` (Optional[str]): 検出ディレクトリ

**戻り値**: `List[Plugin]` - 検出されたプラグイン

---

##### `list_plugins()`
プラグイン一覧

**戻り値**: `List[Dict[str, Any]]` - プラグイン情報リスト

---

##### `enable(name)` / `disable(name)`
プラグイン有効化/無効化

**引数**:
- `name` (str): プラグイン名

**戻り値**: `bool` - 成功時True

---

### 利用可能なフック

#### `on_prediction`
予測実行後に呼ばれる

**シグネチャ**:
```python
def on_prediction(smiles: str, prediction: float, **kwargs) -> float:
    # 処理
    return adjusted_prediction
```

---

#### `on_training`
学習完了後に呼ばれる

**シグネチャ**:
```python
def on_training(experiment, **kwargs) -> None:
    # 処理（例: 通知送信）
    pass
```

---

#### `on_feature_extraction`
特徴量抽出前に呼ばれる

**シグネチャ**:
```python
def on_feature_extraction(smiles_list: List[str]) -> List[str]:
    # 前処理
    return processed_smiles_list
```

---

#### `on_error`
エラー発生時に呼ばれる

**シグネチャ**:
```python
def on_error(error: Exception, context: dict) -> None:
    # エラー処理（例: ロギング、通知）
    pass
```

---

## 設定

### Django設定（chem_ml_project/settings.py）

#### DATABASE
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

#### INSTALLED_APPS
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    # ...
    'core',
    'huey.contrib.djhuey',
]
```

#### REST_FRAMEWORK
```python
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20
}
```

---

### MLflow設定

#### トラッキングURI
```python
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('chemical_ml')
```

#### 使用例
```python
with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({'r2': 0.85, 'mae': 0.23})
    mlflow.sklearn.log_model(model, "model")
```

---

### Huey設定（タスクキュー）

```python
from huey import SqliteHuey

huey = SqliteHuey(filename='huey_db.sqlite3')

@huey.task()
def train_model_async(experiment_id):
    # 非同期学習処理
    pass
```

---

## エラーハンドリング

### 標準エラーコード

| コード | 説明 |
|--------|------|
| `ERR_1001` | バリデーションエラー |
| `ERR_1002` | 必須フィールド欠落 |
| `ERR_2001` | リソース未発見 |
| `ERR_2002` | リソース重複 |
| `ERR_3001` | 無効なSMILES |
| `ERR_3002` | 分子サイズ超過 |
| `ERR_4001` | モデル未学習 |
| `ERR_4002` | 予測失敗 |

### エラーレスポンス例
```json
{
  "error": {
    "code": "ERR_3001",
    "message": "Invalid SMILES syntax",
    "details": {
      "smiles": "INVALID",
      "position": 3
    },
    "timestamp": "2026-01-21T10:00:00Z"
  }
}
```

---

## パフォーマンス最適化

### キャッシング

#### RDKit分子キャッシュ
```python
from core.services.cache import MoleculeCache

cache = MoleculeCache(maxsize=1000)
mol = cache.get_or_create('CCO')
```

#### 特徴量キャッシュ
```python
from core.services.feature_store import FeatureStore

store = FeatureStore()
store.save_features('dataset_1', features_df)
features = store.load_features('dataset_1')
```

---

### バッチ処理

```python
from core.services.utils.batch_processing import batch_process

results = batch_process(
    smiles_list,
    process_func=extract_features,
    batch_size=100,
    n_jobs=4
)
```

---

## セキュリティ

### SMILES検証
```python
from core.services.validation import validate_smiles

is_valid, error = validate_smiles('CCO')
if not is_valid:
    raise ValueError(error)
```

### ファイルアップロード検証
```python
from core.services.validation import validate_csv_file

is_valid, error = validate_csv_file(uploaded_file)
```

---

## 付録

### 用語集

- **SMILES**: 分子構造を文字列で表現する記法
- **記述子**: 分子の特徴を数値化したもの
- **HOMO/LUMO**: 最高被占軌道/最低空軌道
- **UMAP**: 次元削減手法
- **SHAP**: SHapley Additive exPlanationsの略、説明可能AI手法

---

### 参考リンク

- **GitHub**: https://github.com/jckkvs/chem_app
- **RDKit Documentation**: https://www.rdkit.org/docs/
- **XTB Documentation**: https://xtb-docs.readthedocs.io/
- **MLflow Documentation**: https://mlflow.org/docs/latest/

---

**このドキュメントは生成AIでの利用を想定しています。**
**全API、全メソッド、全引数を網羅した完全リファレンスです。**
