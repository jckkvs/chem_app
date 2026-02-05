# Chemical ML Platform インストールマニュアル

**最終更新**: 2026-02-05  
**対象**: Chemical ML Platform 全機能（80+MLライブラリ、30+EDA機能）

---

## 📋 前提条件

- Python 3.9以上
- pip または conda
- Git
- (オプション) PostgreSQL（本番環境）

---

## 🚀 インストール手順

### 方法1: pip（推奨）

#### ステップ1: リポジトリクローン

```bash
git clone https://github.com/jckkvs/chem_app.git
cd chem_ml_app
```

#### ステップ2: 仮想環境作成

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### ステップ3: 基本依存関係インストール

```bash
# 最小構成（コア機能のみ）
pip install rdkit django ninja streamlit pandas scikit-learn xgboost lightgbm shap mlflow optuna

# 完全構成（80+ライブラリ、推奨）
pip install -r requirements.txt
```

#### ステップ4: データベースマイグレーション

```bash
python manage.py migrate
```

#### ステップ5: サーバー起動

```bash
# Django
python manage.py runserver

# Streamlit（別ターミナル）
cd frontend_streamlit
streamlit run app.py
```

---

### 方法2: conda

```bash
# 環境作成
conda create -n chem_ml python=3.9
conda activate chem_ml

# RDKit（condaで推奨）
conda install -c conda-forge rdkit

# 他の依存関係
pip install -r requirements.txt

# XTB（量子計算、オプション）
conda install -c conda-forge xtb
```

---

## 📦 依存関係詳細

### コア依存関係（必須）

| ライブラリ | バージョン | 用途 |
|----------|----------|------|
| Django | >=4.2 | Webフレームワーク |
| django-ninja | >=1.0 | REST API |
| Streamlit | >=1.28 | UI |
| RDKit | >=2023.3 | 化学計算 |
| pandas | >=2.0 | データ操作 |
| numpy | >=1.24 | 数値計算 |
| scikit-learn | >=1.3 | 機械学習 |
| matplotlib | >=3.7 | 可視化 |
| seaborn | >=0.12 | 統計可視化 |

### Boosting（推奨）

```bash
pip install xgboost>=2.0 lightgbm>=4.1 catboost>=1.2 ngboost>=0.4
```

### AutoML（オプション）

```bash
pip install tabpfn>=0.1 tpot>=0.12 \
  auto-sklearn>=0.15 pycaret>=3.0 autogluon>=0.8
```

### ハイパーパラメータ最適化（推奨）

```bash
pip install optuna>=3.4 hyperopt>=0.2 \
  sklearn-genetic-opt>=0.10 ray[tune]>=2.7
```

### Interpretable ML（オプション）

```bash
pip install imodels>=1.3 interpret>=0.5 pygam>=0.8 wittgenstein>=0.3
```

### XAI（推奨）

```bash
pip install shap>=0.43 lime>=0.2
```

### 時系列分析（オプション）

```bash
pip install prophet>=1.1 statsmodels>=0.14 tsfresh>=0.20
```

### 次元削減・クラスタリング（推奨）

```bash
pip install umap-learn>=0.5 scikit-learn>=1.3
```

### 因果推論（オプション）

```bash
pip install econml>=0.14 causalml>=0.15 dowhy>=0.10
```

### 深層学習埋め込み（オプション）

```bash
# Uni-Mol（3D分子）
pip install unimol-tools

# ChemBERTa（SMILES Transformer）
pip install transformers torch

# SchNet（等変GNN）
pip install schnetpack

# TARTE（表形式Transformer）
pip install tarte-ai
```

### 最適化（オプション）

```bash
pip install deap>=1.4 GPy>=1.10 GPyOpt>=1.2
```

### 非同期処理（推奨）

```bash
pip install huey>=2.5
```

### データベース（本番環境）

```bash
# PostgreSQL
pip install psycopg2-binary>=2.9

# または
conda install -c conda-forge postgresql
```

---

## 🔧 オプショナル機能インストール

### XTB（量子化学計算）

```bash
# conda（推奨）
conda install -c conda-forge xtb

# または手動インストール
# https://github.com/grimme-lab/xtb
```

**使い方**:
```python
from core.services.features.xtb_eng import XTBEngine

engine = XTBEngine()
result = engine.compute_properties(['CCO'])
```

### TARTE（表形式Transformer）

```bash
pip install tarte-ai
```

**使い方**:
1. Streamlitサイドバー → 「🤖 TARTE Settings」
2. 「TARTE を使用」をチェック
3. モード選択（Featurizer / Finetuning / Boosting）

### Uni-Mol（3D分子埋め込み）

```bash
pip install unimol-tools
```

**使い方**:
```python
from core.services.features.pretrained_embeddings import PretrainedEmbeddingEngine

engine = PretrainedEmbeddingEngine(model_type='unimol')
result = engine.compute_embeddings(['CCO'])
```

---

## 🗄️ データベース設定

### 開発環境（SQLite、デフォルト）

```python
# settings.py（デフォルト）
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### 本番環境（PostgreSQL）

```bash
# PostgreSQLインストール
sudo apt install postgresql postgresql-contrib  # Ubuntu
brew install postgresql  # macOS

# データベース作成
createdb chem_ml

# .envファイル作成
cat > .env << EOF
DATABASE_URL=postgresql://user:password@localhost:5432/chem_ml
EOF
```

```python
# settings.py
import dj_database_url

DATABASES = {
    'default': dj_database_url.config(
        default='postgresql://user:password@localhost:5432/chem_ml'
    )
}
```

---

## 🌐 プロキシ設定（企業環境）

### 環境変数設定

```bash
# Windows
set HTTP_PROXY=http://proxy.example.com:8080
set HTTPS_PROXY=http://proxy.example.com:8080

# macOS/Linux
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### アプリ内設定

1. **Django Web UI**: http://localhost:8000/proxy-settings
2. **Streamlit UI**: サイドバー → 「🌐 Proxy Settings」タブ

→ 詳細: [PROXY_SETUP_GUIDE.md](PROXY_SETUP_GUIDE.md)

---

## ✅ インストール確認

### 基本動作確認

```bash
# Django
python manage.py runserver
# ブラウザで http://localhost:8000/api/health にアクセス
# {"status": "ok", "timestamp": "..."} が表示されればOK

# RDKit確認
curl http://localhost:8000/api/health/rdkit
# {"rdkit": true, "version": "..."} が表示されればOK
```

### Streamlit確認

```bash
cd frontend_streamlit
streamlit run app.py
# ブラウザで http://localhost:8501 にアクセス
```

### サンプルプログラム実行

```bash
# クイックEDA
python examples/comprehensive_samples.py 1

# SMILES予測
python examples/comprehensive_samples.py 2

# 全8種類のサンプルを確認
python examples/comprehensive_samples.py
```

---

## 🧪 テスト実行

```bash
# 基本テスト
pytest core/tests/ -v

# カバレッジ付き
pytest core/tests/ -v --cov=core --cov-report=html

# 特定テスト
pytest core/tests/test_features.py -v
```

---

## 🔍 トラブルシューティング

### RDKitインポートエラー

```bash
# conda環境で再インストール
conda install -c conda-forge rdkit
```

### XTBが見つからない

```bash
# condaでインストール（推奨）
conda install -c conda-forge xtb

# パス確認
which xtb  # macOS/Linux
where xtb  # Windows
```

### TARTEインポートエラー

```bash
# バージョン確認
pip show tarte-ai

# 再インストール
pip uninstall tarte-ai
pip install tarte-ai
```

### Streamlitポートエラー

```bash
# 別ポート指定
streamlit run app.py --server.port 8502
```

### Djangoマイグレーションエラー

```bash
# データベースリセット
rm db.sqlite3
python manage.py migrate
```

---

## 📚 次のステップ

1. **サンプル実行**: `python examples/comprehensive_samples.py`
2. **ドキュメント確認**: [README.md](README.md)
3. **API試用**: http://localhost:8000/api/docs
4. **再現プロンプト**: [REPRODUCE_PROMPT.md](REPRODUCE_PROMPT.md)

---

## 🔗 参考リンク

- **GitHub**: https://github.com/jckkvs/chem_app
- **RDKit**: https://www.rdkit.org/docs/
- **Django**: https://docs.djangoproject.com/
- **Streamlit**: https://docs.streamlit.io/

---

**インストール完了後は `python manage.py runserver` でサーバーを起動してください！**
