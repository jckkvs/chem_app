# Chemical ML Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2%2B-green.svg)](https://www.djangoproject.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/jckkvs/chem_app/actions/workflows/test.yml/badge.svg)](https://github.com/jckkvs/chem_app/actions/workflows/test.yml)
[![Lint](https://github.com/jckkvs/chem_app/actions/workflows/lint.yml/badge.svg)](https://github.com/jckkvs/chem_app/actions/workflows/lint.yml)

機械学習を使った分子物性予測プラットフォーム

## 📚 ドキュメント

- **[完全リファレンス](REFERENCE.md)** - 全API、全メソッド、全引数を網羅（生成AI時代向け）
- **[サンプルコード](examples/)** - 初心者向け実行可能な例
- **[開発者ガイド](CONTRIBUTING.md)** - プロジェクトへの貢献方法
- **[アーキテクチャ](ARCHITECTURE.md)** - システム設計と拡張ポイント
- **[プラグイン開発](docs/PLUGIN_DEVELOPMENT.md)** - プラグインの作り方
- **[API設計](docs/API_GUIDELINES.md)** - REST API設計標準
- **[データスキーマ](docs/DATA_SCHEMA.md)** - データベース構造

---

## 特徴

- 🧪 **4タスクタイプ対応**: SMILES / 表データ / 混合物 / ハイブリッド
- 📊 **マルチフロントエンド**: Django / Streamlit / PWA（スマホ対応）
- 🔬 **計算化学機能**: 構造最適化、一点計算、HOMO/LUMO
- 🤖 **AutoML**: Optuna自動チューニング
- 📈 **信頼区間付き予測**: Quantile/Bootstrap
- 🗺️ **可視化**: SHAP、PDP、化学空間マップ
- 📱 **PWA対応**: オフラインキャッシュ、ホーム画面追加可能
- 🧠 **Smart Feature Engineering**: 物性別最適化（NEW!）

## Smart Feature Engineering（NEW!）

物性×データセット特性に基づくインテリジェントな特徴量生成。

```python
from core.services.features import SmartFeatureEngine

engine = SmartFeatureEngine(target_property='glass_transition')
result = engine.fit_transform(smiles_list)
```

**ハイライト:**
- 19物性プリセット（光学/機械/熱/電気/化学/薬理）
- 深層学習埋め込み（Uni-Mol, ChemBERTa, MolCLR）
- 自動特徴量選択（Boruta, mRMR）
- Applicability Domain分析

→ 詳細: [core/services/features/README.md](core/services/features/README.md)

## インストール

```bash
pip install rdkit django ninja streamlit pandas scikit-learn xgboost lightgbm shap mlflow optuna

# XTB（量子計算用、オプション）
conda install -c conda-forge xtb

# TARTE（Transformer表形式特徴量、オプション）
pip install tarte-ai
```

### オプショナル機能

| 機能 | インストール | 説明 |
|------|-------------|------|
| **XTB** | `conda install -c conda-forge xtb` | 量子化学記述子 |
| **TARTE** | `pip install tarte-ai` | 表形式Transformer |
| **Uni-Mol** | `pip install unimol-tools` | 3D分子埋め込み |
| **ChemBERTa** | `pip install transformers torch` | SMILES Transformer |
| **SchNet** | `pip install schnetpack` | 等変GNN |
| **SELFIES** | `pip install selfies` | 堅牢なSMILES代替 |

TARTE使用時はStreamlitサイドバーの「🤖 TARTE Settings」から有効化できます。


## クイックスタート

```bash
cd chem_ml_app
python manage.py runserver
```

**3つのアクセス方法：**
- **Django Web**: http://localhost:8000 （推奨）
- **Streamlit**: `cd frontend_streamlit && streamlit run app.py` → http://localhost:8501
- **PWA**: スマホブラウザから http://localhost:8000 → ホームに追加

## API エンドポイント一覧

| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/api/health` | ヘルスチェック |
| GET | `/api/health/rdkit` | RDKit動作確認 |
| POST | `/api/molecules/validate` | SMILES検証 |
| GET | `/api/molecules/{smiles}/properties` | 分子物性取得 |
| GET | `/api/molecules/{smiles}/svg` | 分子SVG画像 |
| GET | `/api/datasets` | データセット一覧 |
| POST | `/api/datasets` | データセットアップロード |
| DELETE | `/api/datasets/{id}` | データセット削除 |
| GET | `/api/experiments` | 実験一覧 |
| POST | `/api/experiments` | 実験作成・開始 |
| GET | `/api/experiments/{id}` | 実験詳細 |
| DELETE | `/api/experiments/{id}` | 実験削除 |
| POST | `/api/experiments/{id}/predict` | 単一予測 |
| POST | `/api/experiments/{id}/batch_predict` | バッチ予測 |

## 使用例

### 簡単なサンプル（初心者向け）
```bash
# サンプルを実行
python examples/01_simple_descriptors.py   # 分子記述子計算
python examples/02_basic_ml.py             # 機械学習
python examples/03_api_usage.py            # REST API
python examples/04_visualization.py        # 可視化
```

**詳細**: [examples/README.md](examples/README.md)

---

### API経由で分子物性取得
```bash
curl http://localhost:8000/api/molecules/CCO/properties
```

### バッチ予測（Python）
```python
import requests

response = requests.post(
    "http://localhost:8000/api/experiments/1/batch_predict",
    json={"smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]}
)
print(response.json())
```

## モジュール構成

```
core/
├── api.py               # REST API（16エンドポイント）
├── views.py             # Djangoビュー + PWA
├── templates/           # Djangoテンプレート（5ページ）
└── services/
    ├── features/        # 特徴量抽出（26モジュール）
    ├── ml/              # 機械学習（32モジュール）
    └── vis/             # 可視化（10モジュール）
```

## テスト実行

```bash
# 全テスト実行
python -m pytest core/tests/ -v

# カバレッジ付き
pytest core/tests/ -v --cov=core --cov-report=html

# CI/CD（自動実行）
# GitHub Actions が自動テスト・Lintを実行
```

---

## 開発

### 開発用依存関係

```bash
pip install -r requirements-dev.txt
```

### コード品質チェック

```bash
# フォーマット
black core/
isort core/

# Lint
flake8 core/

# 型チェック
mypy core/
```

---

## 貢献

プロジェクトへの貢献を歓迎します！

1. [CONTRIBUTING.md](CONTRIBUTING.md)を読む
2. Issueで提案・バグ報告
3. Pull Requestを作成

詳細な開発ガイドは[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。

---

## ライセンス

MIT

---

## 🔗 リンク

- **GitHub**: https://github.com/jckkvs/chem_app
- **完全リファレンス**: [REFERENCE.md](REFERENCE.md)
- **ドキュメント一覧**: [上記参照](#📚-ドキュメント)
