# 🧪 Smart Feature Engineering for chem_ml_app

**物性×データセット特性に基づくインテリジェントな分子特徴量エンジニアリング**

---

## 📚 ドキュメント構成

| レベル | 対象者 | ファイル |
|--------|-------|---------|
| **入門** | 初めての人 | [examples/01_quickstart.py](examples/01_quickstart.py) |
| **基本** | 基本機能を使う | [examples/02_basic_features.py](examples/02_basic_features.py) |
| **中級** | カスタマイズしたい | [examples/03_advanced_selection.py](examples/03_advanced_selection.py) |
| **上級** | 深層学習/SSL | [examples/04_deep_learning.py](examples/04_deep_learning.py) |
| **分析** | データ分析 | [examples/05_analysis_tools.py](examples/05_analysis_tools.py) |

---

## 🚀 クイックスタート（5分で始める）

```python
from core.services.features import SmartFeatureEngine

# SMILESリストと目的物性を指定するだけ
smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
engine = SmartFeatureEngine(target_property='solubility')
result = engine.fit_transform(smiles)

print(result.features)  # 最適化された特徴量
```

---

## 📦 モジュール一覧

### 🎯 コア機能
| モジュール | 機能 | 難易度 |
|-----------|------|-------|
| `SmartFeatureEngine` | 統合特徴量エンジン | ⭐ 入門 |
| `list_presets()` | 19物性プリセット確認 | ⭐ 入門 |

### 🔬 個別特徴量
| モジュール | 機能 | 難易度 |
|-----------|------|-------|
| `RDKitFeatureExtractor` | 分子記述子 | ⭐⭐ 基本 |
| `TarteFeatureExtractor` | 表データ埋め込み | ⭐⭐ 基本 |

### 🤖 深層学習
| モジュール | 機能 | 難易度 |
|-----------|------|-------|
| `PretrainedEmbeddingEngine` | Uni-Mol/ChemBERTa | ⭐⭐⭐ 中級 |
| `SelfSupervisedEmbeddingEngine` | GROVER/MolCLR | ⭐⭐⭐⭐ 上級 |
| `EquivariantEmbeddingEngine` | SchNet/PaiNN | ⭐⭐⭐⭐ 上級 |

### 📊 分析ツール
| モジュール | 機能 | 難易度 |
|-----------|------|-------|
| `DatasetAnalyzer` | データセット分析 | ⭐⭐ 基本 |
| `ScaffoldAnalyzer` | 骨格分析 | ⭐⭐ 基本 |
| `ApplicabilityDomainAnalyzer` | 予測信頼性 | ⭐⭐⭐ 中級 |
| `MolecularSimilaritySearch` | 類似度検索 | ⭐⭐ 基本 |

### 🗂️ 特徴量選択
| モジュール | 機能 | 難易度 |
|-----------|------|-------|
| `MRMRSelector` | mRMR選択 | ⭐⭐⭐ 中級 |
| `BorutaSelector` | Boruta選択 | ⭐⭐⭐ 中級 |
| `EnsembleFeatureSelector` | アンサンブル | ⭐⭐⭐ 中級 |

---

## 🎨 19物性プリセット

```python
from core.services.features import list_presets
print(list_presets())
```

| カテゴリ | プリセット |
|---------|----------|
| 光学 | `refractive_index`, `optical_gap` |
| 機械 | `elastic_modulus`, `tensile_strength`, `hardness` |
| 熱 | `glass_transition`, `melting_point`, `thermal_conductivity` |
| 電気 | `dielectric_constant`, `conductivity` |
| 化学 | `solubility`, `viscosity`, `density` |
| 薬理 | `admet`, `pka` |
| 汎用 | `general` |

---

## 📥 インストール

```bash
# 必須
pip install rdkit pandas numpy scikit-learn

# オプション（深層学習）
pip install torch transformers  # ChemBERTa
pip install unimol-tools        # Uni-Mol
pip install tarte-ai            # TARTE
pip install schnetpack          # SchNet/PaiNN
pip install torchdrug           # GROVER
pip install selfies             # SELFIES
```

---

## 📄 ライセンス

MIT License
