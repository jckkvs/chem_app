"""
04_deep_learning.py - 深層学習モデル埋め込み

難易度: ⭐⭐⭐⭐ 上級
所要時間: 30分
必要パッケージ: torch, transformers, (schnetpack, torchdrug - オプション)

この例で学べること:
- 事前学習済みモデル (Uni-Mol, ChemBERTa) の使い方
- 自己教師あり学習モデル (GROVER, MolCLR) の使い方
- 等変GNN (SchNet, PaiNN) の使い方
- 複数モデルの埋め込み結合
"""

# =============================================================================
# Part 1: 事前学習済みモデル (ChemBERTa)
# =============================================================================

print("=" * 60)
print("Part 1: ChemBERTa埋め込み")
print("=" * 60)

from core.services.features import PretrainedEmbeddingEngine

smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'c1cccnc1']

engine = PretrainedEmbeddingEngine()

# 利用可能なモデル確認
print("利用可能なモデル:")
for model, available in engine.list_available_models().items():
    status = "✓" if available else "✗"
    print(f"  {status} {model}")

# ChemBERTa埋め込み（インストールされている場合）
if engine.is_model_available('chemberta'):
    df = engine.get_embeddings_df(smiles_list, 'chemberta')
    print(f"\nChemBERTa埋め込み: {df.shape}")
else:
    print("\nChemBERTaは未インストール。以下でインストール:")
    print("  pip install transformers torch")


# =============================================================================
# Part 2: 自己教師あり学習モデル (MolCLR)
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: MolCLR自己教師あり学習埋め込み")
print("=" * 60)

from core.services.features.ssl_embeddings import (
    SelfSupervisedEmbeddingEngine,
    get_ssl_embeddings,
)

ssl_engine = SelfSupervisedEmbeddingEngine()

# 利用可能なSSLモデル
print("自己教師あり学習モデル:")
for model, available in ssl_engine.list_available_models().items():
    status = "✓" if available else "✗"
    print(f"  {status} {model}")

# MolCLR埋め込み（PyTorchがあれば動作）
if ssl_engine.is_model_available('molclr'):
    df = get_ssl_embeddings(smiles_list, model='molclr')
    print(f"\nMolCLR埋め込み: {df.shape}")
    print(f"カラム例: {list(df.columns[:5])}")


# =============================================================================
# Part 3: 等変GNN (SchNet/PaiNN)
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: 等変GNN（3D構造考慮）")
print("=" * 60)

from core.services.features.equivariant_gnn import (
    EquivariantEmbeddingEngine,
    MolecularStructure,
)

# 3D構造生成のデモ
struct = MolecularStructure.from_smiles('CCO')
if struct:
    print(f"エタノール 3D構造:")
    print(f"  原子数: {struct.n_atoms}")
    print(f"  原子番号: {struct.atomic_numbers}")
    print(f"  座標形状: {struct.positions.shape}")

# 等変GNNエンジン
egnn_engine = EquivariantEmbeddingEngine()

print("\n等変GNNモデル:")
for model, available in egnn_engine.list_available_models().items():
    status = "✓" if available else "✗"
    print(f"  {status} {model}")

if egnn_engine.is_model_available('schnet'):
    df = egnn_engine.get_embeddings_df(smiles_list, 'schnet')
    print(f"\nSchNet埋め込み: {df.shape}")
else:
    print("\nSchNetは未インストール。以下でインストール:")
    print("  pip install schnetpack")


# =============================================================================
# Part 4: 複数モデルの埋め込み結合
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: 複数モデルの結合")
print("=" * 60)

import pandas as pd

# 利用可能なモデルの埋め込みを結合
embeddings = []

# SSL埋め込み
if ssl_engine.is_model_available('molclr'):
    embeddings.append(ssl_engine.get_embeddings_df(smiles_list, 'molclr'))

if ssl_engine.is_model_available('graphmvp'):
    embeddings.append(ssl_engine.get_embeddings_df(smiles_list, 'graphmvp'))

if embeddings:
    combined = pd.concat(embeddings, axis=1)
    print(f"結合後の埋め込み: {combined.shape}")
    print(f"含まれるモデル: MolCLR, GraphMVP")


# =============================================================================
# Part 5: SmartFeatureEngineと組み合わせる
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: SmartFeatureEngineとの統合")
print("=" * 60)

from core.services.features import SmartFeatureEngine

# 事前学習モデルを指定して特徴量生成
# （モデルがインストールされていれば使用、なければスキップ）
engine = SmartFeatureEngine(
    target_property='solubility',
    use_pretrained=['chemberta', 'unimol'],  # 利用可能なものだけ使用
)

result = engine.fit_transform(smiles_list)

print(f"生成された特徴量: {result.n_features}")
print(f"使用した事前学習モデル: {result.pretrained_models}")
