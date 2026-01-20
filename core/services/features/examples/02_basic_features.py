"""
02_basic_features.py - 基本的な特徴量抽出

難易度: ⭐⭐ 基本
所要時間: 15分
必要パッケージ: rdkit, pandas, numpy

この例で学べること:
- 個別の特徴量抽出器の使い方
- DatasetAnalyzerによるデータセット分析
- 物性プリセットの詳細確認
"""

# =============================================================================
# Part 1: RDKit記述子
# =============================================================================

print("=" * 60)
print("Part 1: RDKit記述子抽出")
print("=" * 60)

from core.services.features import RDKitFeatureExtractor

smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCC']

# 基本的な使い方
extractor = RDKitFeatureExtractor()
features = extractor.transform(smiles_list)

print(f"抽出された記述子数: {features.shape[1]}")
print(f"カラム例: {list(features.columns[:10])}")

# カテゴリを指定して抽出
extractor_specific = RDKitFeatureExtractor(
    categories=['lipophilicity', 'structural']
)
features_specific = extractor_specific.transform(smiles_list)
print(f"\n特定カテゴリのみ: {features_specific.shape[1]}特徴量")


# =============================================================================
# Part 2: データセット分析
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: データセット分析")
print("=" * 60)

from core.services.features import DatasetAnalyzer, analyze_dataset

# 便利関数を使う方法
profile = analyze_dataset(smiles_list)

print(f"有効分子数: {profile.n_valid}/{profile.n_total}")
print(f"平均分子量: {profile.mw_mean:.1f}")
print(f"平均回転可能結合数: {profile.rotatable_bonds_mean:.1f}")
print(f"推奨プリセット: {profile.recommended_preset}")

print("\n分析ノート:")
for note in profile.analysis_notes:
    print(f"  - {note}")


# =============================================================================
# Part 3: 物性プリセットの詳細
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: 物性プリセットの詳細")
print("=" * 60)

from core.services.features import get_preset

# ガラス転移温度プリセットの詳細
preset = get_preset('glass_transition')

print(f"プリセット名: {preset.name_ja}")
print(f"カテゴリ: {preset.category.value}")
print(f"\n選定根拠:\n{preset.rationale}")
print(f"\n使用する記述子 ({len(preset.rdkit_descriptors)}個):")
for desc in preset.rdkit_descriptors[:10]:
    print(f"  - {desc}")

print(f"\n推奨事前学習モデル: {preset.recommended_pretrained}")
print(f"文献参照: {preset.literature_refs}")


# =============================================================================
# Part 4: 骨格分析
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: 骨格分析")
print("=" * 60)

from core.services.features import analyze_scaffolds

# 骨格の多様性を分析
scaffold_profile = analyze_scaffolds(smiles_list)

print(f"ユニーク骨格数: {scaffold_profile.n_unique_scaffolds}")
print(f"骨格多様性: {scaffold_profile.scaffold_diversity:.2%}")

print("\n骨格カテゴリ分布:")
for cat, count in scaffold_profile.scaffold_categories.items():
    print(f"  {cat}: {count}")


# =============================================================================
# Part 5: 類似度検索
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: 類似度検索")
print("=" * 60)

from core.services.features.similarity_search import search_similar

# データベース（検索対象）
database = ['CCO', 'CCCO', 'c1ccccc1', 'c1ccc(O)cc1', 'CCN', 'CCCN']

# クエリ分子に類似した分子を検索
result = search_similar('c1ccc(Cl)cc1', database, k=3)

print(f"クエリ: {result.query_smiles}")
print("\n類似分子:")
for smi, sim in zip(result.similar_smiles, result.similarities):
    print(f"  {smi}: {sim:.3f}")
