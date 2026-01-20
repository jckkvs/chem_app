"""
05_analysis_tools.py - 分析ツール活用

難易度: ⭐⭐ 基本〜中級
所要時間: 15分

この例で学べること:
- 骨格ベースのデータ分割
- 活性クリフ検出
- 化学空間カバレッジ分析
"""

import numpy as np

# =============================================================================
# Part 1: 骨格ベースのデータ分割
# =============================================================================

print("=" * 60)
print("Part 1: 骨格ベース Train/Test 分割")
print("=" * 60)

from core.services.features import scaffold_split

# 同じ骨格の分子は同じ分割に入る（より厳格な汎化評価）
smiles_list = [
    'c1ccccc1',       # ベンゼン
    'c1ccc(C)cc1',    # トルエン (ベンゼン骨格)
    'c1ccc(O)cc1',    # フェノール (ベンゼン骨格)
    'CCO',            # エタノール
    'CCCO',           # プロパノール
    'c1ccncc1',       # ピリジン
    'c1ccnc(N)c1',    # アミノピリジン (ピリジン骨格)
]

train_idx, test_idx = scaffold_split(smiles_list, test_size=0.3)

print(f"Train indices: {train_idx}")
print(f"Test indices: {test_idx}")

print("\nTrain分子:")
for i in train_idx:
    print(f"  {smiles_list[i]}")

print("\nTest分子:")
for i in test_idx:
    print(f"  {smiles_list[i]}")


# =============================================================================
# Part 2: 活性クリフ検出
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: 活性クリフ検出")
print("=" * 60)

from core.services.features.similarity_search import detect_activity_cliffs

# 構造的に類似だが活性が大きく異なるペアを検出
smiles = [
    'c1ccccc1',           # ベンゼン
    'c1ccc(O)cc1',        # フェノール
    'c1ccc(Cl)cc1',       # クロロベンゼン
    'c1ccc(N)cc1',        # アニリン
    'CCO',                # エタノール
]

# 活性値（例: IC50の対数）
activities = np.array([1.0, 10.0, 2.0, 50.0, 0.5])

cliffs = detect_activity_cliffs(
    smiles, 
    activities, 
    similarity_threshold=0.5  # 類似度50%以上
)

if not cliffs.empty:
    print("検出された活性クリフ:")
    for _, row in cliffs.iterrows():
        print(f"  {row['smiles_1']} vs {row['smiles_2']}")
        print(f"    類似度: {row['similarity']:.2%}")
        print(f"    活性比: {row['activity_ratio']:.1f}x")
else:
    print("活性クリフは検出されませんでした")


# =============================================================================
# Part 3: 化学空間カバレッジ
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: 化学空間カバレッジ分析")
print("=" * 60)

from core.services.features.scaffold_analysis import ChemicalSpaceAnalyzer

# トレーニングとテストの化学空間オーバーラップ
train_smiles = ['CCO', 'CCCO', 'c1ccccc1', 'c1ccc(O)cc1', 'CCN']
test_smiles = ['CCCCO', 'c1ccc(Cl)cc1', 'CCCCCCCCCC']  # 一部は外挿

analyzer = ChemicalSpaceAnalyzer()

# ペアワイズ類似度
mean_sim, std_sim = analyzer.compute_pairwise_similarity(train_smiles)
print(f"トレーニング内平均類似度: {mean_sim:.3f} ± {std_sim:.3f}")

# カバレッジ分析
coverage = analyzer.analyze_coverage(train_smiles, test_smiles)

print(f"\nテスト分子のカバレッジ:")
print(f"  平均最近傍類似度: {coverage['train_test_mean_nn_similarity']:.2%}")
print(f"  50%以上類似: {coverage['test_coverage_50']:.0%}")
print(f"  70%以上類似: {coverage['test_coverage_70']:.0%}")
print(f"  外れ値 (<30%): {coverage['test_outliers']:.0%}")


# =============================================================================
# Part 4: 最近傍分析
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: 最近傍分析")
print("=" * 60)

from core.services.features.similarity_search import MolecularSimilaritySearch

search = MolecularSimilaritySearch()
search.index(smiles)

nn_df = search.find_nearest_neighbors(smiles, k=1)

print("各分子の最近傍:")
for _, row in nn_df.iterrows():
    print(f"  {row['smiles']} → {row['nearest_neighbor_smiles']} "
          f"({row['nearest_neighbor_similarity']:.2%})")
