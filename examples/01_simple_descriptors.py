"""
最小限のサンプル - 分子記述子計算

このスクリプトは、SMILESから分子記述子を計算する最も簡単な例です。
"""

from core.services.features import RDKitFeatureExtractor

# サンプルSMILES
smiles_list = [
    'CCO',           # エタノール
    'c1ccccc1',      # ベンゼン
    'CC(=O)O',       # 酢酸
]

# 特徴量抽出器を作成
extractor = RDKitFeatureExtractor()

# 特徴量を計算
features = extractor.transform(smiles_list)

# 結果を表示
print(f"計算された特徴量の形状: {features.shape}")
print(f"特徴量の数: {extractor.n_descriptors}")
print(f"\n最初の5つの特徴量名:")
print(extractor.descriptor_names[:5])

print(f"\n結果（最初の3列のみ表示）:")
print(features.iloc[:, :3])
