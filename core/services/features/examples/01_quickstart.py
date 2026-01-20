"""
01_quickstart.py - 5分で始めるクイックスタート

難易度: ⭐ 入門
所要時間: 5分
必要パッケージ: rdkit, pandas, numpy

この例で学べること:
- SmartFeatureEngineの基本的な使い方
- 物性プリセットの確認方法
- 結果の見方
"""

# =============================================================================
# Step 1: インポート
# =============================================================================

from core.services.features import (
    SmartFeatureEngine,
    list_presets,
)

# =============================================================================
# Step 2: 物性プリセットを確認
# =============================================================================

print("利用可能な物性プリセット:")
presets = list_presets()
for name, name_ja in presets.items():
    print(f"  {name}: {name_ja}")

# =============================================================================
# Step 3: SMILESデータを用意
# =============================================================================

# 例: 有機溶媒のSMILES
smiles_list = [
    'CCO',           # エタノール
    'CC(=O)C',       # アセトン
    'c1ccccc1',      # ベンゼン
    'CCN',           # エチルアミン
    'CC(=O)O',       # 酢酸
]

# =============================================================================
# Step 4: SmartFeatureEngineで特徴量生成
# =============================================================================

# 目的物性を指定するだけ！
engine = SmartFeatureEngine(target_property='solubility')

# fit_transformで特徴量生成
result = engine.fit_transform(smiles_list)

# =============================================================================
# Step 5: 結果を確認
# =============================================================================

print("\n=== 生成された特徴量 ===")
print(f"サンプル数: {result.n_samples}")
print(f"特徴量数: {result.n_features}")
print(f"使用した記述子: {result.rdkit_descriptors[:5]}...")  # 最初の5個

print("\n特徴量データ:")
print(result.features.head())

print("\n=== サマリー ===")
print(result.summary())

# =============================================================================
# Step 6: 他の物性でも試す
# =============================================================================

print("\n\n=== ガラス転移温度(Tg)用の特徴量 ===")
engine_tg = SmartFeatureEngine(target_property='glass_transition')
result_tg = engine_tg.fit_transform(smiles_list)
print(result_tg.summary())
