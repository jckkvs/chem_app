"""
03_advanced_selection.py - 高度な特徴量選択

難易度: ⭐⭐⭐ 中級
所要時間: 20分
必要パッケージ: rdkit, pandas, numpy, scikit-learn

この例で学べること:
- Boruta/mRMRによる特徴量選択
- Applicability Domain (AD) 分析
- カスタマイズしたSmartFeatureEngine
"""

import pandas as pd
import numpy as np

# =============================================================================
# Part 1: 高度な特徴量選択
# =============================================================================

print("=" * 60)
print("Part 1: mRMR特徴量選択")
print("=" * 60)

from core.services.features import (
    RDKitFeatureExtractor,
    MRMRSelector,
    select_features,
)

# サンプルデータ
smiles_list = ['CCO', 'CCCO', 'CCCCO', 'c1ccccc1', 'c1ccc(O)cc1', 
               'CC(=O)O', 'CCC(=O)O', 'CCN', 'CCCN', 'c1cccnc1']
y = pd.Series([1.2, 1.5, 1.8, 2.1, 2.4, 0.8, 1.0, 1.4, 1.6, 2.0])

# 特徴量抽出
extractor = RDKitFeatureExtractor()
X = extractor.transform(smiles_list)
X = X.drop(columns=['SMILES'], errors='ignore')

print(f"元の特徴量数: {X.shape[1]}")

# mRMR選択
X_selected, result = select_features(X, y, method='mrmr', k=10)
print(f"選択後の特徴量数: {X_selected.shape[1]}")
print(f"選択された特徴量: {list(X_selected.columns)}")


# =============================================================================
# Part 2: Boruta選択（シャドウ特徴量ベース）
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Boruta特徴量選択")
print("=" * 60)

from core.services.features import BorutaSelector

# Borutaは統計的検定で重要な特徴量を選択
selector = BorutaSelector(max_iter=20)  # イテレーション数を減らして高速化
selector.fit(X.fillna(0), y)

print(f"選択された特徴量数: {len(selector.selected_features_)}")
print(f"特徴量スコア (top 5):")
sorted_scores = sorted(selector.feature_scores_.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_scores[:5]:
    print(f"  {name}: {score:.2f}")


# =============================================================================
# Part 3: Applicability Domain分析
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Applicability Domain (予測信頼性)")
print("=" * 60)

from core.services.features.applicability_domain import (
    check_applicability_domain,
    ApplicabilityDomainAnalyzer,
)

# トレーニングデータ
train_smiles = ['CCO', 'CCCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'c1ccc(O)cc1']

# テストデータ（一部はトレーニングと似ている、一部は異なる）
test_smiles = ['CCCC', 'c1ccc(Cl)cc1', 'CCCCCCCCCC']  # 長鎖は外れ値かも

# AD分析
result = check_applicability_domain(train_smiles, test_smiles)

print(result.summary())
print("\n各テスト分子の信頼度:")
for smi, conf, in_domain in zip(test_smiles, result.confidence, result.in_domain):
    status = "✓ AD内" if in_domain else "✗ AD外"
    print(f"  {smi}: {conf:.2%} {status}")


# =============================================================================
# Part 4: カスタマイズしたSmartFeatureEngine
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: SmartFeatureEngineのカスタマイズ")
print("=" * 60)

from core.services.features import SmartFeatureEngine, FeatureConfig

# 詳細設定でカスタマイズ
config = FeatureConfig(
    target_property='elastic_modulus',  # 弾性率
    user_additions=['CustomDesc1'],      # カスタム記述子追加
    user_removals=['NumRotatableBonds'], # 特定記述子を除外
    use_morgan_fp=True,                  # Morganフィンガープリントも使用
    remove_high_correlation=True,        # 高相関特徴量を除去
    correlation_threshold=0.90,          # 相関閾値
)

engine = SmartFeatureEngine(config=config)
result = engine.fit_transform(train_smiles)

print(f"カスタマイズ後の特徴量: {result.n_features}")
print(f"Morganフィンガープリント使用: {result.morgan_fp_used}")
