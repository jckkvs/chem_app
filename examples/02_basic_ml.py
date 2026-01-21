"""
基本的な機械学習 - 溶解度予測

このスクリプトは、分子の溶解度を予測する基本的な機械学習の例です。
"""

import numpy as np
from core.services.features import RDKitFeatureExtractor
from core.services.ml.pipeline import MLPipeline

# サンプルデータ（SMILES と 溶解度 logS）
smiles_train = [
    'CCO',           # エタノール
    'c1ccccc1',      # ベンゼン
    'CC(=O)O',       # 酢酸
    'CCN(CC)CC',     # トリエチルアミン
    'c1ccc(cc1)O',   # フェノール
]

# 溶解度データ（logS）
y_train = np.array([-0.77, -2.13, -0.17, -1.18, -0.04])

# テストデータ
smiles_test = ['CCCO', 'c1ccccc1C']  # プロパノール、トルエン
y_test = np.array([-0.34, -2.73])

# パイプライン作成
print("モデル学習中...")
pipeline = MLPipeline(
    feature_extractor=RDKitFeatureExtractor(),
    model_type='xgboost',
    model_params={'n_estimators': 50, 'max_depth': 3}
)

# 学習
pipeline.fit(smiles_train, y_train)
print("学習完了！")

# 予測
predictions = pipeline.predict(smiles_test)

# 結果表示
print("\n予測結果:")
for smiles, pred, actual in zip(smiles_test, predictions, y_test):
    print(f"{smiles:15s} 予測: {pred:6.2f}  実測: {actual:6.2f}  誤差: {abs(pred-actual):5.2f}")

# モデル評価
metrics = pipeline.evaluate(smiles_test, y_test)
print(f"\nR²スコア: {metrics['r2']:.3f}")
print(f"平均絶対誤差: {metrics['mae']:.3f}")
