"""
可視化の例 - SHAP説明と化学空間マップ

このスクリプトは、予測結果の可視化例です。
"""

import numpy as np

from core.services.features import RDKitFeatureExtractor
from core.services.ml.pipeline import MLPipeline
from core.services.vis.chem_space import ChemSpaceVisualizer
from core.services.vis.shap_eng import SHAPVisualizer

# サンプルデータ
smiles_list = [
    'CCO', 'CCCO', 'CCCCO', 'CCCCCO',  # アルコール
    'c1ccccc1', 'c1ccc(cc1)C', 'c1ccc(cc1)CC',  # 芳香族
    'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O',  # カルボン酸
]
y = np.array([-0.77, -0.34, 0.14, 0.58, -2.13, -2.73, -3.15, -0.17, 0.33, 0.79])

# モデル学習
print("モデル学習中...")
extractor = RDKitFeatureExtractor()
pipeline = MLPipeline(
    feature_extractor=extractor,
    model_type='xgboost'
)
pipeline.fit(smiles_list, y)

# 特徴量取得
features = extractor.transform(smiles_list)

# 1. SHAP説明
print("\n1. SHAP説明生成中...")
try:
    shap_viz = SHAPVisualizer(plot_type='summary')
    fig = shap_viz.plot(pipeline.model, features, feature_names=extractor.descriptor_names)
    shap_viz.save(fig, 'shap_summary.png')
    print("   → shap_summary.png に保存")
except Exception as e:
    print(f"   SHAP生成エラー: {e}")

# 2. 化学空間マップ
print("\n2. 化学空間マップ生成中...")
try:
    chem_viz = ChemSpaceVisualizer(method='umap', color_by='target')
    fig = chem_viz.plot(features, y=y, smiles=smiles_list)
    chem_viz.save(fig, 'chemical_space.html')
    print("   → chemical_space.html に保存（ブラウザで開いてください）")
except Exception as e:
    print(f"   化学空間マップ生成エラー: {e}")

print("\n完了！")
