"""
REST API使用例

このスクリプトは、REST APIを使用してプログラムから操作する例です。
"""

import requests

# APIのベースURL
BASE_URL = "http://localhost:8000/api"

# 1. ヘルスチェック
print("1. ヘルスチェック...")
response = requests.get(f"{BASE_URL}/health")
print(f"   ステータス: {response.json()['status']}")

# 2. 分子物性取得
print("\n2. 分子物性取得...")
smiles = "CCO"  # エタノール
response = requests.get(f"{BASE_URL}/molecules/{smiles}/properties")
properties = response.json()['properties']
print(f"   SMILES: {smiles}")
print(f"   分子量: {properties['molecular_weight']:.2f}")
print(f"   LogP: {properties['logP']:.2f}")
print(f"   TPSA: {properties['tpsa']:.2f}")

# 3. SMILES検証
print("\n3. SMILES検証...")
test_smiles = ["CCO", "INVALID"]
for s in test_smiles:
    response = requests.post(
        f"{BASE_URL}/molecules/validate",
        json={"smiles": s}
    )
    result = response.json()
    status = "✓ 有効" if result['valid'] else "✗ 無効"
    print(f"   {s:10s} → {status}")

# 4. バッチ予測（実験ID=1が存在する場合）
print("\n4. バッチ予測の例...")
print("   （実験を作成後、experiment_idを指定してください）")
# response = requests.post(
#     f"{BASE_URL}/experiments/1/batch_predict",
#     json={"smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]}
# )
# predictions = response.json()['predictions']
# for pred in predictions:
#     print(f"   {pred['smiles']:10s} → {pred['prediction']:.2f}")

print("\n完了！")
