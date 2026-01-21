"""
LLMアシスタントAPI使用例

このスクリプトは、Django APIを通じてLLMアシスタントを利用する例です。
"""

import requests

BASE_URL = "http://localhost:8000/api"

# 1. 特徴量選択アドバイス
print("=" * 60)
print("1. 特徴量選択アドバイス")
print("=" * 60)

response = requests.post(
    f"{BASE_URL}/llm/suggest-features",
    json={
        "n_samples": 500,
        "task_type": "regression",
        "target_property": "solubility (logS)",
    },
)

if response.status_code == 200:
    result = response.json()
    print(f"\n推奨特徴量:")
    for feat in result["recommended_features"]:
        print(f"  - {feat}")
    print(f"\n理由: {result['reasoning'][:200]}...")
else:
    print(f"エラー: {response.status_code}")

print()

# 2. 解析プラン提案
print("=" * 60)
print("2. 解析プラン提案")
print("=" * 60)

response = requests.post(
    f"{BASE_URL}/llm/suggest-plan",
    json={
        "problem_description": "Predict aqueous solubility from SMILES",
        "n_samples": 1200,
        "task_type": "regression",
        "target_property": "logS",
    },
)

if response.status_code == 200:
    result = response.json()
    print(f"\n推奨アプローチ: {result['recommended_approach'][:200]}...")
    print(f"\nモデル候補:")
    for model in result["model_suggestions"]:
        print(f"  - {model}")
else:
    print(f"エラー: {response.status_code}")

print()

# 3. 結果解釈
print("=" * 60)
print("3. モデル結果解釈")
print("=" * 60)

response = requests.post(
    f"{BASE_URL}/llm/interpret-results",
    json={"metrics": {"r2": 0.85, "mae": 0.42, "rmse": 0.58}, "model_type": "XGBoost"},
)

if response.status_code == 200:
    result = response.json()
    print(f"\n解釈: {result['interpretation'][:200]}...")
else:
    print(f"エラー: {response.status_code}")

print()

# 4. 自由Q&A
print("=" * 60)
print("4. 自由形式Q&A")
print("=" * 60)

response = requests.post(
    f"{BASE_URL}/llm/ask",
    json={
        "question": "What are Morgan fingerprints and when should I use them?",
        "context": "I'm working on molecular similarity search",
    },
)

if response.status_code == 200:
    result = response.json()
    print(f"\n質問: {result['question']}")
    print(f"回答: {result['answer'][:200]}...")
    print(f"LLM利用可能: {result['llm_available']}")
else:
    print(f"エラー: {response.status_code}")

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)
