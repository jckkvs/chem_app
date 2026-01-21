"""
LLMアシスタント使用例

このスクリプトは、軽量LLM（GPT4All）を使った対話型アシスタントの例です。
"""

from core.services.llm import ChemMLAssistant

# アシスタントを初期化
print("ChemML Assistant を初期化中...")
assistant = ChemMLAssistant(verbose=True)

print(f"GPT4All利用可能: {assistant.is_available}")
print()

# 1. 特徴量選択のアドバイス
print("=" * 60)
print("1. 特徴量選択のアドバイス")
print("=" * 60)

dataset_info = {"n_samples": 500, "task_type": "regression"}

suggestion = assistant.suggest_features(
    dataset_info=dataset_info, target_property="solubility (logS)"
)

print(f"\n推奨特徴量:")
for feat in suggestion.recommended_features:
    print(f"  - {feat}")

print(f"\n理由:")
print(f"  {suggestion.reasoning}")

print()

# 2. 解析プランの提案
print("=" * 60)
print("2. 解析プランの提案")
print("=" * 60)

problem = (
    "Create a regression model to predict aqueous solubility (logS) "
    "from molecular structures"
)

dataset_info = {
    "n_samples": 1200,
    "task_type": "regression",
    "target_property": "logS",
}

plan = assistant.suggest_analysis_plan(
    problem_description=problem, dataset_info=dataset_info
)

print(f"\n目的: {plan.objective}")
print(f"\n推奨アプローチ:")
print(f"  {plan.recommended_approach}")

print(f"\nモデル候補:")
for model in plan.model_suggestions:
    print(f"  - {model}")

print()

# 3. 結果の解釈
print("=" * 60)
print("3. モデル結果の解釈")
print("=" * 60)

metrics = {"r2": 0.85, "mae": 0.42, "rmse": 0.58}

interpretation = assistant.interpret_results(metrics, model_type="XGBoost")

print(f"\nメトリクス: {metrics}")
print(f"\n解釈:")
print(f"  {interpretation}")

print()

# 4. 自由形式の質問
print("=" * 60)
print("4. 自由形式の質問")
print("=" * 60)

question = "What are the best practices for handling imbalanced datasets in cheminformatics?"

answer = assistant.ask(question)

print(f"\n質問: {question}")
print(f"\n回答:")
print(f"  {answer}")

print("\n" + "=" * 60)
print("完了！")
print("=" * 60)

print(
    "\nNote: GPT4Allがインストールされていない場合、"
    "ルールベースの簡易応答が返されます。"
)
print("完全な機能を使用するには: pip install gpt4all")
