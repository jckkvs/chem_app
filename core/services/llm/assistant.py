"""
ChemML Assistant - 軽量LLMによる対話型解析アシスタント

Implements: F-LLM-001
設計思想:
- CPU動作可能な軽量LLM（GPT4All）
- 特徴量選択・解析方針のアドバイスに特化
- プログラミングではなく、ドメイン知識の提供

参考文献:
- GPT4All: https://github.com/nomic-ai/gpt4all
- Phi-3-mini: Microsoft lightweight LLM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# GPT4All利用可能フラグ（遅延チェック）
_GPT4ALL_AVAILABLE: Optional[bool] = None


def _check_gpt4all_available() -> bool:
    """GPT4Allがインストールされているかチェック"""
    global _GPT4ALL_AVAILABLE
    if _GPT4ALL_AVAILABLE is None:
        try:
            import gpt4all

            _GPT4ALL_AVAILABLE = True
            logger.info(
                f"gpt4all v{getattr(gpt4all, '__version__', 'unknown')} が利用可能"
            )
        except ImportError:
            _GPT4ALL_AVAILABLE = False
            logger.debug("gpt4allがインストールされていません")
    return _GPT4ALL_AVAILABLE


@dataclass
class FeatureSuggestion:
    """特徴量選択の提案"""

    recommended_features: List[str]
    reasoning: str
    alternative_features: List[str]
    considerations: List[str]


@dataclass
class AnalysisPlan:
    """解析プランの提案"""

    objective: str
    recommended_approach: str
    feature_engineering_steps: List[str]
    model_suggestions: List[str]
    validation_strategy: str
    potential_challenges: List[str]


class ChemMLAssistant:
    """
    化学ML解析アシスタント（軽量LLM搭載）

    Features:
    - 特徴量選択のアドバイス
    - 解析方針の提案
    - 結果の解釈・要約
    - CPU動作可能（GPT4All利用）

    Example:
        >>> assistant = ChemMLAssistant()
        >>> suggestion = assistant.suggest_features(
        ...     dataset_info={'n_samples': 1000, 'task': 'regression'},
        ...     target_property='solubility'
        ... )

    Note:
        GPT4Allがインストールされていない場合はルールベースで動作。
        インストール: pip install gpt4all
    """

    # デフォルトモデル（軽量・CPU最適化）
    DEFAULT_MODEL = "Phi-3-mini-4k-instruct.gguf"  # 3.8B params, 2.2GB

    # プロンプトテンプレート
    FEATURE_SELECTION_PROMPT_TEMPLATE = """You are an expert in chemical machine learning and cheminformatics.

Dataset Information:
- Number of samples: {n_samples}
- Task type: {task_type}
- Target property: {target_property}
- Available feature groups: {available_features}

Question: Which feature groups should I prioritize for predicting {target_property}? 
Consider the dataset size, task type, and chemical domain knowledge.

Provide a concise response (max 200 words) with:
1. Top 3 recommended feature groups
2. Brief reasoning
3. One alternative approach

Response:"""

    ANALYSIS_PLAN_PROMPT_TEMPLATE = """You are an expert in chemical machine learning.

Problem Description: {problem_description}

Dataset:
- Samples: {n_samples}
- Task: {task_type}
- Target: {target_property}

Question: Suggest a high-level analysis plan for this problem.

Provide a concise response (max 250 words) covering:
1. Recommended modeling approach
2. Feature engineering suggestions
3. Validation strategy
4. Potential challenges

Response:"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_path: Optional[str] = None,
        device: str = "cpu",
        n_threads: int = 4,
        verbose: bool = False,
    ):
        """
        Args:
            model_name: モデル名（GPT4Allからダウンロード）
            model_path: ローカルモデルパス
            device: 実行デバイス（'cpu' or 'gpu'）
            n_threads: CPU推論スレッド数
            verbose: 詳細ログ
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.n_threads = n_threads
        self.verbose = verbose

        self._model = None
        self._gpt4all_available = _check_gpt4all_available()

        if not self._gpt4all_available:
            logger.warning(
                "GPT4Allがインストールされていません。"
                "ルールベースモードで動作します。インストール: pip install gpt4all"
            )

    @property
    def is_available(self) -> bool:
        """GPT4Allが利用可能か"""
        return self._gpt4all_available

    def _load_model(self):
        """モデルを遅延ロード"""
        if self._model is not None:
            return self._model

        if not self._gpt4all_available:
            return None

        try:
            from gpt4all import GPT4All

            if self.verbose:
                logger.info(f"GPT4Allモデルをロード中: {self.model_name}")

            self._model = GPT4All(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                n_threads=self.n_threads,
            )

            if self.verbose:
                logger.info("GPT4Allモデルロード完了")

            return self._model

        except Exception as e:
            logger.error(f"GPT4Allモデルのロードに失敗: {e}")
            return None

    def _generate(self, prompt: str, max_tokens: int = 300) -> str:
        """LLMで文章生成"""
        model = self._load_model()

        if model is None:
            return self._fallback_response(prompt)

        try:
            response = model.generate(
                prompt=prompt, max_tokens=max_tokens, temp=0.7, top_p=0.9
            )
            return response.strip()

        except Exception as e:
            logger.error(f"LLM生成に失敗: {e}")
            return self._fallback_response(prompt)

    def _fallback_response(self, prompt: str) -> str:
        """LLM利用不可時のルールベース応答"""
        if "feature" in prompt.lower():
            return (
                "（ルールベース）一般的に、分子記述子（RDKit）は最も汎用的で、"
                "量子化学記述子（XTB）は物性予測に有効、"
                "フィンガープリント（ECFP）は類似性検索に適しています。"
            )
        elif "analysis" in prompt.lower():
            return (
                "（ルールベース）データサイズに応じてモデルを選択してください。"
                "小規模（<1000）: Random Forest、中規模: XGBoost、大規模: LightGBM推奨。"
            )
        else:
            return "（ルールベース）詳細なアドバイスにはGPT4Allのインストールが必要です。"

    def suggest_features(
        self,
        dataset_info: Dict[str, Any],
        target_property: str,
        available_features: Optional[List[str]] = None,
    ) -> FeatureSuggestion:
        """
        特徴量選択のアドバイス

        Args:
            dataset_info: データセット情報（n_samples, task_type等）
            target_property: 予測対象の物性
            available_features: 利用可能な特徴量グループ

        Returns:
            FeatureSuggestion: 特徴量選択の提案
        """
        if available_features is None:
            available_features = [
                "RDKit descriptors",
                "ECFP fingerprints",
                "XTB quantum descriptors",
                "UMAP embeddings",
                "TARTE transformer features",
            ]

        n_samples = dataset_info.get("n_samples", "unknown")
        task_type = dataset_info.get("task_type", "regression")

        prompt = self.FEATURE_SELECTION_PROMPT_TEMPLATE.format(
            n_samples=n_samples,
            task_type=task_type,
            target_property=target_property,
            available_features=", ".join(available_features),
        )

        response = self._generate(prompt, max_tokens=200)

        # 応答をパース（簡易版）
        lines = [l.strip() for l in response.split("\n") if l.strip()]

        return FeatureSuggestion(
            recommended_features=available_features[
                :3
            ],  # 簡易版: 最初の3つを推奨
            reasoning=response,
            alternative_features=available_features[3:],
            considerations=[
                f"Dataset size: {n_samples}",
                f"Task type: {task_type}",
                f"Target: {target_property}",
            ],
        )

    def suggest_analysis_plan(
        self, problem_description: str, dataset_info: Dict[str, Any]
    ) -> AnalysisPlan:
        """
        解析プランの提案

        Args:
            problem_description: 問題の説明
            dataset_info: データセット情報

        Returns:
            AnalysisPlan: 解析プランの提案
        """
        n_samples = dataset_info.get("n_samples", "unknown")
        task_type = dataset_info.get("task_type", "regression")
        target_property = dataset_info.get("target_property", "unknown")

        prompt = self.ANALYSIS_PLAN_PROMPT_TEMPLATE.format(
            problem_description=problem_description,
            n_samples=n_samples,
            task_type=task_type,
            target_property=target_property,
        )

        response = self._generate(prompt, max_tokens=300)

        return AnalysisPlan(
            objective=problem_description,
            recommended_approach=response,
            feature_engineering_steps=[
                "データクリーニング",
                "特徴量抽出",
                "特徴量選択",
            ],
            model_suggestions=["XGBoost", "LightGBM", "Random Forest"],
            validation_strategy="5-fold cross-validation",
            potential_challenges=["データ不足", "外れ値", "不均衡データ"],
        )

    def interpret_results(
        self, metrics: Dict[str, float], model_type: str = "unknown"
    ) -> str:
        """
        モデル結果の解釈

        Args:
            metrics: 評価指標（r2, mae, rmse等）
            model_type: モデルタイプ

        Returns:
            str: 結果の解釈
        """
        metrics_str = ", ".join([f"{k}={v:.3f}" for k, v in metrics.items()])

        prompt = f"""
You are an expert in machine learning model evaluation.

Model Type: {model_type}
Metrics: {metrics_str}

Question: Interpret these results. Are they good? What could be improved?

Provide a concise response (max 150 words):
"""

        return self._generate(prompt, max_tokens=200)

    def ask(self, question: str, context: Optional[str] = None) -> str:
        """
        自由形式の質問応答

        Args:
            question: 質問
            context: コンテキスト情報

        Returns:
            str: 回答
        """
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        return self._generate(prompt, max_tokens=250)

    def __repr__(self) -> str:
        return (
            f"ChemMLAssistant(model='{self.model_name}', "
            f"available={self._gpt4all_available}, "
            f"device='{self.device}')"
        )
