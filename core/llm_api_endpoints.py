"""
LLM Assistant API Endpoints - Append to core/api.py
"""

# --- LLM Assistant Endpoints ---


class LLMFeatureSuggestionIn(Schema):
    """特徴量選択アドバイスリクエスト"""

    n_samples: int
    task_type: str = "regression"  # or "classification"
    target_property: str
    available_features: Optional[List[str]] = None


class LLMFeatureSuggestionOut(Schema):
    """特徴量選択アドバイスレスポンス"""

    recommended_features: List[str]
    reasoning: str
    alternative_features: List[str]
    considerations: List[str]


class LLMAnalysisPlanIn(Schema):
    """解析プラン提案リクエスト"""

    problem_description: str
    n_samples: int
    task_type: str = "regression"
    target_property: str


class LLMAnalysisPlanOut(Schema):
    """解析プラン提案レスポンス"""

    objective: str
    recommended_approach: str
    feature_engineering_steps: List[str]
    model_suggestions: List[str]
    validation_strategy: str
    potential_challenges: List[str]


class LLMInterpretResultsIn(Schema):
    """結果解釈リクエスト"""

    metrics: Dict[str, float]
    model_type: str = "unknown"


class LLMInterpretResultsOut(Schema):
    """結果解釈レスポンス"""

    interpretation: str


class LLMAskIn(Schema):
    """自由形式Q&Aリクエスト"""

    question: str
    context: Optional[str] = None


class LLMAskOut(Schema):
    """自由形式Q&Aレスポンス"""

    question: str
    answer: str
    llm_available: bool


@api.post("/llm/suggest-features", response={200: LLMFeatureSuggestionOut, 400: ErrorOut})
def llm_suggest_features(request, payload: LLMFeatureSuggestionIn):
    """
    LLMアシスタント: 特徴量選択のアドバイス

    データセット情報と予測対象の物性から、適切な特徴量を推奨します。
    """
    try:
        from core.services.llm import ChemMLAssistant

        assistant = ChemMLAssistant()

        dataset_info = {
            "n_samples": payload.n_samples,
            "task_type": payload.task_type,
            "target_property": payload.target_property,
        }

        suggestion = assistant.suggest_features(
            dataset_info=dataset_info,
            target_property=payload.target_property,
            available_features=payload.available_features,
        )

        return {
            "recommended_features": suggestion.recommended_features,
            "reasoning": suggestion.reasoning,
            "alternative_features": suggestion.alternative_features,
            "considerations": suggestion.considerations,
        }

    except Exception as e:
        logger.error(f"LLM feature suggestion failed: {e}")
        return 400, {"detail": str(e)}


@api.post("/llm/suggest-plan", response={200: LLMAnalysisPlanOut, 400: ErrorOut})
def llm_suggest_analysis_plan(request, payload: LLMAnalysisPlanIn):
    """
    LLMアシスタント: 解析プランの提案

    問題記述とデータセット情報から、適切な解析戦略を提案します。
    """
    try:
        from core.services.llm import ChemMLAssistant

        assistant = ChemMLAssistant()

        dataset_info = {
            "n_samples": payload.n_samples,
            "task_type": payload.task_type,
            "target_property": payload.target_property,
        }

        plan = assistant.suggest_analysis_plan(
            problem_description=payload.problem_description, dataset_info=dataset_info
        )

        return {
            "objective": plan.objective,
            "recommended_approach": plan.recommended_approach,
            "feature_engineering_steps": plan.feature_engineering_steps,
            "model_suggestions": plan.model_suggestions,
            "validation_strategy": plan.validation_strategy,
            "potential_challenges": plan.potential_challenges,
        }

    except Exception as e:
        logger.error(f"LLM analysis plan failed: {e}")
        return 400, {"detail": str(e)}


@api.post("/llm/interpret-results", response={200: LLMInterpretResultsOut, 400: ErrorOut})
def llm_interpret_results(request, payload: LLMInterpretResultsIn):
    """
    LLMアシスタント: モデル結果の解釈

    評価指標から、結果の解釈と改善案を提案します。
    """
    try:
        from core.services.llm import ChemMLAssistant

        assistant = ChemMLAssistant()

        interpretation = assistant.interpret_results(
            metrics=payload.metrics, model_type=payload.model_type
        )

        return {"interpretation": interpretation}

    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        return 400, {"detail": str(e)}


@api.post("/llm/ask", response={200: LLMAskOut, 400: ErrorOut})
def llm_ask(request, payload: LLMAskIn):
    """
    LLMアシスタント: 自由形式Q&A

    化学機械学習に関する質問に回答します。
    """
    try:
        from core.services.llm import ChemMLAssistant

        assistant = ChemMLAssistant()

        answer = assistant.ask(question=payload.question, context=payload.context)

        return {
            "question": payload.question,
            "answer": answer,
            "llm_available": assistant.is_available,
        }

    except Exception as e:
        logger.error(f"LLM Q&A failed: {e}")
        return 400, {"detail": str(e)}
