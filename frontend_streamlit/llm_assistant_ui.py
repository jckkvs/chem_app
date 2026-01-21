"""
LLM Assistant UI - Append to frontend_streamlit/app.py
"""


def render_llm_assistant():
    """LLMアシスタントページ"""
    st.header("🤖 LLM Assistant")
    st.markdown("*軽量LLMによる対話型解析アドバイス*")

    # LLM利用可能性確認
    st.info(
        "💡 **ヒント**: フルLLM機能を使用するには `pip install gpt4all` が必要です。"
        "未インストールの場合はルールベースの簡易アドバイスが返されます。"
    )

    # サブタブ
    assistant_tabs = st.tabs([
        "📊 特徴量選択アドバイス",
        "📋 解析プラン提案",
        "🎯 結果解釈",
        "❓ 自由Q&A"
    ])

    with assistant_tabs[0]:
        _render_feature_suggestion()

    with assistant_tabs[1]:
        _render_analysis_plan()

    with assistant_tabs[2]:
        _render_result_interpretation()

    with assistant_tabs[3]:
        _render_free_qa()


def _render_feature_suggestion():
    """特徴量選択アドバイス"""
    st.subheader("📊 特徴量選択アドバイス")
    st.markdown("データセット情報から、最適な特徴量セットを推奨します。")

    with st.form("feature_suggest_form"):
        col1, col2 = st.columns(2)

        with col1:
            n_samples = st.number_input(
                "サンプル数", min_value=10, max_value=100000, value=500, step=10
            )
            task_type = st.selectbox("タスクタイプ", ["regression", "classification"])

        with col2:
            target_property = st.text_input(
                "予測対象の物性", value="solubility (logS)", placeholder="e.g., LogP, Tg"
            )

        submitted = st.form_submit_button("🤔 アドバイスを取得", use_container_width=True)

        if submitted:
            with st.spinner("LLMが考え中..."):
                try:
                    res = requests.post(
                        f"{API_URL}/llm/suggest-features",
                        json={
                            "n_samples": n_samples,
                            "task_type": task_type,
                            "target_property": target_property,
                        },
                        timeout=60,
                    )

                    if res.status_code == 200:
                        result = res.json()

                        st.success("✅ アドバイスを取得しました")

                        # 推奨特徴量
                        st.markdown("### 🎯 推奨特徴量")
                        for feat in result["recommended_features"]:
                            st.markdown(f"- **{feat}**")

                        # 理由
                        st.markdown("### 💡 理由")
                        st.write(result["reasoning"])

                        # 代替案
                        if result["alternative_features"]:
                            st.markdown("### 🔄 代替オプション")
                            for feat in result["alternative_features"]:
                                st.markdown(f"- {feat}")

                        # 考慮事項
                        with st.expander("📝 考慮事項"):
                            for consideration in result["considerations"]:
                                st.write(f"• {consideration}")

                    else:
                        st.error(f"エラー: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"リクエストエラー: {e}")


def _render_analysis_plan():
    """解析プラン提案"""
    st.subheader("📋 解析プラン提案")
    st.markdown("問題記述から、解析戦略を提案します。")

    with st.form("analysis_plan_form"):
        problem_description = st.text_area(
            "問題の説明",
            value="Predict aqueous solubility from SMILES",
            height=100,
            placeholder="What are you trying to predict?",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            n_samples = st.number_input("サンプル数", min_value=10, value=1200, step=10)

        with col2:
            task_type = st.selectbox("タスク", ["regression", "classification"])

        with col3:
            target_property = st.text_input("物性", value="logS")

        submitted = st.form_submit_button("💡 プランを提案", use_container_width=True)

        if submitted:
            with st.spinner("解析プランを作成中..."):
                try:
                    res = requests.post(
                        f"{API_URL}/llm/suggest-plan",
                        json={
                            "problem_description": problem_description,
                            "n_samples": n_samples,
                            "task_type": task_type,
                            "target_property": target_property,
                        },
                        timeout=60,
                    )

                    if res.status_code == 200:
                        result = res.json()

                        st.success("✅ 解析プランを作成しました")

                        # 目的
                        st.markdown("### 🎯 目的")
                        st.write(result["objective"])

                        # 推奨アプローチ
                        st.markdown("### 🧭 推奨アプローチ")
                        st.write(result["recommended_approach"])

                        # モデル候補
                        st.markdown("### 🤖 推奨モデル")
                        model_cols = st.columns(len(result["model_suggestions"]))
                        for i, model in enumerate(result["model_suggestions"]):
                            model_cols[i].info(model)

                        # 検証戦略
                        st.markdown("### ✅ 検証戦略")
                        st.write(result["validation_strategy"])

                        # 課題
                        with st.expander("⚠️ 想定される課題"):
                            for challenge in result["potential_challenges"]:
                                st.write(f"• {challenge}")

                    else:
                        st.error(f"エラー: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"リクエストエラー: {e}")


def _render_result_interpretation():
    """結果解釈"""
    st.subheader("🎯 モデル結果の解釈")
    st.markdown("評価指標から、結果の解釈と改善案を提案します。")

    with st.form("interpret_form"):
        st.markdown("#### 評価指標を入力")

        col1, col2 = st.columns(2)

        with col1:
            r2 = st.number_input("R² Score", min_value=-1.0, max_value=1.0, value=0.85, step=0.01)
            mae = st.number_input("MAE", min_value=0.0, value=0.42, step=0.01)

        with col2:
            rmse = st.number_input("RMSE", min_value=0.0, value=0.58, step=0.01)
            model_type = st.text_input("モデルタイプ", value="XGBoost")

        submitted = st.form_submit_button("🔍 結果を解釈", use_container_width=True)

        if submitted:
            with st.spinner("解釈中..."):
                try:
                    res = requests.post(
                        f"{API_URL}/llm/interpret-results",
                        json={
                            "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
                            "model_type": model_type,
                        },
                        timeout=60,
                    )

                    if res.status_code == 200:
                        result = res.json()

                        st.success("✅ 解釈が完了しました")

                        # 解釈
                        st.markdown("### 💭 解釈")
                        st.write(result["interpretation"])

                        # メトリクスサマリー
                        with st.expander("📊 入力されたメトリクス"):
                            metric_cols = st.columns(3)
                            metric_cols[0].metric("R²", f"{r2:.3f}")
                            metric_cols[1].metric("MAE", f"{mae:.3f}")
                            metric_cols[2].metric("RMSE", f"{rmse:.3f}")

                    else:
                        st.error(f"エラー: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"リクエストエラー: {e}")


def _render_free_qa():
    """自由形式Q&A"""
    st.subheader("❓ 自由Q&A")
    st.markdown("化学機械学習に関する質問に回答します。")

    # サンプル質問
    sample_questions = [
        "Morgan fingerprintsはいつ使うべきですか？",
        "小さいデータセット（<100サンプル）で過学習を避けるには？",
        "XGBoostとLightGBMの違いは？",
        "SHAP値の解釈方法は？",
    ]

    selected_sample = st.selectbox(
        "サンプル質問（または下に自由入力）",
        ["-- 自由入力 --"] + sample_questions
    )

    if selected_sample != "-- 自由入力 --":
        question = selected_sample
    else:
        question = st.text_area(
            "質問を入力",
            height=100,
            placeholder="例: バッチ正規化とは何ですか？",
        )

    context = st.text_input("コンテキスト（オプション）", placeholder="e.g., I'm working on QSAR modeling")

    if st.button("🤔 質問する", use_container_width=True):
        if not question:
            st.warning("質問を入力してください")
            return

        with st.spinner("考え中..."):
            try:
                payload = {"question": question}
                if context:
                    payload["context"] = context

                res = requests.post(
                    f"{API_URL}/llm/ask",
                    json=payload,
                    timeout=60,
                )

                if res.status_code == 200:
                    result = res.json()

                    st.success("✅ 回答が完了しました")

                    # 質問
                    st.markdown("### ❓ 質問")
                    st.info(result["question"])

                    # 回答
                    st.markdown("### 💡 回答")
                    st.write(result["answer"])

                    # LLM利用状況
                    if result.get("llm_available"):
                        st.caption("✨ GPT4All (Full LLM) を使用")
                    else:
                        st.caption("📋 ルールベースモード（簡易回答）")

                else:
                    st.error(f"エラー: {res.status_code} - {res.text}")

            except Exception as e:
                st.error(f"リクエストエラー: {e}")
