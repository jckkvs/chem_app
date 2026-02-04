"""
EDAダッシュボード - Streamlit版

人間がデータを丁寧に見るためのインタラクティブUI
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from typing import Optional

# APIエンドポイント
API_URL = "http://127.0.0.1:8000/api"


def render_eda_dashboard():
    """EDAダッシュボードをレンダリング"""
    st.title("📊 探索的データ分析（EDA）")
    st.markdown("**人間がデータを丁寧に見ることこそが最良のデータサイエンス**")
    
    st.info("""
    このダッシュボードは、データの本質を理解するための包括的なツールです：
    - 📈 データプロファイリング（基本統計、欠損値、重複）
    - 🔍 相関分析（ヒートマップ、高相関ペア検出）
    - 📊 Pairplot（散布図行列）
    - 📉 分布可視化（ヒストグラム + KDE）
    - 🎯 PCA/t-SNE（次元削減）
    - 🔢 K-meansクラスタリング
    """)
    
    # データセット選択
    st.sidebar.header("⚙️ 設定")
    
    try:
        response = requests.get(f"{API_URL}/datasets/")
        if response.status_code == 200:
            datasets = response.json()
            
            if not datasets:
                st.warning("データセットがありません。まずデータをアップロードしてください。")
                return
            
            dataset_options = {ds['name']: ds['id'] for ds in datasets}
            selected_name = st.sidebar.selectbox("データセット", list(dataset_options.keys()))
            dataset_id = dataset_options[selected_name]
            
            # データセット取得
            dataset_response = requests.get(f"{API_URL}/datasets/{dataset_id}/")
            if dataset_response.status_code == 200:
                dataset_info = dataset_response.json()
                
                # CSVファイルから直接読み込み（ローカル）
                import os
                file_path = dataset_info.get('file')
                if file_path and os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                else:
                    st.error("データファイルが見つかりません")
                    return
                
                # EDAツール読み込み
                from core.services.vis.eda_dashboard import EDADashboard
                
                # ターゲット列選択
                target_col = st.sidebar.selectbox(
                    "ターゲット変数（オプション）",
                    ["なし"] + df.columns.tolist()
                )
                target_col = None if target_col == "なし" else target_col
                
                # ダッシュボード初期化
                dashboard = EDADashboard(df, target_column=target_col)
                
                # タブ
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 データプロファイル",
                    "🔍 相関分析",
                    "📈 Pairplot",
                    "📉 分布",
                    "🎯 次元削減",
                    "🔢 クラスタリング"
                ])
                
                # タブ1: データプロファイル
                with tab1:
                    st.subheader("📊 データプロファイル")
                    
                    profile = dashboard.generate_data_profile()
                    
                    # 基本情報
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("行数", f"{profile['basic_info']['n_rows']:,}")
                    with col2:
                        st.metric("列数", f"{profile['basic_info']['n_columns']}")
                    with col3:
                        st.metric("数値列", f"{profile['basic_info']['n_numeric']}")
                    with col4:
                        st.metric("カテゴリ列", f"{profile['basic_info']['n_categorical']}")
                    
                    # 欠損値
                    st.markdown("### 欠損値")
                    missing = pd.DataFrame({
                        '列': list(profile['missing_values'].keys()),
                        '欠損数': list(profile['missing_values'].values()),
                        '欠損率(%)': [profile['missing_percent'][k] for k in profile['missing_values'].keys()],
                    })
                    missing = missing[missing['欠損数'] > 0].sort_values('欠損数', ascending=False)
                    if not missing.empty:
                        st.dataframe(missing, use_container_width=True)
                    else:
                        st.success("✅ 欠損値なし")
                    
                    # 重複
                    st.markdown("### 重複行")
                    dup = profile['duplicates']
                    st.write(f"重複数: **{dup['n_duplicates']:,}** ({dup['duplicate_percent']:.2f}%)")
                    
                    # 数値列統計
                    if 'numeric_stats' in profile:
                        st.markdown("### 数値列統計")
                        st.dataframe(pd.DataFrame(profile['numeric_stats']), use_container_width=True)
                
                # タブ2: 相関分析
                with tab2:
                    st.subheader("🔍 相関分析")
                    
                    method = st.selectbox("相関係数", ["pearson", "spearman", "kendall"])
                    top_n = st.slider("表示特徴量数", 5, min(30, len(dashboard.numeric_cols)), 15)
                    
                    if st.button("相関ヒートマップ生成", type="primary"):
                        with st.spinner("生成中..."):
                            fig = dashboard.plot_correlation_heatmap(method=method, top_n=top_n, show=False)
                            if fig:
                                st.pyplot(fig)
                    
                    # 高相関ペア
                    st.markdown("### 高相関ペア")
                    threshold = st.slider("相関閾値", 0.5, 0.99, 0.7, 0.05)
                    high_corr = dashboard.find_high_correlations(threshold=threshold)
                    if not high_corr.empty:
                        st.dataframe(high_corr, use_container_width=True)
                    else:
                        st.info(f"閾値 {threshold} 以上の相関ペアは見つかりませんでした")
                
                # タブ3: Pairplot
                with tab3:
                    st.subheader("📈 Pairplot（散布図行列）")
                    
                    max_features = st.slider("特徴量数", 3, 8, 5)
                    hue_col = st.selectbox("色分け（hue）", ["なし"] + df.columns.tolist())
                    hue_col = None if hue_col == "なし" else hue_col
                    
                    if st.button("Pairplot生成", type="primary"):
                        with st.spinner("生成中（時間がかかる場合があります）..."):
                            grid = dashboard.plot_pairplot(max_features=max_features, hue=hue_col, show=False)
                            if grid:
                                st.pyplot(grid.fig)
                
                # タブ4: 分布
                with tab4:
                    st.subheader("📉 特徴量分布")
                    
                    if st.button("分布プロット生成", type="primary"):
                        with st.spinner("生成中..."):
                            fig = dashboard.plot_distributions(show=False)
                            if fig:
                                st.pyplot(fig)
                
                # タブ5: 次元削減
                with tab5:
                    st.subheader("🎯 次元削減可視化")
                    
                    method = st.radio("手法", ["PCA", "t-SNE"])
                    color_col = st.selectbox("色分け", ["なし"] + df.columns.tolist(), key="dim_color")
                    color_col = None if color_col == "なし" else color_col
                    
                    if st.button(f"{method}実行", type="primary"):
                        with st.spinner(f"{method}実行中..."):
                            if method == "PCA":
                                fig, pca = dashboard.plot_pca(color_by=color_col, show=False)
                                if fig:
                                    st.pyplot(fig)
                            else:  # t-SNE
                                perplexity = st.slider("Perplexity", 5, 50, 30)
                                fig = dashboard.plot_tsne(perplexity=perplexity, color_by=color_col, show=False)
                                if fig:
                                    st.pyplot(fig)
                
                # タブ6: クラスタリング
                with tab6:
                    st.subheader("🔢 K-meansクラスタリング")
                    
                    n_clusters = st.slider("クラスタ数", 2, 10, 3)
                    
                    if st.button("クラスタリング実行", type="primary"):
                        with st.spinner("実行中..."):
                            fig, kmeans = dashboard.plot_kmeans_clusters(n_clusters=n_clusters, show=False)
                            if fig:
                                st.pyplot(fig)
                                st.success(f"✅ Inertia: {kmeans.inertia_:.2f}")
                
        else:
            st.error("データセット取得に失敗しました")
    
    except Exception as e:
        st.error(f"エラー: {e}")
        import traceback
        st.code(traceback.format_exc())
