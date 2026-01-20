"""
Experiment Comparison ページ

Implements: F-COMP-001
設計思想:
- 複数実験のメトリクス比較
- チャート可視化
- 必要なimportを完備
"""

import streamlit as st
import requests
import pandas as pd

# APIエンドポイント
API_URL = "http://127.0.0.1:8000/api"


def render_comparison():
    """
    モデル比較ページをレンダリング
    
    Features:
    - 完了した実験の一覧表示
    - メトリクス比較テーブル
    - パフォーマンスチャート
    """
    st.header("📊 Model Comparison")
    
    try:
        res = requests.get(f"{API_URL}/experiments", timeout=10)
        
        if res.status_code != 200:
            st.error(f"APIエラー: {res.status_code}")
            return
        
        experiments = res.json()
        
        if not experiments:
            st.info("実験がまだありません。Experimentsタブで新しい実験を作成してください。")
            return
        
        # 完了した実験のみフィルタ
        completed = [
            exp for exp in experiments 
            if exp['status'] == 'COMPLETED' and exp.get('metrics')
        ]
        
        if not completed:
            st.warning("完了済みの実験がありません。")
            return
        
        # 比較テーブル構築
        data = []
        for exp in completed:
            row = {
                'ID': exp['id'],
                'Name': exp['name'],
                'Model': exp['config'].get('model_type', 'N/A'),
                'Features': ", ".join(exp['config'].get('features', [])),
            }
            
            # メトリクスを追加
            metrics = exp.get('metrics', {})
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row[k] = v
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # テーブル表示
        st.subheader("📋 Experiment Results")
        st.dataframe(df, use_container_width=True)
        
        # メトリクス選択
        st.subheader("📈 Performance Chart")
        metric_cols = [
            c for c in df.columns 
            if c not in ['ID', 'Name', 'Model', 'Features']
        ]
        
        if not metric_cols:
            st.info("比較可能なメトリクスがありません。")
            return
        
        col1, col2 = st.columns([2, 2])
        
        with col1:
            selected_metric = st.selectbox(
                "比較するメトリクスを選択", 
                metric_cols,
                index=0,
            )
        
        with col2:
            chart_type = st.selectbox(
                "チャートタイプ",
                ["バーチャート", "ラインチャート"],
            )
        
        # チャート描画
        if selected_metric and not df[selected_metric].dropna().empty:
            chart_df = df.set_index('Name')[selected_metric].dropna()
            
            if chart_type == "バーチャート":
                st.bar_chart(chart_df)
            else:
                st.line_chart(chart_df)
        else:
            st.warning(f"'{selected_metric}' のデータがありません。")
        
        # ベストモデル表示
        st.subheader("🏆 Best Models")
        
        for metric in metric_cols[:3]:  # 上位3メトリクス
            valid_df = df.dropna(subset=[metric])
            if not valid_df.empty:
                # メトリクス名から最大/最小を判断
                if any(word in metric.lower() for word in ['loss', 'error', 'mse', 'mae', 'rmse']):
                    best_idx = valid_df[metric].idxmin()
                else:
                    best_idx = valid_df[metric].idxmax()
                
                best_row = valid_df.loc[best_idx]
                st.metric(
                    label=f"Best {metric}",
                    value=f"{best_row[metric]:.4f}",
                    delta=best_row['Name'],
                )
                
    except requests.exceptions.ConnectionError:
        st.error("⚠️ APIサーバーに接続できません。Django サーバーが起動しているか確認してください。")
    except Exception as e:
        st.error(f"エラー: {e}")
