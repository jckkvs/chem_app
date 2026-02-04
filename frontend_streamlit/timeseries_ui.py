"""
時系列UI - Streamlit時系列データ対応

Implements: F-TIME-SERIES-UI-001
設計思想:
- 時系列データの自動検出
- 日時カラム選択UI
- 予測期間指定
- 時系列可視化
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

def render_timeseries_settings(df: pd.DataFrame) -> dict:
    """
    時系列設定UIをレンダリング
    
    Args:
        df: データフレーム
        
    Returns:
        dict: 時系列設定
    """
    st.subheader("⏰ 時系列データ設定")
    
    settings = {}
    
    # 時系列データ検出
    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    if not datetime_columns:
        # datetime型カラムがない場合、変換を提案
        st.info("datetime型のカラムが見つかりません。カラムを選択して変換してください。")
        
        candidate_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if candidate_columns:
            st.write("**日時候補カラム:**", candidate_columns)
        
        date_column = st.selectbox(
            "日時カラムを選択",
            options=df.columns.tolist(),
            help="日時情報を含むカラムを選択してください"
        )
        
        # 日時変換オプション
        date_format = st.text_input(
            "日時フォーマット（オプション）",
            value="",
            help="例: %Y-%m-%d, %Y/%m/%d %H:%M:%S など。空欄の場合は自動推測"
        )
        
        settings['date_column'] = date_column
        settings['date_format'] = date_format if date_format else None
        settings['needs_conversion'] = True
        
    else:
        # datetime型カラムがある場合
        st.success(f"✅ {len(datetime_columns)}個の日時カラムを検出しました")
        
        date_column = st.selectbox(
            "日時カラムを選択",
            options=datetime_columns,
            help="予測に使用する日時カラムを選択してください"
        )
        
        settings['date_column'] = date_column
        settings['needs_conversion'] = False
    
    # 時系列モデル選択
    st.subheader("📊 時系列モデル選択")
    
    model_type = st.selectbox(
        "モデルタイプ",
        options=['Prophet', 'ARIMA', 'SARIMA', 'LSTM', 'Transformer'],
        help="""
        - Prophet: Facebook製、季節性・休日対応
        - ARIMA: 古典的時系列モデル
        - SARIMA: 季節性ARIMA
        - LSTM: ディープラーニング（長期依存）
        - Transformer: 最新のディープラーニング
        """
    )
    settings['model_type'] = model_type.lower()
    
    # モデル別パラメータ
    st.subheader("⚙️ モデルパラメータ")
    
    if model_type == 'Prophet':
        col1, col2 = st.columns(2)
        with col1:
            growth = st.selectbox("成長トレンド", ['linear', 'logistic'], index=0)
            yearly_seasonality = st.checkbox("年次季節性", value=True)
        with col2:
            changepoint_prior_scale = st.slider("変化点感度", 0.001, 0.5, 0.05, 0.001)
            weekly_seasonality = st.checkbox("週次季節性", value=True)
        
        settings['model_params'] = {
            'growth': growth,
            'changepoint_prior_scale': changepoint_prior_scale,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
        }
    
    elif model_type in ['ARIMA', 'SARIMA']:
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR次数 (p)", 0, 10, 1)
        with col2:
            d = st.number_input("差分次数 (d)", 0, 2, 1)
        with col3:
            q = st.number_input("MA次数 (q)", 0, 10, 1)
        
        settings['model_params'] = {
            'order': (p, d, q)
        }
        
        if model_type == 'SARIMA':
            st.write("**季節性パラメータ**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                P = st.number_input("季節AR (P)", 0, 10, 1)
            with col2:
                D = st.number_input("季節差分 (D)", 0, 2, 1)
            with col3:
                Q = st.number_input("季節MA (Q)", 0, 10, 1)
            with col4:
                S = st.number_input("周期 (S)", 1, 365, 12)
            
            settings['model_params']['seasonal_order'] = (P, D, Q, S)
    
    # 予測期間設定
    st.subheader("🔮 予測設定")
    
    forecast_horizon = st.number_input(
        "予測期間（ステップ数）",
        min_value=1,
        max_value=1000,
        value=30,
        help="何ステップ先まで予測するか"
    )
    settings['forecast_horizon'] = forecast_horizon
    
    # 時系列分割設定
    st.subheader("✂️ 時系列交差検証")
    
    use_timeseries_cv = st.checkbox("時系列交差検証を使用", value=True)
    
    if use_timeseries_cv:
        col1, col2 = st.columns(2)
        with col1:
            n_splits = st.number_input("分割数", 2, 10, 5)
        with col2:
            test_size = st.number_input("テストサイズ", 1, 1000, 30)
        
        settings['cv_params'] = {
            'n_splits': n_splits,
            'test_size': test_size
        }
    
    return settings


def convert_to_datetime(df: pd.DataFrame, column: str, date_format: Optional[str] = None) -> pd.DataFrame:
    """
    カラムをdatetime型に変換
    
    Args:
        df: データフレーム
        column: 変換するカラム名
        date_format: 日時フォーマット
        
    Returns:
        変換後のデータフレーム
    """
    df = df.copy()
    
    try:
        if date_format:
            df[column] = pd.to_datetime(df[column], format=date_format)
        else:
            df[column] = pd.to_datetime(df[column])
        
        st.success(f"✅ '{column}' をdatetime型に変換しました")
    except Exception as e:
        st.error(f"❌ 変換エラー: {e}")
        st.info("日時フォーマットを明示的に指定してください")
    
    return df


def validate_timeseries_data(df: pd.DataFrame, date_column: str, target_column: str) -> bool:
    """
    時系列データの妥当性チェック
    
    Args:
        df: データフレーム
        date_column: 日時カラム
        target_column: 目的変数カラム
        
    Returns:
        True if valid
    """
    issues = []
    
    # 日時カラムチェック
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        issues.append(f"'{date_column}' がdatetime型ではありません")
    
    # 欠損値チェック
    if df[date_column].isnull().any():
        issues.append(f"'{date_column}' に欠損値があります（{df[date_column].isnull().sum()}件）")
    
    if df[target_column].isnull().any():
        issues.append(f"'{target_column}' に欠損値があります（{df[target_column].isnull().sum()}件）")
    
    # ソートチェック
    if not df[date_column].is_monotonic_increasing:
        issues.append(f"'{date_column}' が昇順にソートされていません")
        if st.checkbox("自動ソートする"):
            df.sort_values(date_column, inplace=True)
            st.success("✅ 日時でソートしました")
    
    # 結果表示
    if issues:
        st.warning("⚠️ データ品質の問題が検出されました:")
        for issue in issues:
            st.write(f"- {issue}")
        return False
    else:
        st.success("✅ 時系列データは妥当です")
        return True
