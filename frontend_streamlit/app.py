"""
Chemical ML Platform - Streamlit Frontend

Implements: F-UI-001
設計思想:
- 初心者にはシンプル、熟練者には詳細設定
- プログレッシブ開示UI
- リアルタイム可視化
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import base64
from typing import Optional, Dict, Any

# TARTE利用可能性チェック（遅延インポート）
def _check_tarte_available() -> bool:
    try:
        from core.services.features.tarte_eng import is_tarte_available
        return is_tarte_available()
    except ImportError:
        return False

try:
    from comparison import render_comparison
except ImportError:
    try:
        from frontend_streamlit.comparison import render_comparison
    except ImportError:
        def render_comparison():
            st.warning("Comparison module not found")


# API設定
API_URL = "http://127.0.0.1:8000/api"

# ページ設定
st.set_page_config(
    page_title="ChemML Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """メインエントリーポイント"""
    st.title("🧪 Chemical ML Platform")
    st.markdown("*機械学習を使った分子物性予測プラットフォーム*")
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ Settings")
        show_advanced = st.checkbox("詳細設定を表示", value=False)
        auto_refresh = st.checkbox("自動更新 (10秒)", value=False)
        
        # TARTE設定
        with st.expander("🤖 TARTE Settings", expanded=False):
            tarte_available = _check_tarte_available()
            if tarte_available:
                st.success("✅ tarte-ai インストール済み")
                use_tarte = st.checkbox("TARTEを使用", value=False, key="use_tarte_global")
                if use_tarte:
                    tarte_mode = st.radio(
                        "モード選択",
                        ["Featurizer（高速）", "Finetuning（高精度）", "Boosting（最高精度）"],
                        help="""
                        **Featurizer**: 事前学習モデルで埋め込み生成（推奨）
                        **Finetuning**: データに合わせてモデル調整
                        **Boosting**: アンサンブル学習（最も時間がかかる）
                        """,
                        key="tarte_mode_global"
                    )
                    # セッション状態に保存
                    mode_map = {
                        "Featurizer（高速）": "featurizer",
                        "Finetuning（高精度）": "finetuning",
                        "Boosting（最高精度）": "boosting",
                    }
                    st.session_state["tarte_mode"] = mode_map.get(tarte_mode, "featurizer")
            else:
                st.warning("⚠️ tarte-aiがインストールされていません")
                st.markdown("表形式データのTransformer特徴量を使用するにはインストールが必要です:")
                st.code("pip install tarte-ai", language="bash")
                st.caption("[📚 TARTE Documentation](https://github.com/soda-inria/tarte-ai)")
        
        st.divider()
        st.caption("Version 2.2 - TARTE Enhanced")
    
    # メインタブ
    tabs = st.tabs([
        "📂 Datasets",
        "⚗️ Experiments",
        "📊 Analysis",
        "🔬 Molecule Viewer",
        "📦 Batch Predict",
        "⚖️ Comparison",
        "🤖 LLM Assistant",  # NEW
    ])

    with tabs[0]:
        render_datasets()
    with tabs[1]:
        render_experiments(show_advanced)
    with tabs[2]:
        render_analysis()
    with tabs[3]:
        render_molecule_viewer()
    with tabs[4]:
        render_batch_predict()
    with tabs[5]:
        render_comparison()
    with tabs[6]:  # NEW
        render_llm_assistant()

    # 自動更新
    if auto_refresh:
        time.sleep(10)
        st.rerun()



def render_datasets():
    """データセット管理ページ"""
    st.header("📂 Dataset Management")
    
    # アップロードフォーム
    with st.expander("📤 Upload New Dataset", expanded=False):
        with st.form("upload_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Dataset Name", placeholder="e.g., Solubility Dataset")
                uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            
            with col2:
                smiles_col = st.text_input("SMILES Column", value="SMILES")
                target_col = st.text_input("Target Column", value="target")
            
            submitted = st.form_submit_button("Upload", use_container_width=True)
            
            if submitted and uploaded_file:
                _upload_dataset(name, uploaded_file, smiles_col, target_col)
    
    # データセット一覧
    st.subheader("📋 Available Datasets")
    _display_datasets()


def _upload_dataset(name: str, file, smiles_col: str, target_col: str):
    """データセットをアップロード"""
    files = {'file': (file.name, file, 'text/csv')}
    data = {
        'name': name or file.name,
        'smiles_col': smiles_col,
        'target_col': target_col,
    }
    
    try:
        with st.spinner("Uploading..."):
            res = requests.post(f"{API_URL}/datasets", files=files, data=data, timeout=30)
        
        if res.status_code == 200:
            st.success(f"✅ Dataset '{name}' uploaded successfully!")
            st.rerun()
        else:
            st.error(f"Error: {res.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API server")
    except Exception as e:
        st.error(f"Error: {e}")


def _display_datasets():
    """データセット一覧を表示"""
    try:
        res = requests.get(f"{API_URL}/datasets", timeout=10)
        
        if res.status_code == 200:
            datasets = res.json()
            if datasets:
                df = pd.DataFrame(datasets)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No datasets found. Upload one to get started!")
        else:
            st.error("Failed to fetch datasets")
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API server not available")
    except Exception as e:
        st.warning(f"Could not load datasets: {e}")


def render_experiments(show_advanced: bool = False):
    """実験セットアップページ"""
    st.header("⚗️ Experiment Setup")
    
    # タスクタイプ選択
    st.subheader("📋 タスクタイプ")
    task_type = st.radio(
        "予測タスクを選択",
        [
            "① SMILES → 物性予測",
            "② 表データ → 特性予測", 
            "③ 混合物（SMILES＋割合） → 物性予測",
            "④ SMILES＋表データ → 物性予測",
        ],
        horizontal=True,
        help="データの形式に応じて選択してください"
    )
    
    # タスクタイプをconfig用に変換
    task_type_map = {
        "① SMILES → 物性予測": "smiles_only",
        "② 表データ → 特性予測": "tabular_only",
        "③ 混合物（SMILES＋割合） → 物性予測": "mixture",
        "④ SMILES＋表データ → 物性予測": "smiles_tabular",
    }
    selected_task_type = task_type_map.get(task_type, "smiles_only")
    
    st.divider()
    
    # 実験作成フォーム
    with st.form("exp_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            exp_name = st.text_input("Experiment Name", placeholder="e.g., LogP Prediction v1")
            
            # データセット選択
            dataset_options = _fetch_dataset_options()
            selected_ds_label = st.selectbox(
                "Select Dataset",
                list(dataset_options.keys()) if dataset_options else ["No datasets available"]
            )
        
        with col2:
            model_type = st.selectbox(
                "Model Type",
                ["lightgbm", "xgboost", "random_forest"],
                help="LightGBM is recommended for most cases"
            )
            
            # タスクタイプに応じた特徴量選択
            if selected_task_type == "tabular_only":
                features = ["tabular"]
                st.info("📊 表データモード：データセットの数値カラムを使用")
            elif selected_task_type == "mixture":
                features = ["mixture"]
                st.info("🧪 混合物モード：SMILES + 割合から加重平均記述子を計算")
            else:
                # SMILES系タスク: 通常の特徴量選択 + TARTE対応
                available_features = ["rdkit", "xtb", "uma"]
                feature_help = "rdkit: Molecular descriptors, xtb: Quantum properties, uma: UMAP embeddings"
                
                # TARTEが有効な場合は追加（表データ混合タスク向け）
                if st.session_state.get("use_tarte_global", False):
                    if selected_task_type == "smiles_tabular":
                        available_features.append("tarte")
                        feature_help += ", tarte: Transformer tabular features"
                
                features = st.multiselect(
                    "Features",
                    available_features,
                    default=["rdkit"],
                    help=feature_help
                )
        
        # 物性プリセット選択（Smart Feature Engineering）
        st.divider()
        st.subheader("🎯 目的物性（Smart Feature Selection）")
        
        # 物性カテゴリ別のプリセット
        property_options = {
            "-- 汎用 --": {
                "general": "汎用（バランス型）",
            },
            "-- 光学物性 --": {
                "refractive_index": "屈折率",
                "optical_gap": "光学バンドギャップ",
            },
            "-- 機械物性 --": {
                "elastic_modulus": "弾性率・ヤング率",
                "tensile_strength": "引張強度",
                "hardness": "硬度",
            },
            "-- 熱物性 --": {
                "glass_transition": "ガラス転移温度(Tg)",
                "melting_point": "融点",
                "thermal_conductivity": "熱伝導率",
                "thermal_stability": "熱安定性",
            },
            "-- 電気物性 --": {
                "dielectric_constant": "誘電率",
                "conductivity": "電気伝導度",
            },
            "-- 化学物性 --": {
                "solubility": "溶解度・LogP",
                "viscosity": "粘度",
                "density": "密度",
            },
            "-- 輸送物性 --": {
                "gas_permeability": "ガス透過性",
            },
            "-- 薬理学 --": {
                "admet": "ADMET・薬物動態",
                "pka": "pKa",
            },
        }
        
        # フラットなリストに変換（表示用）
        flat_options = []
        option_to_key = {}
        for category, presets in property_options.items():
            flat_options.append(category)
            for key, name in presets.items():
                display = f"  {name}"
                flat_options.append(display)
                option_to_key[display] = key
        
        selected_property_display = st.selectbox(
            "目的物性を選択（最適な記述子セットを自動選択）",
            flat_options,
            index=1,  # "汎用" をデフォルト
            help="予測したい物性に応じて、最適な分子記述子セットが自動選択されます"
        )
        
        # カテゴリヘッダーは選択できない
        if selected_property_display.startswith("--"):
            target_property = "general"
        else:
            target_property = option_to_key.get(selected_property_display, "general")
        
        # 詳細設定（プログレッシブ開示）
        if show_advanced:
            st.divider()
            st.subheader("🔧 Advanced Settings")
            
            adv_col1, adv_col2 = st.columns(2)
            
            with adv_col1:
                cv_folds = st.slider("CV Folds", min_value=2, max_value=10, value=5)
                ml_task_type = st.selectbox("Task Type", ["regression", "classification"])
            
            with adv_col2:
                preprocessor = st.selectbox(
                    "Preprocessor Preset",
                    ["tree_optimized", "default", "robust", "normalized"],
                    help="tree_optimized is recommended for tree-based models"
                )
                use_smart_engine = st.checkbox(
                    "SmartFeatureEngine使用",
                    value=True,
                    help="物性×データセット特性に基づく最適特徴量選択"
                )
            
            # 事前学習モデル選択
            st.markdown("**🤖 事前学習モデル（オプション）**")
            pretrained_col1, pretrained_col2 = st.columns(2)
            with pretrained_col1:
                use_unimol = st.checkbox("Uni-Mol（3D構造）", value=False)
            with pretrained_col2:
                use_chemberta = st.checkbox("ChemBERTa（SMILES）", value=False)
            
            pretrained_models = []
            if use_unimol:
                pretrained_models.append("unimol")
            if use_chemberta:
                pretrained_models.append("chemberta")
        else:
            cv_folds = 5
            ml_task_type = "regression"
            preprocessor = "tree_optimized"
            use_smart_engine = True
            pretrained_models = []
        
        submitted = st.form_submit_button("🚀 Start Experiment", use_container_width=True)
        
        if submitted and dataset_options and selected_ds_label in dataset_options:
            _start_experiment(
                dataset_id=dataset_options[selected_ds_label],
                name=exp_name,
                features=features,
                model_type=model_type,
                cv_folds=cv_folds,
                task_type=ml_task_type,
                task_type_mode=selected_task_type,
                target_property=target_property,
                use_smart_engine=use_smart_engine,
                pretrained_models=pretrained_models,
            )
    
    # 実験一覧
    st.subheader("📋 Recent Experiments")
    _display_experiments()


def _fetch_dataset_options() -> Dict[str, int]:
    """データセットオプションを取得"""
    options = {}
    try:
        res = requests.get(f"{API_URL}/datasets", timeout=10)
        if res.status_code == 200:
            for d in res.json():
                options[f"{d['id']}: {d['name']}"] = d['id']
    except Exception:
        pass
    return options


def _start_experiment(
    dataset_id: int,
    name: str,
    features: list,
    model_type: str,
    cv_folds: int,
    task_type: str,
    task_type_mode: str = "smiles_only",
    target_property: str = "general",
    use_smart_engine: bool = True,
    pretrained_models: list = None,
):
    """
    実験を開始
    
    Args:
        dataset_id: データセットID
        name: 実験名
        features: 使用する特徴量リスト
        model_type: モデルタイプ
        cv_folds: CV分割数
        task_type: タスクタイプ（regression/classification）
        task_type_mode: データタイプ（smiles_only/tabular_only/mixture/smiles_tabular）
        target_property: 目的物性プリセット
        use_smart_engine: SmartFeatureEngine使用フラグ
        pretrained_models: 使用する事前学習モデルリスト
    """
    payload = {
        "dataset_id": dataset_id,
        "name": name or f"Experiment_{int(time.time())}",
        "features": features,
        "model_type": model_type,
        "cv_folds": cv_folds,
        "task_type": task_type,
        "task_type_mode": task_type_mode,
        "target_property": target_property,
        "use_smart_engine": use_smart_engine,
        "pretrained_models": pretrained_models or [],
    }
    
    try:
        with st.spinner("Starting experiment..."):
            res = requests.post(f"{API_URL}/experiments", json=payload, timeout=30)
        
        if res.status_code == 200:
            exp_data = res.json()
            st.success(f"✅ Experiment started! ID: {exp_data['id']}")
            
            # 設定サマリー表示
            with st.expander("📋 Experiment Settings", expanded=False):
                st.write(f"**Task Mode:** {task_type_mode}")
                st.write(f"**Target Property:** {target_property}")
                st.write(f"**Model:** {model_type}")
                st.write(f"**Features:** {', '.join(features)}")
                if pretrained_models:
                    st.write(f"**Pretrained Models:** {', '.join(pretrained_models)}")
            
            st.info("Check the Analysis tab for results once completed.")
        else:
            st.error(f"Failed: {res.text}")
    except Exception as e:
        st.error(f"Error: {e}")


def _display_experiments():
    """実験一覧を表示"""
    try:
        res = requests.get(f"{API_URL}/experiments", timeout=10)
        if res.status_code == 200:
            experiments = res.json()
            if experiments:
                df = pd.DataFrame(experiments)
                # ステータスに色を付ける
                st.dataframe(df, use_container_width=True)
    except Exception:
        pass


def render_analysis():
    """結果分析ページ"""
    st.header("📊 Results Analysis")
    
    # 実験選択
    exp_options = _fetch_experiment_options()
    
    if not exp_options:
        st.info("No experiments available. Create one first!")
        return
    
    selected_exp = st.selectbox("Select Experiment", list(exp_options.keys()))
    
    if not selected_exp:
        return
    
    exp_id = exp_options[selected_exp]
    
    try:
        res = requests.get(f"{API_URL}/experiments/{exp_id}", timeout=10)
        
        if res.status_code != 200:
            st.error("Failed to load experiment details")
            return
        
        exp = res.json()
        
        # ステータス表示
        status = exp['status']
        if status == 'COMPLETED':
            st.success(f"Status: ✅ {status}")
        elif status == 'RUNNING':
            st.warning(f"Status: ⏳ {status}")
        elif status == 'FAILED':
            st.error(f"Status: ❌ {status}")
        else:
            st.info(f"Status: {status}")
        
        # メトリクス
        if exp.get('metrics'):
            st.subheader("📈 Validation Metrics")
            metrics = exp['metrics']
            
            # エラーがある場合
            if 'error' in metrics:
                st.error(f"Error: {metrics['error']}")
            else:
                # メトリクスをカラムで表示
                cols = st.columns(4)
                for i, (k, v) in enumerate(metrics.items()):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        cols[i % 4].metric(k, f"{v:.4f}")
        
        # 完了した実験の詳細
        if status == 'COMPLETED':
            st.divider()
            _render_completed_analysis(exp_id, exp)
            
    except Exception as e:
        st.error(f"Error: {e}")


def _fetch_experiment_options() -> Dict[str, int]:
    """実験オプションを取得"""
    options = {}
    try:
        res = requests.get(f"{API_URL}/experiments", timeout=10)
        if res.status_code == 200:
            for e in res.json():
                label = f"{e['id']}: {e['name']} ({e['status']})"
                options[label] = e['id']
    except Exception:
        pass
    return options


def _render_completed_analysis(exp_id: int, exp: Dict[str, Any]):
    """完了した実験の分析を表示"""
    # 可視化
    st.subheader("🎨 Global Explanations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**SHAP Summary**")
        _display_artifact(exp_id, "shap_summary.png")
    
    with col2:
        st.markdown("**Feature Importance**")
        _display_artifact(exp_id, "feature_importance.png")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Learning Curve**")
        _display_artifact(exp_id, "learning_curve.png")
    
    with col4:
        st.markdown("**Predicted vs Actual**")
        _display_artifact(exp_id, "predicted_vs_actual.png")
    
    # インタラクティブ予測
    st.divider()
    st.subheader("🔮 Interactive Prediction")
    
    smi_input = st.text_input("Enter SMILES", value="c1ccccc1", placeholder="e.g., CCO, c1ccccc1")
    
    if st.button("Predict", use_container_width=True):
        _run_prediction(exp_id, smi_input)


def _display_artifact(exp_id: int, filename: str):
    """アーティファクトを表示"""
    try:
        res = requests.get(f"{API_URL}/experiments/{exp_id}/artifacts/{filename}", timeout=10)
        if res.status_code == 200:
            st.image(res.content, use_container_width=True)
        else:
            st.info("Not available")
    except Exception:
        st.info("Not available")


def _run_prediction(exp_id: int, smiles: str):
    """予測を実行"""
    try:
        with st.spinner("Calculating..."):
            res = requests.post(
                f"{API_URL}/experiments/{exp_id}/predict",
                json={"smiles": smiles},
                timeout=30,
            )
        
        if res.status_code == 200:
            data = res.json()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Prediction", f"{data['prediction']:.4f}")
            
            with col2:
                if data.get('shap_image'):
                    st.markdown("**SHAP Explanation**")
                    img_bytes = base64.b64decode(data['shap_image'])
                    st.image(img_bytes, use_container_width=True)
        else:
            st.error(f"Prediction failed: {res.text}")
            
    except Exception as e:
        st.error(f"Error: {e}")


def render_molecule_viewer():
    """分子ビューワーページ"""
    st.header("🔬 Molecule Viewer")
    st.markdown("SMILESから分子構造と物性を表示")
    
    smiles_input = st.text_input("SMILES入力", value="c1ccccc1", placeholder="例: CCO, CC(=O)O")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if smiles_input:
            st.subheader("分子構造")
            try:
                # APIから分子SVGを取得
                svg_res = requests.get(f"{API_URL}/molecules/{smiles_input}/svg", timeout=10)
                if svg_res.status_code == 200:
                    st.image(svg_res.content, use_container_width=True)
                else:
                    st.warning("分子の描画に失敗しました")
            except Exception as e:
                st.error(f"SVG取得エラー: {e}")
    
    with col2:
        if smiles_input:
            st.subheader("物性情報")
            try:
                props_res = requests.get(f"{API_URL}/molecules/{smiles_input}/properties", timeout=10)
                if props_res.status_code == 200:
                    props = props_res.json()
                    st.metric("分子量", f"{props['molecular_weight']:.2f} g/mol")
                    st.metric("LogP", f"{props['logp']:.2f}")
                    st.metric("TPSA", f"{props['tpsa']:.2f} Å²")
                    
                    with st.expander("詳細情報"):
                        st.write(f"水素結合ドナー: {props['hbd']}")
                        st.write(f"水素結合アクセプター: {props['hba']}")
                        st.write(f"回転可能結合: {props['rotatable_bonds']}")
                        st.write(f"環の数: {props['num_rings']}")
                        st.write(f"原子数: {props['num_atoms']}")
                else:
                    st.warning("物性計算に失敗しました")
            except Exception as e:
                st.error(f"物性取得エラー: {e}")
    
    # SMILES検証
    st.divider()
    st.subheader("SMILES検証")
    validate_smiles = st.text_input("検証するSMILES", key="validate_smiles")
    if st.button("検証"):
        try:
            res = requests.post(f"{API_URL}/molecules/validate", json={"smiles": validate_smiles}, timeout=10)
            data = res.json()
            if data['valid']:
                st.success(f"✅ 有効なSMILES - 正規化: `{data['canonical_smiles']}`")
            else:
                st.error(f"❌ 無効なSMILES: {data['error']}")
        except Exception as e:
            st.error(f"検証エラー: {e}")


def render_batch_predict():
    """バッチ予測ページ"""
    st.header("📦 バッチ予測")
    st.markdown("複数のSMILESに対して一括で予測を実行")
    
    # 実験選択
    exp_options = _fetch_experiment_options()
    completed_options = {k: v for k, v in exp_options.items() if "COMPLETED" in k}
    
    if not completed_options:
        st.warning("完了した実験がありません。まず実験を作成・完了させてください。")
        return
    
    selected_exp = st.selectbox("実験を選択", list(completed_options.keys()))
    exp_id = completed_options.get(selected_exp)
    
    # 入力方式選択
    input_method = st.radio("入力方法", ["テキスト入力", "CSVアップロード"], horizontal=True)
    
    smiles_list = []
    
    if input_method == "テキスト入力":
        smiles_text = st.text_area(
            "SMILESリスト（1行に1つ）",
            value="CCO\nc1ccccc1\nCC(=O)O\nCCCCC",
            height=200
        )
        smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
    else:
        uploaded = st.file_uploader("CSVファイル（SMILESカラムを含む）", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            smiles_col = st.selectbox("SMILESカラム", df.columns.tolist())
            smiles_list = df[smiles_col].dropna().tolist()
            st.info(f"{len(smiles_list)}件のSMILESを読み込みました")
    
    if smiles_list and st.button("🚀 バッチ予測実行", use_container_width=True):
        with st.spinner(f"{len(smiles_list)}件のSMILESを処理中..."):
            try:
                res = requests.post(
                    f"{API_URL}/experiments/{exp_id}/batch_predict",
                    json={"smiles_list": smiles_list},
                    timeout=120
                )
                
                if res.status_code == 200:
                    data = res.json()
                    predictions = data['predictions']
                    
                    st.success(f"✅ {len(predictions)}件の予測が完了しました")
                    
                    # 結果表示
                    result_df = pd.DataFrame(predictions)
                    st.dataframe(result_df, use_container_width=True)
                    
                    # CSV出力
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 結果をCSVでダウンロード",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # 簡易統計
                    if 'prediction' in result_df.columns:
                        st.subheader("📊 統計情報")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("平均", f"{result_df['prediction'].mean():.4f}")
                        col2.metric("標準偏差", f"{result_df['prediction'].std():.4f}")
                        col3.metric("最小", f"{result_df['prediction'].min():.4f}")
                        col4.metric("最大", f"{result_df['prediction'].max():.4f}")
                else:
                    st.error(f"バッチ予測失敗: {res.text}")
                    
            except Exception as e:
                st.error(f"エラー: {e}")


if __name__ == "__main__":
    main()

"""
LLM Assistant UI - Append to frontend_streamlit/app.py
"""


def render_llm_assistant():
    """LLMアシスタント�Eージ"""
    st.header("🤁ELLM Assistant")
    st.markdown("*軽量LLMによる対話型解析アドバイス*")

    # LLM利用可能性確誁E
    st.info(
        "💡 **ヒンチE*: フルLLM機�Eを使用するには `pip install gpt4all` が忁E��です、E
        "未インスト�Eルの場合�Eルールベ�Eスの簡易アドバイスが返されます、E
    )

    # サブタチE
    assistant_tabs = st.tabs([
        "📊 特徴量選択アドバイス",
        "📋 解析�Eラン提桁E,
        "🎯 結果解釁E,
        "❁E自由Q&A"
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
    st.markdown("チE�EタセチE��惁E��から、最適な特徴量セチE��を推奨します、E)

    with st.form("feature_suggest_form"):
        col1, col2 = st.columns(2)

        with col1:
            n_samples = st.number_input(
                "サンプル数", min_value=10, max_value=100000, value=500, step=10
            )
            task_type = st.selectbox("タスクタイチE, ["regression", "classification"])

        with col2:
            target_property = st.text_input(
                "予測対象の物性", value="solubility (logS)", placeholder="e.g., LogP, Tg"
            )

        submitted = st.form_submit_button("🤁Eアドバイスを取征E, use_container_width=True)

        if submitted:
            with st.spinner("LLMが老E��中..."):
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

                        st.success("✁Eアドバイスを取得しました")

                        # 推奨特徴釁E
                        st.markdown("### 🎯 推奨特徴釁E)
                        for feat in result["recommended_features"]:
                            st.markdown(f"- **{feat}**")

                        # 琁E��
                        st.markdown("### 💡 琁E��")
                        st.write(result["reasoning"])

                        # 代替桁E
                        if result["alternative_features"]:
                            st.markdown("### 🔄 代替オプション")
                            for feat in result["alternative_features"]:
                                st.markdown(f"- {feat}")

                        # 老E�E事頁E
                        with st.expander("📝 老E�E事頁E):
                            for consideration in result["considerations"]:
                                st.write(f"• {consideration}")

                    else:
                        st.error(f"エラー: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"リクエストエラー: {e}")


def _render_analysis_plan():
    """解析�Eラン提桁E""
    st.subheader("📋 解析�Eラン提桁E)
    st.markdown("問題記述から、解析戦略を提案します、E)

    with st.form("analysis_plan_form"):
        problem_description = st.text_area(
            "問題�E説昁E,
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

        submitted = st.form_submit_button("💡 プランを提桁E, use_container_width=True)

        if submitted:
            with st.spinner("解析�Eランを作�E中..."):
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

                        st.success("✁E解析�Eランを作�Eしました")

                        # 目皁E
                        st.markdown("### 🎯 目皁E)
                        st.write(result["objective"])

                        # 推奨アプローチE
                        st.markdown("### 🧭 推奨アプローチE)
                        st.write(result["recommended_approach"])

                        # モチE��候裁E
                        st.markdown("### 🤁E推奨モチE��")
                        model_cols = st.columns(len(result["model_suggestions"]))
                        for i, model in enumerate(result["model_suggestions"]):
                            model_cols[i].info(model)

                        # 検証戦略
                        st.markdown("### ✁E検証戦略")
                        st.write(result["validation_strategy"])

                        # 課顁E
                        with st.expander("⚠�E�E想定される課顁E):
                            for challenge in result["potential_challenges"]:
                                st.write(f"• {challenge}")

                    else:
                        st.error(f"エラー: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"リクエストエラー: {e}")


def _render_result_interpretation():
    """結果解釁E""
    st.subheader("🎯 モチE��結果の解釁E)
    st.markdown("評価持E��から、結果の解釈と改喁E��を提案します、E)

    with st.form("interpret_form"):
        st.markdown("#### 評価持E��を入劁E)

        col1, col2 = st.columns(2)

        with col1:
            r2 = st.number_input("R² Score", min_value=-1.0, max_value=1.0, value=0.85, step=0.01)
            mae = st.number_input("MAE", min_value=0.0, value=0.42, step=0.01)

        with col2:
            rmse = st.number_input("RMSE", min_value=0.0, value=0.58, step=0.01)
            model_type = st.text_input("モチE��タイチE, value="XGBoost")

        submitted = st.form_submit_button("🔍 結果を解釁E, use_container_width=True)

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

                        st.success("✁E解釈が完亁E��ました")

                        # 解釁E
                        st.markdown("### 💭 解釁E)
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
    st.subheader("❁E自由Q&A")
    st.markdown("化学機械学習に関する質問に回答します、E)

    # サンプル質啁E
    sample_questions = [
        "Morgan fingerprintsはぁE��使ぁE��きですか�E�E,
        "小さぁE��ータセチE���E�E100サンプル�E�で過学習を避けるには�E�E,
        "XGBoostとLightGBMの違いは�E�E,
        "SHAP値の解釈方法�E�E�E,
    ]

    selected_sample = st.selectbox(
        "サンプル質問（また�E下に自由入力！E,
        ["-- 自由入劁E--"] + sample_questions
    )

    if selected_sample != "-- 自由入劁E--":
        question = selected_sample
    else:
        question = st.text_area(
            "質問を入劁E,
            height=100,
            placeholder="侁E バッチ正規化とは何ですか�E�E,
        )

    context = st.text_input("コンチE��スト（オプション�E�E, placeholder="e.g., I'm working on QSAR modeling")

    if st.button("🤁E質問すめE, use_container_width=True):
        if not question:
            st.warning("質問を入力してください")
            return

        with st.spinner("老E��中..."):
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

                    st.success("✁E回答が完亁E��ました")

                    # 質啁E
                    st.markdown("### ❁E質啁E)
                    st.info(result["question"])

                    # 回筁E
                    st.markdown("### 💡 回筁E)
                    st.write(result["answer"])

                    # LLM利用状況E
                    if result.get("llm_available"):
                        st.caption("✨ GPT4All (Full LLM) を使用")
                    else:
                        st.caption("📋 ルールベ�Eスモード（簡易回答！E)

                else:
                    st.error(f"エラー: {res.status_code} - {res.text}")

            except Exception as e:
                st.error(f"リクエストエラー: {e}")
