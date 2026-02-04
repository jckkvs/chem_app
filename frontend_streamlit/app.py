"""
Chemical ML Platform - Streamlit Frontend

Implements: F-UI-001
險ｭ險域晄Φ:
- 蛻晏ｿ��↓縺ｯ繧ｷ繝ｳ繝励Ν縲∫�邱ｴ閠�↓縺ｯ隧ｳ邏ｰ險ｭ螳�
- 繝励Ο繧ｰ繝ｬ繝�す繝夜幕遉ｺUI
- 繝ｪ繧｢繝ｫ繧ｿ繧､繝�蜿ｯ隕門喧
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import base64
from typing import Optional, Dict, Any

# TARTE蛻ｩ逕ｨ蜿ｯ閭ｽ諤ｧ繝√ぉ繝�け�磯≦蟒ｶ繧､繝ｳ繝昴�繝茨ｼ�
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

# プロキシ設定モジュール
try:
    from proxy_settings import render_proxy_settings
except ImportError:
    try:
        from frontend_streamlit.proxy_settings import render_proxy_settings
    except ImportError:
        def render_proxy_settings():
            st.error("Proxy settings module not found")

# 時系列UIモジュール
try:
    from timeseries_ui import render_timeseries_settings
except ImportError:
    try:
        from frontend_streamlit.timeseries_ui import render_timeseries_settings
    except ImportError:
        def render_timeseries_settings():
            st.error("Time series UI module not found")

# EDAダッシュボードモジュール
try:
    from eda_dashboard_ui import render_eda_dashboard
except ImportError:
    try:
        from frontend_streamlit.eda_dashboard_ui import render_eda_dashboard
    except ImportError:
        def render_eda_dashboard():
            st.error("EDA dashboard module not found")


# API險ｭ螳
API_URL = "http://127.0.0.1:8000/api"

# 繝壹�繧ｸ險ｭ螳�
st.set_page_config(
    page_title="ChemML Platform",
    page_icon="�ｧｪ",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """繝｡繧､繝ｳ繧ｨ繝ｳ繝医Μ繝ｼ繝昴う繝ｳ繝�"""
    st.title("�ｧｪ Chemical ML Platform")
    st.markdown("*讖滓｢ｰ蟄ｦ鄙偵ｒ菴ｿ縺｣縺溷�蟄千黄諤ｧ莠域ｸｬ繝励Λ繝�ヨ繝輔か繝ｼ繝�*")
    
    # 繧ｵ繧､繝峨ヰ繝ｼ
    with st.sidebar:
        st.header("笞呻ｸ� Settings")
        show_advanced = st.checkbox("隧ｳ邏ｰ險ｭ螳壹ｒ陦ｨ遉ｺ", value=False)
        auto_refresh = st.checkbox("閾ｪ蜍墓峩譁ｰ (10遘�)", value=False)
        
        # TARTE險ｭ螳�
        with st.expander("�､� TARTE Settings", expanded=False):
            tarte_available = _check_tarte_available()
            if tarte_available:
                st.success("笨� tarte-ai 繧､繝ｳ繧ｹ繝医�繝ｫ貂医∩")
                use_tarte = st.checkbox("TARTE繧剃ｽｿ逕ｨ", value=False, key="use_tarte_global")
                if use_tarte:
                    tarte_mode = st.radio(
                        "繝｢繝ｼ繝蛾∈謚�",
                        ["Featurizer�磯ｫ倬滂ｼ�", "Finetuning�磯ｫ倡ｲｾ蠎ｦ��", "Boosting�域怙鬮倡ｲｾ蠎ｦ��"],
                        help="""
                        **Featurizer**: 莠句燕蟄ｦ鄙偵Δ繝�Ν縺ｧ蝓九ａ霎ｼ縺ｿ逕滓��域耳螂ｨ��
                        **Finetuning**: 繝��繧ｿ縺ｫ蜷医ｏ縺帙※繝｢繝�Ν隱ｿ謨ｴ
                        **Boosting**: 繧｢繝ｳ繧ｵ繝ｳ繝悶Ν蟄ｦ鄙抵ｼ域怙繧よ凾髢薙′縺九°繧具ｼ�
                        """,
                        key="tarte_mode_global"
                    )
                    # 繧ｻ繝�す繝ｧ繝ｳ迥ｶ諷九↓菫晏ｭ�
                    mode_map = {
                        "Featurizer�磯ｫ倬滂ｼ�": "featurizer",
                        "Finetuning�磯ｫ倡ｲｾ蠎ｦ��": "finetuning",
                        "Boosting�域怙鬮倡ｲｾ蠎ｦ��": "boosting",
                    }
                    st.session_state["tarte_mode"] = mode_map.get(tarte_mode, "featurizer")
            else:
                st.warning("笞��� tarte-ai縺後う繝ｳ繧ｹ繝医�繝ｫ縺輔ｌ縺ｦ縺�∪縺帙ｓ")
                st.markdown("陦ｨ蠖｢蠑上ョ繝ｼ繧ｿ縺ｮTransformer迚ｹ蠕ｴ驥上ｒ菴ｿ逕ｨ縺吶ｋ縺ｫ縺ｯ繧､繝ｳ繧ｹ繝医�繝ｫ縺悟ｿ�ｦ√〒縺�:")
                st.warning("笞 tarte-ai縺後う繝ｳ繧ｹ繝医繝ｫ縺輔ｌ縺ｦ縺∪縺帙ｓ")
                st.markdown("陦ｨ蠖｢蠑上ョ繝ｼ繧ｿ縺ｮTransformer迚ｹ蠕ｴ驥上ｒ菴ｿ逕ｨ縺吶ｋ縺ｫ縺ｯ繧､繝ｳ繧ｹ繝医繝ｫ縺悟ｿｦ√〒縺:")
                st.code("pip install tarte-ai", language="bash")
                st.caption("[答 TARTE Documentation](https://github.com/soda-inria/tarte-ai)")
        
        st.divider()
        st.caption("Version 2.2 - TARTE Enhanced")
    
    # 繝｡繧､繝ｳ繧ｿ繝
    tabs = st.tabs([
        "唐 Datasets",
        "笞暦ｸ Experiments",
        "投 Analysis",
        "溌 Molecule Viewer",
        "逃 Batch Predict",
        "笞厄ｸ Comparison",
        "､ LLM Assistant",
        "🌐 Proxy Settings",
        "⏰ Time Series",
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
    with tabs[6]:
        render_llm_assistant()
    with tabs[7]:
        render_proxy_settings()
    with tabs[8]:
        st.subheader("⏰ 時系列データ分析")
        st.info("""
        **時系列分析機能**
        
        Prophet、ARIMA、SARIMAなどのモデルを使用して、
        時間的な順序を持つデータの予測・分析が可能です。
        
        データセットから日時カラムを自動検出し、
        最適なモデル設定を提案します。
        """)
        
        # データセット選択（簡易版）
        try:
            response = requests.get(f"{API_URL}/datasets/")
            if response.status_code == 200:
                datasets = response.json()
                if datasets:
                    dataset_names = [ds.get('name', 'Unknown') for ds in datasets]
                    selected_dataset = st.selectbox("データセット選択", dataset_names)
                    
                    if st.button("⏰ 時系列分析を開始", type="primary"):
                        st.success("✅ 詳細な時系列設定はDjango Web UIをご利用ください")
                        st.markdown("[Django時系列分析ページへ](/timeseries)")
                else:
                    st.warning("データセットがありません。まずデータをアップロードしてください。")
        except Exception as e:
            st.error(f"データセット読み込みエラー: {e}")

    # 閾ｪ蜍墓峩譁ｰ
    if auto_refresh:
        time.sleep(10)
        st.rerun()



def render_datasets():
    """繝��繧ｿ繧ｻ繝�ヨ邂｡逅��繝ｼ繧ｸ"""
    st.header("�唐 Dataset Management")
    
    # 繧｢繝��繝ｭ繝ｼ繝峨ヵ繧ｩ繝ｼ繝�
    with st.expander("�豆 Upload New Dataset", expanded=False):
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
    
    # 繝��繧ｿ繧ｻ繝�ヨ荳隕ｧ
    st.subheader("�搭 Available Datasets")
    _display_datasets()


def _upload_dataset(name: str, file, smiles_col: str, target_col: str):
    """繝��繧ｿ繧ｻ繝�ヨ繧偵い繝��繝ｭ繝ｼ繝�"""
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
            st.success(f"笨� Dataset '{name}' uploaded successfully!")
            st.rerun()
        else:
            st.error(f"Error: {res.text}")
    except requests.exceptions.ConnectionError:
        st.error("笞��� Cannot connect to API server")
    except Exception as e:
        st.error(f"Error: {e}")


def _display_datasets():
    """繝��繧ｿ繧ｻ繝�ヨ荳隕ｧ繧定｡ｨ遉ｺ"""
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
        st.warning("笞��� API server not available")
    except Exception as e:
        st.warning(f"Could not load datasets: {e}")


def render_experiments(show_advanced: bool = False):
    """螳滄ｨ薙そ繝�ヨ繧｢繝��繝壹�繧ｸ"""
    st.header("笞暦ｸ� Experiment Setup")
    
    # 繧ｿ繧ｹ繧ｯ繧ｿ繧､繝鈴∈謚�
    st.subheader("�搭 繧ｿ繧ｹ繧ｯ繧ｿ繧､繝�")
    task_type = st.radio(
        "莠域ｸｬ繧ｿ繧ｹ繧ｯ繧帝∈謚�",
        [
            "竭� SMILES 竊� 迚ｩ諤ｧ莠域ｸｬ",
            "竭｡ 陦ｨ繝��繧ｿ 竊� 迚ｹ諤ｧ莠域ｸｬ", 
            "竭｢ 豺ｷ蜷育黄��MILES�句牡蜷茨ｼ� 竊� 迚ｩ諤ｧ莠域ｸｬ",
            "竭｣ SMILES�玖｡ｨ繝��繧ｿ 竊� 迚ｩ諤ｧ莠域ｸｬ",
        ],
        horizontal=True,
        help="繝��繧ｿ縺ｮ蠖｢蠑上↓蠢懊§縺ｦ驕ｸ謚槭＠縺ｦ縺上□縺輔＞"
    )
    
    # 繧ｿ繧ｹ繧ｯ繧ｿ繧､繝励ｒconfig逕ｨ縺ｫ螟画鋤
    task_type_map = {
        "竭� SMILES 竊� 迚ｩ諤ｧ莠域ｸｬ": "smiles_only",
        "竭｡ 陦ｨ繝��繧ｿ 竊� 迚ｹ諤ｧ莠域ｸｬ": "tabular_only",
        "竭｢ 豺ｷ蜷育黄��MILES�句牡蜷茨ｼ� 竊� 迚ｩ諤ｧ莠域ｸｬ": "mixture",
        "竭｣ SMILES�玖｡ｨ繝��繧ｿ 竊� 迚ｩ諤ｧ莠域ｸｬ": "smiles_tabular",
    }
    selected_task_type = task_type_map.get(task_type, "smiles_only")
    
    st.divider()
    
    # 螳滄ｨ謎ｽ懈�繝輔か繝ｼ繝�
    with st.form("exp_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            exp_name = st.text_input("Experiment Name", placeholder="e.g., LogP Prediction v1")
            
            # 繝��繧ｿ繧ｻ繝�ヨ驕ｸ謚�
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
            
            # 繧ｿ繧ｹ繧ｯ繧ｿ繧､繝励↓蠢懊§縺溽音蠕ｴ驥城∈謚�
            if selected_task_type == "tabular_only":
                features = ["tabular"]
                st.info("�投 陦ｨ繝��繧ｿ繝｢繝ｼ繝会ｼ壹ョ繝ｼ繧ｿ繧ｻ繝�ヨ縺ｮ謨ｰ蛟､繧ｫ繝ｩ繝�繧剃ｽｿ逕ｨ")
            elif selected_task_type == "mixture":
                features = ["mixture"]
                st.info("�ｧｪ 豺ｷ蜷育黄繝｢繝ｼ繝会ｼ售MILES + 蜑ｲ蜷医°繧牙刈驥榊ｹｳ蝮�ｨ倩ｿｰ蟄舌ｒ險育ｮ�")
            else:
                # SMILES邉ｻ繧ｿ繧ｹ繧ｯ: 騾壼ｸｸ縺ｮ迚ｹ蠕ｴ驥城∈謚� + TARTE蟇ｾ蠢�
                available_features = ["rdkit", "xtb", "uma"]
                feature_help = "rdkit: Molecular descriptors, xtb: Quantum properties, uma: UMAP embeddings"
                
                # TARTE縺梧怏蜉ｹ縺ｪ蝣ｴ蜷医�霑ｽ蜉��郁｡ｨ繝��繧ｿ豺ｷ蜷医ち繧ｹ繧ｯ蜷代￠��
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
        
        # 迚ｩ諤ｧ繝励Μ繧ｻ繝�ヨ驕ｸ謚橸ｼ�mart Feature Engineering��
        st.divider()
        st.subheader("�識 逶ｮ逧�黄諤ｧ��mart Feature Selection��")
        
        # 迚ｩ諤ｧ繧ｫ繝�ざ繝ｪ蛻･縺ｮ繝励Μ繧ｻ繝�ヨ
        property_options = {
            "-- 豎守畑 --": {
                "general": "豎守畑�医ヰ繝ｩ繝ｳ繧ｹ蝙具ｼ�",
            },
            "-- 蜈牙ｭｦ迚ｩ諤ｧ --": {
                "refractive_index": "螻域釜邇�",
                "optical_gap": "蜈牙ｭｦ繝舌Φ繝峨ぐ繝｣繝��",
            },
            "-- 讖滓｢ｰ迚ｩ諤ｧ --": {
                "elastic_modulus": "蠑ｾ諤ｧ邇��繝､繝ｳ繧ｰ邇�",
                "tensile_strength": "蠑募ｼｵ蠑ｷ蠎ｦ",
                "hardness": "遑ｬ蠎ｦ",
            },
            "-- 辭ｱ迚ｩ諤ｧ --": {
                "glass_transition": "繧ｬ繝ｩ繧ｹ霆｢遘ｻ貂ｩ蠎ｦ(Tg)",
                "melting_point": "陞咲せ",
                "thermal_conductivity": "辭ｱ莨晏ｰ守紫",
                "thermal_stability": "辭ｱ螳牙ｮ壽ｧ",
            },
            "-- 髮ｻ豌礼黄諤ｧ --": {
                "dielectric_constant": "隱倬崕邇�",
                "conductivity": "髮ｻ豌嶺ｼ晏ｰ主ｺｦ",
            },
            "-- 蛹門ｭｦ迚ｩ諤ｧ --": {
                "solubility": "貅ｶ隗｣蠎ｦ繝ｻLogP",
                "viscosity": "邊伜ｺｦ",
                "density": "蟇�ｺｦ",
            },
            "-- 霈ｸ騾∫黄諤ｧ --": {
                "gas_permeability": "繧ｬ繧ｹ騾城℃諤ｧ",
            },
            "-- 阮ｬ逅�ｭｦ --": {
                "admet": "ADMET繝ｻ阮ｬ迚ｩ蜍墓�",
                "pka": "pKa",
            },
        }
        
        # 繝輔Λ繝�ヨ縺ｪ繝ｪ繧ｹ繝医↓螟画鋤�郁｡ｨ遉ｺ逕ｨ��
        flat_options = []
        option_to_key = {}
        for category, presets in property_options.items():
            flat_options.append(category)
            for key, name in presets.items():
                display = f"  {name}"
                flat_options.append(display)
                option_to_key[display] = key
        
        selected_property_display = st.selectbox(
            "逶ｮ逧�黄諤ｧ繧帝∈謚橸ｼ域怙驕ｩ縺ｪ險倩ｿｰ蟄舌そ繝�ヨ繧定�蜍暮∈謚橸ｼ�",
            flat_options,
            index=1,  # "豎守畑" 繧偵ョ繝輔か繝ｫ繝�
            help="莠域ｸｬ縺励◆縺�黄諤ｧ縺ｫ蠢懊§縺ｦ縲∵怙驕ｩ縺ｪ蛻�ｭ占ｨ倩ｿｰ蟄舌そ繝�ヨ縺瑚�蜍暮∈謚槭＆繧後∪縺�"
        )
        
        # 繧ｫ繝�ざ繝ｪ繝倥ャ繝繝ｼ縺ｯ驕ｸ謚槭〒縺阪↑縺�
        if selected_property_display.startswith("--"):
            target_property = "general"
        else:
            target_property = option_to_key.get(selected_property_display, "general")
        
        # 隧ｳ邏ｰ險ｭ螳夲ｼ医�繝ｭ繧ｰ繝ｬ繝�す繝夜幕遉ｺ��
        if show_advanced:
            st.divider()
            st.subheader("�肌 Advanced Settings")
            
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
                    "SmartFeatureEngine菴ｿ逕ｨ",
                    value=True,
                    help="迚ｩ諤ｧﾃ励ョ繝ｼ繧ｿ繧ｻ繝�ヨ迚ｹ諤ｧ縺ｫ蝓ｺ縺･縺乗怙驕ｩ迚ｹ蠕ｴ驥城∈謚�"
                )
            
            # 莠句燕蟄ｦ鄙偵Δ繝�Ν驕ｸ謚�
            st.markdown("**�､� 莠句燕蟄ｦ鄙偵Δ繝�Ν�医が繝励す繝ｧ繝ｳ��**")
            pretrained_col1, pretrained_col2 = st.columns(2)
            with pretrained_col1:
                use_unimol = st.checkbox("Uni-Mol��3D讒矩���", value=False)
            with pretrained_col2:
                use_chemberta = st.checkbox("ChemBERTa��MILES��", value=False)
            
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
        
        submitted = st.form_submit_button("�噫 Start Experiment", use_container_width=True)
        
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
    
    # 螳滄ｨ謎ｸ隕ｧ
    st.subheader("�搭 Recent Experiments")
    _display_experiments()


def _fetch_dataset_options() -> Dict[str, int]:
    """繝��繧ｿ繧ｻ繝�ヨ繧ｪ繝励す繝ｧ繝ｳ繧貞叙蠕�"""
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
    螳滄ｨ薙ｒ髢句ｧ�
    
    Args:
        dataset_id: 繝��繧ｿ繧ｻ繝�ヨID
        name: 螳滄ｨ灘錐
        features: 菴ｿ逕ｨ縺吶ｋ迚ｹ蠕ｴ驥上Μ繧ｹ繝�
        model_type: 繝｢繝�Ν繧ｿ繧､繝�
        cv_folds: CV蛻�牡謨ｰ
        task_type: 繧ｿ繧ｹ繧ｯ繧ｿ繧､繝暦ｼ�egression/classification��
        task_type_mode: 繝��繧ｿ繧ｿ繧､繝暦ｼ�miles_only/tabular_only/mixture/smiles_tabular��
        target_property: 逶ｮ逧�黄諤ｧ繝励Μ繧ｻ繝�ヨ
        use_smart_engine: SmartFeatureEngine菴ｿ逕ｨ繝輔Λ繧ｰ
        pretrained_models: 菴ｿ逕ｨ縺吶ｋ莠句燕蟄ｦ鄙偵Δ繝�Ν繝ｪ繧ｹ繝�
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
            st.success(f"笨� Experiment started! ID: {exp_data['id']}")
            
            # 險ｭ螳壹し繝槭Μ繝ｼ陦ｨ遉ｺ
            with st.expander("�搭 Experiment Settings", expanded=False):
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
    """螳滄ｨ謎ｸ隕ｧ繧定｡ｨ遉ｺ"""
    try:
        res = requests.get(f"{API_URL}/experiments", timeout=10)
        if res.status_code == 200:
            experiments = res.json()
            if experiments:
                df = pd.DataFrame(experiments)
                # 繧ｹ繝��繧ｿ繧ｹ縺ｫ濶ｲ繧剃ｻ倥￠繧�
                st.dataframe(df, use_container_width=True)
    except Exception:
        pass


def render_analysis():
    """邨先棡蛻�梵繝壹�繧ｸ"""
    st.header("�投 Results Analysis")
    
    # 螳滄ｨ馴∈謚�
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
        
        # 繧ｹ繝��繧ｿ繧ｹ陦ｨ遉ｺ
        status = exp['status']
        if status == 'COMPLETED':
            st.success(f"Status: 笨� {status}")
        elif status == 'RUNNING':
            st.warning(f"Status: 竢ｳ {status}")
        elif status == 'FAILED':
            st.error(f"Status: 笶� {status}")
        else:
            st.info(f"Status: {status}")
        
        # 繝｡繝医Μ繧ｯ繧ｹ
        if exp.get('metrics'):
            st.subheader("�嶋 Validation Metrics")
            metrics = exp['metrics']
            
            # 繧ｨ繝ｩ繝ｼ縺後≠繧句�ｴ蜷�
            if 'error' in metrics:
                st.error(f"Error: {metrics['error']}")
            else:
                # 繝｡繝医Μ繧ｯ繧ｹ繧偵き繝ｩ繝�縺ｧ陦ｨ遉ｺ
                cols = st.columns(4)
                for i, (k, v) in enumerate(metrics.items()):
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        cols[i % 4].metric(k, f"{v:.4f}")
        
        # 螳御ｺ�＠縺溷ｮ滄ｨ薙�隧ｳ邏ｰ
        if status == 'COMPLETED':
            st.divider()
            _render_completed_analysis(exp_id, exp)
            
    except Exception as e:
        st.error(f"Error: {e}")


def _fetch_experiment_options() -> Dict[str, int]:
    """螳滄ｨ薙が繝励す繝ｧ繝ｳ繧貞叙蠕�"""
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
    """螳御ｺ�＠縺溷ｮ滄ｨ薙�蛻�梵繧定｡ｨ遉ｺ"""
    # 蜿ｯ隕門喧
    st.subheader("�耳 Global Explanations")
    
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
    
    # 繧､繝ｳ繧ｿ繝ｩ繧ｯ繝�ぅ繝紋ｺ域ｸｬ
    st.divider()
    st.subheader("�醗 Interactive Prediction")
    
    smi_input = st.text_input("Enter SMILES", value="c1ccccc1", placeholder="e.g., CCO, c1ccccc1")
    
    if st.button("Predict", use_container_width=True):
        _run_prediction(exp_id, smi_input)


def _display_artifact(exp_id: int, filename: str):
    """繧｢繝ｼ繝�ぅ繝輔ぃ繧ｯ繝医ｒ陦ｨ遉ｺ"""
    try:
        res = requests.get(f"{API_URL}/experiments/{exp_id}/artifacts/{filename}", timeout=10)
        if res.status_code == 200:
            st.image(res.content, use_container_width=True)
        else:
            st.info("Not available")
    except Exception:
        st.info("Not available")


def _run_prediction(exp_id: int, smiles: str):
    """莠域ｸｬ繧貞ｮ溯｡�"""
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
    """蛻�ｭ舌ン繝･繝ｼ繝ｯ繝ｼ繝壹�繧ｸ"""
    st.header("�溌 Molecule Viewer")
    st.markdown("SMILES縺九ｉ蛻�ｭ先ｧ矩�縺ｨ迚ｩ諤ｧ繧定｡ｨ遉ｺ")
    
    smiles_input = st.text_input("SMILES蜈･蜉�", value="c1ccccc1", placeholder="萓�: CCO, CC(=O)O")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if smiles_input:
            st.subheader("蛻�ｭ先ｧ矩�")
            try:
                # API縺九ｉ蛻�ｭ心VG繧貞叙蠕�
                svg_res = requests.get(f"{API_URL}/molecules/{smiles_input}/svg", timeout=10)
                if svg_res.status_code == 200:
                    st.image(svg_res.content, use_container_width=True)
                else:
                    st.warning("蛻�ｭ舌�謠冗判縺ｫ螟ｱ謨励＠縺ｾ縺励◆")
            except Exception as e:
                st.error(f"SVG蜿門ｾ励お繝ｩ繝ｼ: {e}")
    
    with col2:
        if smiles_input:
            st.subheader("迚ｩ諤ｧ諠��ｱ")
            try:
                props_res = requests.get(f"{API_URL}/molecules/{smiles_input}/properties", timeout=10)
                if props_res.status_code == 200:
                    props = props_res.json()
                    st.metric("蛻�ｭ宣㍼", f"{props['molecular_weight']:.2f} g/mol")
                    st.metric("LogP", f"{props['logp']:.2f}")
                    st.metric("TPSA", f"{props['tpsa']:.2f} ﾃ�ｲ")
                    
                    with st.expander("隧ｳ邏ｰ諠��ｱ"):
                        st.write(f"豌ｴ邏�邨仙粋繝峨リ繝ｼ: {props['hbd']}")
                        st.write(f"豌ｴ邏�邨仙粋繧｢繧ｯ繧ｻ繝励ち繝ｼ: {props['hba']}")
                        st.write(f"蝗櫁ｻ｢蜿ｯ閭ｽ邨仙粋: {props['rotatable_bonds']}")
                        st.write(f"迺ｰ縺ｮ謨ｰ: {props['num_rings']}")
                        st.write(f"蜴溷ｭ先焚: {props['num_atoms']}")
                else:
                    st.warning("迚ｩ諤ｧ險育ｮ励↓螟ｱ謨励＠縺ｾ縺励◆")
            except Exception as e:
                st.error(f"迚ｩ諤ｧ蜿門ｾ励お繝ｩ繝ｼ: {e}")
    
    # SMILES讀懆ｨｼ
    st.divider()
    st.subheader("SMILES讀懆ｨｼ")
    validate_smiles = st.text_input("讀懆ｨｼ縺吶ｋSMILES", key="validate_smiles")
    if st.button("讀懆ｨｼ"):
        try:
            res = requests.post(f"{API_URL}/molecules/validate", json={"smiles": validate_smiles}, timeout=10)
            data = res.json()
            if data['valid']:
                st.success(f"笨� 譛牙柑縺ｪSMILES - 豁｣隕丞喧: `{data['canonical_smiles']}`")
            else:
                st.error(f"笶� 辟｡蜉ｹ縺ｪSMILES: {data['error']}")
        except Exception as e:
            st.error(f"讀懆ｨｼ繧ｨ繝ｩ繝ｼ: {e}")


def render_batch_predict():
    """繝舌ャ繝∽ｺ域ｸｬ繝壹�繧ｸ"""
    st.header("�逃 繝舌ャ繝∽ｺ域ｸｬ")
    st.markdown("隍�焚縺ｮSMILES縺ｫ蟇ｾ縺励※荳諡ｬ縺ｧ莠域ｸｬ繧貞ｮ溯｡�")
    
    # 螳滄ｨ馴∈謚�
    exp_options = _fetch_experiment_options()
    completed_options = {k: v for k, v in exp_options.items() if "COMPLETED" in k}
    
    if not completed_options:
        st.warning("螳御ｺ�＠縺溷ｮ滄ｨ薙′縺ゅｊ縺ｾ縺帙ｓ縲ゅ∪縺壼ｮ滄ｨ薙ｒ菴懈�繝ｻ螳御ｺ�＆縺帙※縺上□縺輔＞縲�")
        return
    
    selected_exp = st.selectbox("螳滄ｨ薙ｒ驕ｸ謚�", list(completed_options.keys()))
    exp_id = completed_options.get(selected_exp)
    
    # 蜈･蜉帶婿蠑城∈謚�
    input_method = st.radio("蜈･蜉帶婿豕�", ["繝�く繧ｹ繝亥�蜉�", "CSV繧｢繝��繝ｭ繝ｼ繝�"], horizontal=True)
    
    smiles_list = []
    
    if input_method == "繝�く繧ｹ繝亥�蜉�":
        smiles_text = st.text_area(
            "SMILES繝ｪ繧ｹ繝茨ｼ�1陦後↓1縺､��",
            value="CCO\nc1ccccc1\nCC(=O)O\nCCCCC",
            height=200
        )
        smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
    else:
        uploaded = st.file_uploader("CSV繝輔ぃ繧､繝ｫ��MILES繧ｫ繝ｩ繝�繧貞性繧��", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            smiles_col = st.selectbox("SMILES繧ｫ繝ｩ繝�", df.columns.tolist())
            smiles_list = df[smiles_col].dropna().tolist()
            st.info(f"{len(smiles_list)}莉ｶ縺ｮSMILES繧定ｪｭ縺ｿ霎ｼ縺ｿ縺ｾ縺励◆")
    
    if smiles_list and st.button("�噫 繝舌ャ繝∽ｺ域ｸｬ螳溯｡�", use_container_width=True):
        with st.spinner(f"{len(smiles_list)}莉ｶ縺ｮSMILES繧貞�逅�ｸｭ..."):
            try:
                res = requests.post(
                    f"{API_URL}/experiments/{exp_id}/batch_predict",
                    json={"smiles_list": smiles_list},
                    timeout=120
                )
                
                if res.status_code == 200:
                    data = res.json()
                    predictions = data['predictions']
                    
                    st.success(f"笨� {len(predictions)}莉ｶ縺ｮ莠域ｸｬ縺悟ｮ御ｺ�＠縺ｾ縺励◆")
                    
                    # 邨先棡陦ｨ遉ｺ
                    result_df = pd.DataFrame(predictions)
                    st.dataframe(result_df, use_container_width=True)
                    
                    # CSV蜃ｺ蜉�
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "�踏 邨先棡繧辰SV縺ｧ繝繧ｦ繝ｳ繝ｭ繝ｼ繝�",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
                    # 邁｡譏鍋ｵｱ險�
                    if 'prediction' in result_df.columns:
                        st.subheader("�投 邨ｱ險域ュ蝣ｱ")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("蟷ｳ蝮�", f"{result_df['prediction'].mean():.4f}")
                        col2.metric("讓呎ｺ門￥蟾ｮ", f"{result_df['prediction'].std():.4f}")
                        col3.metric("譛蟆�", f"{result_df['prediction'].min():.4f}")
                        col4.metric("譛螟ｧ", f"{result_df['prediction'].max():.4f}")
                else:
                    st.error(f"繝舌ャ繝∽ｺ域ｸｬ螟ｱ謨�: {res.text}")
                    
            except Exception as e:
                st.error(f"繧ｨ繝ｩ繝ｼ: {e}")


if __name__ == "__main__":
    main()

"""
LLM Assistant UI - Append to frontend_streamlit/app.py
"""


def render_llm_assistant():
    """LLM繧｢繧ｷ繧ｹ繧ｿ繝ｳ繝医・繝ｼ繧ｸ"""
    st.header("�､・LLM Assistant")
    st.markdown("*霆ｽ驥臭LM縺ｫ繧医ｋ蟇ｾ隧ｱ蝙玖ｧ｣譫舌い繝峨ヰ繧､繧ｹ*")

    # LLM蛻ｩ逕ｨ蜿ｯ閭ｽ諤ｧ遒ｺ隱・
    st.info(
        "�庁 **繝偵Φ繝・*: 繝輔ΝLLM讖溯・繧剃ｽｿ逕ｨ縺吶ｋ縺ｫ縺ｯ `pip install gpt4all` 縺悟ｿ・ｦ√〒縺吶・
        "譛ｪ繧､繝ｳ繧ｹ繝医・繝ｫ縺ｮ蝣ｴ蜷医・繝ｫ繝ｼ繝ｫ繝吶・繧ｹ縺ｮ邁｡譏薙い繝峨ヰ繧､繧ｹ縺瑚ｿ斐＆繧後∪縺吶・
    )

    # 繧ｵ繝悶ち繝・
    assistant_tabs = st.tabs([
        "�投 迚ｹ蠕ｴ驥城∈謚槭い繝峨ヰ繧､繧ｹ",
        "�搭 隗｣譫舌・繝ｩ繝ｳ謠先｡・,
        "�識 邨先棡隗｣驥・,
        "笶・閾ｪ逕ｱQ&A"
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
    """迚ｹ蠕ｴ驥城∈謚槭い繝峨ヰ繧､繧ｹ"""
    st.subheader("�投 迚ｹ蠕ｴ驥城∈謚槭い繝峨ヰ繧､繧ｹ")
    st.markdown("繝・・繧ｿ繧ｻ繝・ヨ諠・�ｱ縺九ｉ縲∵怙驕ｩ縺ｪ迚ｹ蠕ｴ驥上そ繝・ヨ繧呈耳螂ｨ縺励∪縺吶・)

    with st.form("feature_suggest_form"):
        col1, col2 = st.columns(2)

        with col1:
            n_samples = st.number_input(
                "繧ｵ繝ｳ繝励Ν謨ｰ", min_value=10, max_value=100000, value=500, step=10
            )
            task_type = st.selectbox("繧ｿ繧ｹ繧ｯ繧ｿ繧､繝・, ["regression", "classification"])

        with col2:
            target_property = st.text_input(
                "莠域ｸｬ蟇ｾ雎｡縺ｮ迚ｩ諤ｧ", value="solubility (logS)", placeholder="e.g., LogP, Tg"
            )

        submitted = st.form_submit_button("�､・繧｢繝峨ヰ繧､繧ｹ繧貞叙蠕・, use_container_width=True)

        if submitted:
            with st.spinner("LLM縺瑚・∴荳ｭ..."):
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

                        st.success("笨・繧｢繝峨ヰ繧､繧ｹ繧貞叙蠕励＠縺ｾ縺励◆")

                        # 謗ｨ螂ｨ迚ｹ蠕ｴ驥・
                        st.markdown("### �識 謗ｨ螂ｨ迚ｹ蠕ｴ驥・)
                        for feat in result["recommended_features"]:
                            st.markdown(f"- **{feat}**")

                        # 逅・罰
                        st.markdown("### �庁 逅・罰")
                        st.write(result["reasoning"])

                        # 莉｣譖ｿ譯・
                        if result["alternative_features"]:
                            st.markdown("### �売 莉｣譖ｿ繧ｪ繝励す繝ｧ繝ｳ")
                            for feat in result["alternative_features"]:
                                st.markdown(f"- {feat}")

                        # 閠・・莠矩�・
                        with st.expander("�統 閠・・莠矩�・):
                            for consideration in result["considerations"]:
                                st.write(f"窶｢ {consideration}")

                    else:
                        st.error(f"繧ｨ繝ｩ繝ｼ: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"繝ｪ繧ｯ繧ｨ繧ｹ繝医お繝ｩ繝ｼ: {e}")


def _render_analysis_plan():
    """隗｣譫舌・繝ｩ繝ｳ謠先｡・""
    st.subheader("�搭 隗｣譫舌・繝ｩ繝ｳ謠先｡・)
    st.markdown("蝠城｡瑚ｨ倩ｿｰ縺九ｉ縲∬ｧ｣譫先姶逡･繧呈署譯医＠縺ｾ縺吶・)

    with st.form("analysis_plan_form"):
        problem_description = st.text_area(
            "蝠城｡後・隱ｬ譏・,
            value="Predict aqueous solubility from SMILES",
            height=100,
            placeholder="What are you trying to predict?",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            n_samples = st.number_input("繧ｵ繝ｳ繝励Ν謨ｰ", min_value=10, value=1200, step=10)

        with col2:
            task_type = st.selectbox("繧ｿ繧ｹ繧ｯ", ["regression", "classification"])

        with col3:
            target_property = st.text_input("迚ｩ諤ｧ", value="logS")

        submitted = st.form_submit_button("�庁 繝励Λ繝ｳ繧呈署譯・, use_container_width=True)

        if submitted:
            with st.spinner("隗｣譫舌・繝ｩ繝ｳ繧剃ｽ懈・荳ｭ..."):
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

                        st.success("笨・隗｣譫舌・繝ｩ繝ｳ繧剃ｽ懈・縺励∪縺励◆")

                        # 逶ｮ逧・
                        st.markdown("### �識 逶ｮ逧・)
                        st.write(result["objective"])

                        # 謗ｨ螂ｨ繧｢繝励Ο繝ｼ繝・
                        st.markdown("### �ｧｭ 謗ｨ螂ｨ繧｢繝励Ο繝ｼ繝・)
                        st.write(result["recommended_approach"])

                        # 繝｢繝・Ν蛟呵｣・
                        st.markdown("### �､・謗ｨ螂ｨ繝｢繝・Ν")
                        model_cols = st.columns(len(result["model_suggestions"]))
                        for i, model in enumerate(result["model_suggestions"]):
                            model_cols[i].info(model)

                        # 讀懆ｨｼ謌ｦ逡･
                        st.markdown("### 笨・讀懆ｨｼ謌ｦ逡･")
                        st.write(result["validation_strategy"])

                        # 隱ｲ鬘・
                        with st.expander("笞�・・諠ｳ螳壹＆繧後ｋ隱ｲ鬘・):
                            for challenge in result["potential_challenges"]:
                                st.write(f"窶｢ {challenge}")

                    else:
                        st.error(f"繧ｨ繝ｩ繝ｼ: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"繝ｪ繧ｯ繧ｨ繧ｹ繝医お繝ｩ繝ｼ: {e}")


def _render_result_interpretation():
    """邨先棡隗｣驥・""
    st.subheader("�識 繝｢繝・Ν邨先棡縺ｮ隗｣驥・)
    st.markdown("隧穂ｾ｡謖・ｨ吶°繧峨∫ｵ先棡縺ｮ隗｣驥医→謾ｹ蝟・｡医ｒ謠先｡医＠縺ｾ縺吶・)

    with st.form("interpret_form"):
        st.markdown("#### 隧穂ｾ｡謖・ｨ吶ｒ蜈･蜉・)

        col1, col2 = st.columns(2)

        with col1:
            r2 = st.number_input("Rﾂｲ Score", min_value=-1.0, max_value=1.0, value=0.85, step=0.01)
            mae = st.number_input("MAE", min_value=0.0, value=0.42, step=0.01)

        with col2:
            rmse = st.number_input("RMSE", min_value=0.0, value=0.58, step=0.01)
            model_type = st.text_input("繝｢繝・Ν繧ｿ繧､繝・, value="XGBoost")

        submitted = st.form_submit_button("�剥 邨先棡繧定ｧ｣驥・, use_container_width=True)

        if submitted:
            with st.spinner("隗｣驥井ｸｭ..."):
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

                        st.success("笨・隗｣驥医′螳御ｺ・＠縺ｾ縺励◆")

                        # 隗｣驥・
                        st.markdown("### �眺 隗｣驥・)
                        st.write(result["interpretation"])

                        # 繝｡繝医Μ繧ｯ繧ｹ繧ｵ繝槭Μ繝ｼ
                        with st.expander("�投 蜈･蜉帙＆繧後◆繝｡繝医Μ繧ｯ繧ｹ"):
                            metric_cols = st.columns(3)
                            metric_cols[0].metric("Rﾂｲ", f"{r2:.3f}")
                            metric_cols[1].metric("MAE", f"{mae:.3f}")
                            metric_cols[2].metric("RMSE", f"{rmse:.3f}")

                    else:
                        st.error(f"繧ｨ繝ｩ繝ｼ: {res.status_code} - {res.text}")

                except Exception as e:
                    st.error(f"繝ｪ繧ｯ繧ｨ繧ｹ繝医お繝ｩ繝ｼ: {e}")


def _render_free_qa():
    """閾ｪ逕ｱ蠖｢蠑讐&A"""
    st.subheader("笶・閾ｪ逕ｱQ&A")
    st.markdown("蛹門ｭｦ讖滓｢ｰ蟄ｦ鄙偵↓髢｢縺吶ｋ雉ｪ蝠上↓蝗樒ｭ斐＠縺ｾ縺吶・)

    # 繧ｵ繝ｳ繝励Ν雉ｪ蝠・
    sample_questions = [
        "Morgan fingerprints縺ｯ縺・▽菴ｿ縺・∋縺阪〒縺吶°・・,
        "蟆上＆縺・ョ繝ｼ繧ｿ繧ｻ繝・ヨ・・100繧ｵ繝ｳ繝励Ν・峨〒驕主ｭｦ鄙偵ｒ驕ｿ縺代ｋ縺ｫ縺ｯ・・,
        "XGBoost縺ｨLightGBM縺ｮ驕輔＞縺ｯ・・,
        "SHAP蛟､縺ｮ隗｣驥域婿豕輔・・・,
    ]

    selected_sample = st.selectbox(
        "繧ｵ繝ｳ繝励Ν雉ｪ蝠擾ｼ医∪縺溘・荳九↓閾ｪ逕ｱ蜈･蜉幢ｼ・,
        ["-- 閾ｪ逕ｱ蜈･蜉・--"] + sample_questions
    )

    if selected_sample != "-- 閾ｪ逕ｱ蜈･蜉・--":
        question = selected_sample
    else:
        question = st.text_area(
            "雉ｪ蝠上ｒ蜈･蜉・,
            height=100,
            placeholder="萓・ 繝舌ャ繝∵ｭ｣隕丞喧縺ｨ縺ｯ菴輔〒縺吶°・・,
        )

    context = st.text_input("繧ｳ繝ｳ繝・く繧ｹ繝茨ｼ医が繝励す繝ｧ繝ｳ・・, placeholder="e.g., I'm working on QSAR modeling")

    if st.button("�､・雉ｪ蝠上☆繧・, use_container_width=True):
        if not question:
            st.warning("雉ｪ蝠上ｒ蜈･蜉帙＠縺ｦ縺上□縺輔＞")
            return

        with st.spinner("閠・∴荳ｭ..."):
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

                    st.success("笨・蝗樒ｭ斐′螳御ｺ・＠縺ｾ縺励◆")

                    # 雉ｪ蝠・
                    st.markdown("### 笶・雉ｪ蝠・)
                    st.info(result["question"])

                    # 蝗樒ｭ・
                    st.markdown("### �庁 蝗樒ｭ・)
                    st.write(result["answer"])

                    # LLM蛻ｩ逕ｨ迥ｶ豕・
                    if result.get("llm_available"):
                        st.caption("笨ｨ GPT4All (Full LLM) 繧剃ｽｿ逕ｨ")
                    else:
                        st.caption("�搭 繝ｫ繝ｼ繝ｫ繝吶・繧ｹ繝｢繝ｼ繝会ｼ育ｰ｡譏灘屓遲費ｼ・)

                else:
                    st.error(f"繧ｨ繝ｩ繝ｼ: {res.status_code} - {res.text}")

            except Exception as e:
                st.error(f"繝ｪ繧ｯ繧ｨ繧ｹ繝医お繝ｩ繝ｼ: {e}")
