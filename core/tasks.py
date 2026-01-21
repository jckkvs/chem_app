"""
Huey バックグラウンドタスク - 4タスクタイプ対応版

Implements: F-TASK-002
設計思想:
- 4タスクタイプ対応:
  1. smiles_only: SMILES→物性予測
  2. tabular_only: 表データ→特性予測
  3. mixture: 混合物（SMILES＋割合）→物性予測
  4. smiles_tabular: SMILES＋表データ→物性予測
- SmartFeatureEngineによる物性別記述子選択
- TARTE統合（表データ用Transformer）
- 継続的改善を前提とした設計
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import django
from django.conf import settings

# Django設定
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chem_ml_project.settings')
    django.setup()

from huey import SqliteHuey

logger = logging.getLogger(__name__)

# Huey初期化
huey = SqliteHuey(
    'chem_ml_huey',
    filename=os.path.join(settings.BASE_DIR, 'huey_db.sqlite3'),
)


# =============================================================================
# タスクタイプ別の特徴量生成
# =============================================================================

def extract_features_smiles_only(
    smiles: List[str],
    y,
    config: Dict[str, Any],
    tracker
) -> 'pd.DataFrame':
    """
    SMILES onlyモード: 分子記述子のみ
    
    SmartFeatureEngineを使用して物性別の最適記述子を選択
    """
    import pandas as pd

    from core.services.features import (
        RDKitFeatureExtractor,
        SmartFeatureEngine,
        UMAFeatureExtractor,
        XTBFeatureExtractor,
    )
    from core.services.features.descriptor_selector import DescriptorSelector
    
    target_property = config.get('target_property', 'general')
    use_smart_engine = config.get('use_smart_engine', True)
    
    if use_smart_engine:
        # SmartFeatureEngine使用（推奨）
        logger.info(f"SmartFeatureEngine使用: target_property={target_property}")
        
        engine = SmartFeatureEngine(
            target_property=target_property,
            use_pretrained=config.get('pretrained_models', []),
            use_morgan_fp=config.get('use_morgan_fp', False),
            use_xtb='xtb' in config.get('features', []),
        )
        
        result = engine.fit_transform(smiles, y)
        
        # データセット分析結果をログ
        if result.dataset_profile:
            for note in result.dataset_profile.analysis_notes:
                logger.info(f"Dataset analysis: {note}")
        
        return result.features
    
    else:
        # 従来モード（互換性のため維持）
        features_df_list = []
        
        if 'rdkit' in config.get('features', []):
            logger.info("Extracting RDKit features...")
            rdkit_eng = RDKitFeatureExtractor(
                categories=['lipophilicity', 'structural', 'topological']
            )
            features_df_list.append(rdkit_eng.transform(smiles))
        
        if 'xtb' in config.get('features', []):
            logger.info("Extracting XTB features...")
            xtb_eng = XTBFeatureExtractor(
                xtb_path=config.get('xtb_path', 'xtb')
            )
            features_df_list.append(xtb_eng.transform(smiles))
        
        if 'uma' in config.get('features', []):
            logger.info("Extracting UMA features...")
            uma_eng = UMAFeatureExtractor(n_components=10)
            uma_eng.fit(smiles, y=y)
            features_df_list.append(uma_eng.transform(smiles))
            
            # UMAモデルを保存
            uma_path = os.path.join(tempfile.gettempdir(), "uma_reducer.joblib")
            uma_eng.save(uma_path)
            tracker.log_artifact(uma_path)
        
        if not features_df_list:
            raise ValueError("No features selected")
        
        X = pd.concat(
            [f.drop(columns=['SMILES'], errors='ignore') for f in features_df_list],
            axis=1,
        )
        
        # 記述子選択
        if config.get('descriptor_selection', True) and len(X.columns) > 30:
            logger.info("Selecting optimal descriptors...")
            selector = DescriptorSelector(
                preset=target_property if target_property != 'general' else None,
                selection_method='correlation',
                top_k=30,
                vif_threshold=10.0,
            )
            X = selector.fit_transform(X, y)
        
        return X


def extract_features_tabular_only(
    df: 'pd.DataFrame',
    y,
    config: Dict[str, Any],
    tracker
) -> 'pd.DataFrame':
    """
    Tabular onlyモード: 表データの数値カラムを使用
    
    TARTEが利用可能な場合は埋め込みも追加
    """
    import pandas as pd

    from core.services.features.tarte_eng import TarteFeatureExtractor, is_tarte_available
    from core.services.ml.preprocessor import SmartPreprocessor
    
    target_col = config.get('target_col', 'target')
    
    # 数値カラムを抽出
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = df[numeric_cols].copy()
    logger.info(f"Tabular features: {len(numeric_cols)} numeric columns")
    
    # TARTE埋め込み（オプション）
    if config.get('use_tarte', False) and is_tarte_available():
        logger.info("Adding TARTE embeddings...")
        tarte_mode = config.get('tarte_mode', 'featurizer')
        tarte_eng = TarteFeatureExtractor(mode=tarte_mode)
        tarte_eng.fit(X, y=y)
        tarte_features = tarte_eng.transform(X)
        
        # 元の特徴量とTARTE埋め込みを結合
        X = pd.concat([X.reset_index(drop=True), tarte_features.reset_index(drop=True)], axis=1)
        logger.info(f"Combined with TARTE: {X.shape[1]} features")
    
    return X


def extract_features_mixture(
    df: 'pd.DataFrame',
    y,
    config: Dict[str, Any],
    tracker
) -> 'pd.DataFrame':
    """
    混合物モード: 複数SMILES＋割合から加重平均記述子を計算
    
    想定カラム構造:
    - smiles_1, smiles_2, ... : 各成分のSMILES
    - ratio_1, ratio_2, ... : 各成分の割合（合計1.0）
    """
    import numpy as np
    import pandas as pd

    from core.services.features import RDKitFeatureExtractor

    # SMILESカラムと割合カラムを特定
    smiles_cols = [c for c in df.columns if c.lower().startswith('smiles')]
    ratio_cols = [c for c in df.columns if c.lower().startswith('ratio')]
    
    if not smiles_cols:
        raise ValueError("混合物モードにはsmiles_*カラムが必要です")
    
    logger.info(f"Mixture mode: {len(smiles_cols)} components")
    
    rdkit_eng = RDKitFeatureExtractor()
    
    # 各成分の記述子を計算し、加重平均
    weighted_features = None
    
    for i, (smi_col, ratio_col) in enumerate(zip(smiles_cols, ratio_cols)):
        smiles_list = df[smi_col].fillna('').tolist()
        ratios = df[ratio_col].fillna(0).values.reshape(-1, 1)
        
        # 空のSMILESをダミーに置換
        smiles_list = [s if s else 'C' for s in smiles_list]
        
        features = rdkit_eng.transform(smiles_list)
        features = features.drop(columns=['SMILES'], errors='ignore')
        
        # 加重
        weighted = features.values * ratios
        
        if weighted_features is None:
            weighted_features = weighted
        else:
            weighted_features += weighted
    
    X = pd.DataFrame(weighted_features, columns=[f"mix_{c}" for c in features.columns])
    logger.info(f"Mixture features: {X.shape}")
    
    return X


def extract_features_smiles_tabular(
    df: 'pd.DataFrame',
    smiles: List[str],
    y,
    config: Dict[str, Any],
    tracker
) -> 'pd.DataFrame':
    """
    SMILES＋Tabularハイブリッドモード
    
    分子記述子と表データ特徴量を結合
    """
    import pandas as pd

    from core.services.features import SmartFeatureEngine
    from core.services.features.tarte_eng import TarteFeatureExtractor, is_tarte_available
    
    target_property = config.get('target_property', 'general')
    target_col = config.get('target_col', 'target')
    
    # 1. 分子記述子
    logger.info("Extracting molecular features...")
    engine = SmartFeatureEngine(
        target_property=target_property,
        use_pretrained=config.get('pretrained_models', []),
    )
    mol_result = engine.fit_transform(smiles, y)
    mol_features = mol_result.features
    
    # 2. 表データ特徴量
    logger.info("Extracting tabular features...")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    # SMILESカラムとターゲットカラムを除外
    exclude_cols = config.get('exclude_cols', []) + [target_col]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    tabular_features = df[numeric_cols].copy() if numeric_cols else pd.DataFrame()
    
    # 3. TARTE埋め込み（表データ部分に対して）
    if config.get('use_tarte', False) and is_tarte_available() and not tabular_features.empty:
        logger.info("Adding TARTE embeddings for tabular data...")
        tarte_eng = TarteFeatureExtractor(mode=config.get('tarte_mode', 'featurizer'))
        tarte_eng.fit(tabular_features, y=y)
        tarte_features = tarte_eng.transform(tabular_features)
        tabular_features = pd.concat([
            tabular_features.reset_index(drop=True), 
            tarte_features.reset_index(drop=True)
        ], axis=1)
    
    # 4. 結合
    X = pd.concat([
        mol_features.reset_index(drop=True),
        tabular_features.reset_index(drop=True),
    ], axis=1)
    
    logger.info(f"Combined features: {X.shape}")
    
    return X


# =============================================================================
# メインタスク
# =============================================================================

@huey.task()
def run_training_task(experiment_id: int):
    """
    非同期訓練タスク - 4タスクタイプ対応
    
    Args:
        experiment_id: 実験ID
    """
    logger.info(f"Training task started: experiment_id={experiment_id}")
    
    try:
        import pandas as pd

        from core.models import Experiment
        from core.services.ml.pipeline import MLPipeline
        from core.services.ml.tracking import MLTracker

        # 実験取得
        exp = Experiment.objects.get(id=experiment_id)
        exp.status = 'RUNNING'
        exp.save()
        logger.info(f"Experiment {exp.id} status set to RUNNING")
        
        # データ読み込み
        df = pd.read_csv(exp.dataset.file_path)
        y = df[exp.dataset.target_col]
        
        # Tracker初期化
        tracker = MLTracker(experiment_name=exp.name)
        
        # タスクタイプを取得
        task_type = exp.config.get('task_type_mode', 'smiles_only')
        logger.info(f"Task type: {task_type}")
        
        # タスクタイプに応じた特徴量抽出
        if task_type == 'smiles_only':
            smiles = df[exp.dataset.smiles_col].tolist()
            logger.info(f"SMILES only mode: {len(smiles)} samples")
            X = extract_features_smiles_only(smiles, y, exp.config, tracker)
            
        elif task_type == 'tabular_only':
            logger.info(f"Tabular only mode: {len(df)} samples")
            X = extract_features_tabular_only(df, y, exp.config, tracker)
            
        elif task_type == 'mixture':
            logger.info(f"Mixture mode: {len(df)} samples")
            X = extract_features_mixture(df, y, exp.config, tracker)
            
        elif task_type == 'smiles_tabular':
            smiles = df[exp.dataset.smiles_col].tolist()
            logger.info(f"SMILES + Tabular mode: {len(smiles)} samples")
            X = extract_features_smiles_tabular(df, smiles, y, exp.config, tracker)
            
        else:
            # デフォルト: SMILES only
            smiles = df[exp.dataset.smiles_col].tolist()
            logger.warning(f"Unknown task type '{task_type}', falling back to smiles_only")
            X = extract_features_smiles_only(smiles, y, exp.config, tracker)
        
        logger.info(f"Features generated: {X.shape}")
        
        # パイプライン訓練
        pipeline = MLPipeline(
            model_type=exp.config.get('model_type', 'lightgbm'),
            task_type=exp.config.get('task_type', 'regression'),
            cv_folds=exp.config.get('cv_folds', 5),
            tracker=tracker,
            config=exp.config,
        )
        
        metrics = pipeline.train(X, y, run_name=f"exp_{exp.id}")
        
        # 完了
        exp.metrics = metrics
        exp.status = 'COMPLETED'
        exp.save()
        
        logger.info(f"Experiment {exp.id} completed: {metrics}")
        
    except Exception as e:
        logger.error(f"Training task failed: {e}", exc_info=True)
        
        try:
            exp = Experiment.objects.get(id=experiment_id)
            exp.status = 'FAILED'
            exp.metrics = {'error': str(e)}
            exp.save()
        except Exception:
            pass
