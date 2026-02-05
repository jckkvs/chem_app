"""
Chemical ML Platform - サンプルプログラム集

人間がデータを丁寧に見ることを支援する、実践的な使用例
"""

# ========================================
# サンプル1: クイックEDA（1行起動）
# ========================================

def example_01_quick_eda():
    """最速EDA - データを1行で把握"""
    import pandas as pd
    from core.services.vis.eda_dashboard import quick_eda
    
    # サンプルデータ
    df = pd.DataFrame({
        'SMILES': ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCC', 'c1ccncc1'] * 20,
        'LogP': [0.5, 2.1, 0.3, 1.8, 1.5] * 20,
        'MolWeight': [46, 78, 60, 58, 79] * 20,
        'TPSA': [20, 0, 37, 0, 13] * 20,
    })
    
    # 1行でEDA完了
    dashboard = quick_eda(df, target_column='LogP')
    
    # カスタムプロット生成
    dashboard.plot_pairplot(max_features=4)
    dashboard.plot_correlation_heatmap()
    dashboard.plot_pca(color_by='LogP')
    
    print("✅ EDA完了！")


# ========================================
# サンプル2: SMILES → 物性予測
# ========================================

def example_02_smiles_prediction():
    """SMI

LESから分子物性を予測"""
    import pandas as pd
    from core.services.features.fingerprint import FingerprintEngine
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # サンプルデータ
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCC', 'c1ccncc1'] * 20
    logp_values = [0.5, 2.1, 0.3, 1.8, 1.5] * 20
    
    # 特徴量抽出
    fp_engine = FingerprintEngine()
    fp_df, _ = fp_engine.compute_fingerprints(smiles, fp_type='morgan', radius=2, n_bits=1024)
    
    # 訓練
    X = fp_df.values
    y = logp_values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 予測
    score = model.score(X_test, y_test)
    print(f"✅ R² Score: {score:.3f}")
    
    # 新規分子の予測
    new_smiles = ['CCCCO']
    new_fp, _ = fp_engine.compute_fingerprints(new_smiles, fp_type='morgan', radius=2, n_bits=1024)
    prediction = model.predict(new_fp.values)
    print(f"✅ 予測LogP: {prediction[0]:.2f}")


# ========================================
# サンプル3: SmartFeatureEngine（物性別最適化）
# ========================================

def example_03_smart_feature_engine():
    """物性別に最適化された特徴量生成"""
    from core.services.features.smart_feature_engine import SmartFeatureEngine, FeatureConfig
    
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O'] * 10
    
    # ガラス転移温度（Tg）向けに最適化
    engine = SmartFeatureEngine(
        config=FeatureConfig(
            target_property='glass_transition',
            include_rdkit=True,
            include_morgan_fp=True,
            include_maccs=False,
        )
    )
    
    result = engine.fit_transform(smiles)
    
    print(f"✅ 生成特徴量: {result.combined_features.shape[1]}次元")
    print(f"✅ 除外特徴量（低分散）: {len(result.removed_low_variance)}")
    print(f"✅ 除外特徴量（高相関）: {len(result.removed_high_correlation)}")


# ========================================
# サンプル4: 相関分析 + Pairplot
# ========================================

def example_04_correlation_pairplot():
    """データの関係性を可視化"""
    import pandas as pd
    from core.services.vis.eda_dashboard import EDADashboard
    
    df = pd.DataFrame({
        'LogP': [0.5, 2.1, 0.3, 1.8, 1.5] * 20,
        'MolWeight': [46, 78, 60, 58, 79] * 20,
        'TPSA': [20, 0, 37, 0, 13] * 20,
        'HBA': [1, 0, 2, 0, 1] * 20,
        'HBD': [1, 0, 1, 0, 0] * 20,
    })
    
    dashboard = EDADashboard(df)
    
    # 相関分析
    dashboard.plot_correlation_heatmap(method='spearman')
    high_corr = dashboard.find_high_correlations(threshold=0.7)
    print(f"✅ 高相関ペア: {len(high_corr)}個")
    
    # Pairplot
    dashboard.plot_pairplot(max_features=5)
    
    print("✅ 相関分析完了！")


# ========================================
# サンプル5: PCA + クラスタリング
# ========================================

def example_05_pca_clustering():
    """次元削減とクラスタリング"""
    import pandas as pd
    import numpy as np
    from core.services.vis.eda_dashboard import EDADashboard
    
    # ダミーデータ
    np.random.seed(42)
    df = pd.DataFrame({
        'feat1': np.random.randn(100),
        'feat2': np.random.randn(100) + 2,
        'feat3': np.random.randn(100) - 1,
        'feat4': np.random.randn(100) * 2,
        'feat5': np.random.randn(100) + 1,
    })
    
    dashboard = EDADashboard(df)
    
    # PCA
    fig, pca = dashboard.plot_pca(n_components=2)
    print(f"✅ PCA寄与率: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    
    # K-meansクラスタリング
    fig, kmeans = dashboard.plot_kmeans_clusters(n_clusters=3)
    print(f"✅ K-means Inertia: {kmeans.inertia_:.2f}")


# ========================================
# サンプル6: フルEDAレポート自動生成
# ========================================

def example_06_full_eda_report():
    """全可視化を一括生成"""
    import pandas as pd
    from core.services.vis.eda_dashboard import EDADashboard
    
    df = pd.DataFrame({
        'LogP': [0.5, 2.1, 0.3, 1.8, 1.5] * 20,
        'MolWeight': [46, 78, 60, 58, 79] * 20,
        'TPSA': [20, 0, 37, 0, 13] * 20,
    })
    
    dashboard = EDADashboard(df, target_column='LogP')
    
    # フルレポート生成（全プロット自動作成）
    report = dashboard.generate_full_report(output_dir='./eda_output')
    
    print(f"✅ 生成プロット数: {report['n_plots']}")
    print(f"✅ 出力先: ./eda_output")


# ========================================
# サンプル7: API経由でバッチ予測
# ========================================

def example_07_api_batch_predict():
    """REST API経由でバッチ予測"""
    import requests
    
    # NOTE: Django serverが起動している必要があります
    # python manage.py runserver
    
    API_URL = "http://localhost:8000/api"
    
    # バッチ予測
    smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
    
    try:
        response = requests.post(
            f"{API_URL}/experiments/1/batch_predict",
            json={"smiles_list": smiles_list},
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ バッチ予測成功！")
            for smi, pred in zip(smiles_list, results['predictions']):
                print(f"  {smi}: {pred:.3f}")
        else:
            print(f"⚠️ エラー: {response.text}")
    except requests.ConnectionError:
        print("⚠️ APIサーバーに接続できません。Django serverを起動してください。")


# ========================================
# サンプル8: 機械学習パイプライン完全版
# ========================================

def example_08_complete_ml_pipeline():
    """前処理からモデル評価まで"""
    import pandas as pd
    from core.services.features.fingerprint import FingerprintEngine
    from core.services.ml.preprocessor import SmartPreprocessor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # データ
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCC', 'c1ccncc1'] * 20
    target = [0.5, 2.1, 0.3, 1.8, 1.5] * 20
    
    # 特徴量抽出
    fp_engine = FingerprintEngine()
    fp_df, _ = fp_engine.compute_fingerprints(smiles, fp_type='morgan')
    
    # 前処理
    preprocessor = SmartPreprocessor(
        continuous_scaler='standard',
        handle_outliers='clip'
    )
    X_processed = preprocessor.fit_transform(fp_df)
    
    # モデル訓練
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # クロスバリデーション
    cv_scores = cross_val_score(model, X_processed, target, cv=5, scoring='r2')
    print(f"✅ CV R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # モデル訓練
    model.fit(X_processed, target)
    print("✅ パイプライン完了！")


# ========================================
# メイン実行
# ========================================

if __name__ == "__main__":
    print("=" * 80)
    print("Chemical ML Platform - サンプルプログラム集")
    print("=" * 80)
    
    import sys
    
    samples = {
        '1': ('クイックEDA（1行起動）', example_01_quick_eda),
        '2': ('SMILES → 物性予測', example_02_smiles_prediction),
        '3': ('SmartFeatureEngine（物性別最適化）', example_03_smart_feature_engine),
        '4': ('相関分析 + Pairplot', example_04_correlation_pairplot),
        '5': ('PCA + クラスタリング', example_05_pca_clustering),
        '6': ('フルEDAレポート自動生成', example_06_full_eda_report),
        '7': ('API経由でバッチ予測', example_07_api_batch_predict),
        '8': ('機械学習パイプライン完全版', example_08_complete_ml_pipeline),
    }
    
    print("\n実行サンプル:")
    for key, (name, _) in samples.items():
        print(f"  {key}. {name}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n実行するサンプル番号を選択 (1-8): ")
    
    if choice in samples:
        name, func = samples[choice]
        print(f"\n{'=' * 80}")
        print(f"サンプル{choice}: {name}")
        print(f"{'=' * 80}\n")
        func()
    else:
        print("無効な選択です")
