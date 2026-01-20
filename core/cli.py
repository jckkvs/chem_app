#!/usr/bin/env python
"""
ChemML コマンドラインインターフェース

使用方法:
    python -m core.cli predict --smiles "CCO" --model my_model
    python -m core.cli extract --input data.csv --output features.csv
    python -m core.cli train --input data.csv --target logP --model rf
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def create_parser() -> argparse.ArgumentParser:
    """CLIパーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="chemml",
        description="ChemML - Chemical Machine Learning Platform",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # === predict コマンド ===
    predict_parser = subparsers.add_parser(
        "predict", help="Predict properties for molecules"
    )
    predict_parser.add_argument(
        "--smiles", "-s", type=str, help="Single SMILES string"
    )
    predict_parser.add_argument(
        "--input", "-i", type=str, help="Input CSV file with SMILES"
    )
    predict_parser.add_argument(
        "--model", "-m", type=str, required=True, help="Model name"
    )
    predict_parser.add_argument(
        "--output", "-o", type=str, help="Output file"
    )
    predict_parser.add_argument(
        "--uncertainty", action="store_true", help="Include uncertainty"
    )
    
    # === extract コマンド ===
    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from molecules"
    )
    extract_parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input CSV file"
    )
    extract_parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output CSV file"
    )
    extract_parser.add_argument(
        "--smiles-column", type=str, default="smiles", help="SMILES column name"
    )
    extract_parser.add_argument(
        "--preset", "-p", type=str, default="general", help="Descriptor preset"
    )
    
    # === train コマンド ===
    train_parser = subparsers.add_parser(
        "train", help="Train a prediction model"
    )
    train_parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input CSV file"
    )
    train_parser.add_argument(
        "--target", "-t", type=str, required=True, help="Target column"
    )
    train_parser.add_argument(
        "--model", "-m", type=str, default="rf", 
        choices=["rf", "gb", "lgbm", "xgb"],
        help="Model type"
    )
    train_parser.add_argument(
        "--name", "-n", type=str, help="Model name for saving"
    )
    train_parser.add_argument(
        "--preset", "-p", type=str, default="general", help="Descriptor preset"
    )
    
    # === analyze コマンド ===
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a molecular dataset"
    )
    analyze_parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input CSV file"
    )
    analyze_parser.add_argument(
        "--smiles-column", type=str, default="smiles", help="SMILES column"
    )
    analyze_parser.add_argument(
        "--output", "-o", type=str, help="Output report file"
    )
    
    # === list コマンド ===
    list_parser = subparsers.add_parser(
        "list", help="List available resources"
    )
    list_parser.add_argument(
        "resource", choices=["presets", "models"],
        help="Resource to list"
    )
    
    return parser


def cmd_predict(args: argparse.Namespace) -> int:
    """予測コマンド"""
    from core.services.utils import load_model
    from core.services.features import SmartFeatureEngine
    
    # SMILESを取得
    if args.smiles:
        smiles_list = [args.smiles]
    elif args.input:
        df = pd.read_csv(args.input)
        smiles_list = df['smiles'].tolist()
    else:
        print("Error: --smiles or --input required")
        return 1
    
    # モデルをロード
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # 特徴量抽出
    engine = SmartFeatureEngine()
    result = engine.fit_transform(smiles_list)
    X = result.features
    
    # 予測
    predictions = model.predict(X)
    
    # 出力
    output_df = pd.DataFrame({
        'smiles': smiles_list,
        'prediction': predictions,
    })
    
    if args.output:
        output_df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
    else:
        print(output_df.to_string(index=False))
    
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """特徴量抽出コマンド"""
    from core.services.features import SmartFeatureEngine
    
    # データ読み込み
    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_column].tolist()
    
    print(f"Extracting features from {len(smiles_list)} molecules...")
    
    # 特徴量抽出
    engine = SmartFeatureEngine(target_property=args.preset)
    result = engine.fit_transform(smiles_list)
    
    # 保存
    result.features.to_csv(args.output, index=False)
    print(f"Saved {result.n_features} features to {args.output}")
    
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """学習コマンド"""
    from core.services.features import SmartFeatureEngine
    from core.services.utils import save_model
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    
    # データ読み込み
    df = pd.read_csv(args.input)
    
    if 'smiles' not in df.columns:
        print("Error: 'smiles' column not found")
        return 1
    
    if args.target not in df.columns:
        print(f"Error: Target column '{args.target}' not found")
        return 1
    
    smiles_list = df['smiles'].tolist()
    y = df[args.target].values
    
    print(f"Training on {len(smiles_list)} samples...")
    
    # 特徴量抽出
    engine = SmartFeatureEngine(target_property=args.preset)
    result = engine.fit_transform(smiles_list)
    X = result.features.values
    
    # モデル選択
    if args.model == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif args.model == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        try:
            if args.model == 'lgbm':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            elif args.model == 'xgb':
                from xgboost import XGBRegressor
                model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        except ImportError:
            print(f"Error: {args.model} not installed")
            return 1
    
    # 交差検証
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"CV R² Score: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # 学習
    model.fit(X, y)
    
    # 保存
    model_name = args.name or f"{args.target}_{args.model}"
    metrics = {'r2_cv': scores.mean(), 'r2_cv_std': scores.std()}
    path = save_model(model, model_name, metrics=metrics)
    
    print(f"Saved model to {path}")
    
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """分析コマンド"""
    from core.services.features import analyze_dataset, analyze_scaffolds
    
    # データ読み込み
    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_column].tolist()
    
    print(f"Analyzing {len(smiles_list)} molecules...\n")
    
    # データセット分析
    profile = analyze_dataset(smiles_list)
    
    print("=== Dataset Profile ===")
    print(f"Valid molecules: {profile.n_valid}/{profile.n_total}")
    print(f"Average MW: {profile.mw_mean:.1f}")
    print(f"Recommended preset: {profile.recommended_preset}")
    
    # 骨格分析
    scaffold = analyze_scaffolds(smiles_list)
    
    print(f"\n=== Scaffold Analysis ===")
    print(f"Unique scaffolds: {scaffold.n_unique_scaffolds}")
    print(f"Scaffold diversity: {scaffold.scaffold_diversity:.2%}")
    
    # レポート出力
    if args.output:
        from core.services.utils.report_generator import ReportGenerator
        
        gen = ReportGenerator()
        report = gen.create_experiment_report(
            experiment_name="Dataset Analysis",
            smiles_list=smiles_list,
        )
        
        report.add_section(
            "Dataset Profile",
            f"Valid: {profile.n_valid}, MW: {profile.mw_mean:.1f}",
        )
        
        gen.export_html(report, args.output)
        print(f"\nReport saved to {args.output}")
    
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """リストコマンド"""
    if args.resource == "presets":
        from core.services.features import list_presets
        
        presets = list_presets()
        print("Available Presets:")
        for name, name_ja in presets.items():
            print(f"  {name}: {name_ja}")
    
    elif args.resource == "models":
        from core.services.utils import ModelPersistence
        
        mp = ModelPersistence()
        models = mp.list_models()
        
        if models:
            print("Saved Models:")
            for name in models:
                versions = mp.list_versions(name)
                print(f"  {name} ({len(versions)} versions)")
        else:
            print("No saved models found.")
    
    return 0


def main() -> int:
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "predict": cmd_predict,
        "extract": cmd_extract,
        "train": cmd_train,
        "analyze": cmd_analyze,
        "list": cmd_list,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
