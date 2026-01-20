"""
コマンドラインインターフェース（Typer/Click inspired）

Implements: F-CLI-001
設計思想:
- サブコマンド構造
- バッチ処理
- プログレス表示
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional


def main():
    """メインエントリポイント"""
    parser = argparse.ArgumentParser(
        prog='chemml',
        description='Chemical ML Platform CLI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # predict コマンド
    predict_parser = subparsers.add_parser('predict', help='Predict molecular properties')
    predict_parser.add_argument('smiles', nargs='+', help='SMILES strings')
    predict_parser.add_argument('--model', default='model.pkl', help='Model path')
    predict_parser.add_argument('--output', '-o', help='Output file')
    
    # features コマンド
    features_parser = subparsers.add_parser('features', help='Extract features')
    features_parser.add_argument('input', help='Input CSV file')
    features_parser.add_argument('--output', '-o', required=True, help='Output file')
    features_parser.add_argument('--type', default='rdkit', choices=['rdkit', 'morgan', 'all'])
    
    # train コマンド
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('data', help='Training data CSV')
    train_parser.add_argument('--target', required=True, help='Target column')
    train_parser.add_argument('--model-type', default='lightgbm')
    train_parser.add_argument('--output', '-o', default='model.pkl')
    
    # analyze コマンド
    analyze_parser = subparsers.add_parser('analyze', help='Analyze molecule')
    analyze_parser.add_argument('smiles', help='SMILES string')
    analyze_parser.add_argument('--all', action='store_true', help='Show all properties')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        cmd_predict(args.smiles, args.model, args.output)
    elif args.command == 'features':
        cmd_features(args.input, args.output, args.type)
    elif args.command == 'train':
        cmd_train(args.data, args.target, args.model_type, args.output)
    elif args.command == 'analyze':
        cmd_analyze(args.smiles, args.all)
    else:
        parser.print_help()


def cmd_predict(smiles_list: List[str], model_path: str, output: Optional[str]):
    """予測コマンド"""
    print(f"Predicting {len(smiles_list)} molecules...")
    
    try:
        import joblib
        from core.services.features.rdkit_eng import RDKitFeatureExtractor
        
        model = joblib.load(model_path)
        extractor = RDKitFeatureExtractor()
        
        X = extractor.transform(smiles_list)
        X = X.drop(columns=['SMILES'], errors='ignore')
        
        predictions = model.predict(X)
        
        results = [
            {'smiles': smi, 'prediction': float(pred)}
            for smi, pred in zip(smiles_list, predictions)
        ]
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output}")
        else:
            for r in results:
                print(f"{r['smiles']}: {r['prediction']:.4f}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_features(input_file: str, output_file: str, feature_type: str):
    """特徴量抽出コマンド"""
    print(f"Extracting {feature_type} features from {input_file}...")
    
    try:
        import pandas as pd
        from core.services.features.rdkit_eng import RDKitFeatureExtractor
        from core.services.features.fingerprint import FingerprintCalculator
        
        df = pd.read_csv(input_file)
        smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
        smiles_list = df[smiles_col].tolist()
        
        if feature_type == 'morgan':
            extractor = FingerprintCalculator(fp_type='morgan')
            features = extractor.calculate_batch(smiles_list)
        else:
            extractor = RDKitFeatureExtractor()
            features = extractor.transform(smiles_list)
        
        features.to_csv(output_file, index=False)
        print(f"Features saved to {output_file} ({len(features.columns)} columns)")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_train(data_file: str, target: str, model_type: str, output: str):
    """訓練コマンド"""
    print(f"Training {model_type} model on {data_file}...")
    
    try:
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(data_file)
        
        if target not in df.columns:
            print(f"Target column '{target}' not found", file=sys.stderr)
            sys.exit(1)
        
        y = df[target]
        X = df.drop(columns=[target])
        X = X.select_dtypes(include=['number'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        if model_type == 'lightgbm':
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=100, verbose=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100)
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        joblib.dump(model, output)
        print(f"Model saved to {output} (R² = {score:.4f})")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_analyze(smiles: str, show_all: bool):
    """分析コマンド"""
    print(f"Analyzing: {smiles}")
    print("=" * 50)
    
    try:
        from core.services.features.mol_property import MolecularPropertyCalculator
        from core.services.features.admet import ADMETPredictor
        from core.services.features.solubility import SolubilityPredictor
        
        # 基本物性
        calc = MolecularPropertyCalculator()
        props = calc.calculate(smiles)
        
        if props:
            print(f"MW: {props.molecular_weight:.1f}")
            print(f"LogP: {props.logp:.2f}")
            print(f"TPSA: {props.tpsa:.1f}")
            print(f"Drug-like: {'Yes' if props.is_druglike else 'No'}")
        
        if show_all:
            # ADMET
            admet = ADMETPredictor()
            admet_result = admet.predict(smiles)
            if admet_result:
                print(f"\nADMET:")
                print(f"  HIA: {admet_result.hia}%")
                print(f"  BBB: {admet_result.bbb_penetration}")
            
            # 溶解度
            sol = SolubilityPredictor()
            sol_result = sol.predict(smiles)
            if sol_result:
                print(f"\nSolubility:")
                print(f"  LogS: {sol_result.logS}")
                print(f"  Class: {sol_result.solubility_class}")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
