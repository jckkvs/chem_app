"""
CLIのテスト

カバレッジ目標: 80%以上（160 stmts）
"""
import argparse
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import sys

from core import cli


class TestCLIParser:
    """パーサー作成テスト"""
    
    def test_create_parser(self):
        """パーサー作成"""
        parser = cli.create_parser()
        
        assert parser is not None
        assert parser.prog == "chemml"
    
    def test_parser_predict_command(self):
        """predictコマンドパーサー"""
        parser = cli.create_parser()
        
        args = parser.parse_args(['predict', '-m', 'test_model', '-s', 'CCO'])
        
        assert args.command == 'predict'
        assert args.model == 'test_model'
        assert args.smiles == 'CCO'
    
    def test_parser_extract_command(self):
        """extractコマンドパーサー"""
        parser = cli.create_parser()
        
        args = parser.parse_args([
            'extract', '-i', 'input.csv', '-o', 'output.csv'
        ])
        
        assert args.command == 'extract'
        assert args.input == 'input.csv'
        assert args.output == 'output.csv'
    
    def test_parser_train_command(self):
        """trainコマンドパーサー"""
        parser = cli.create_parser()
        
        args = parser.parse_args([
            'train', '-i', 'data.csv', '-t', 'logP', '-m', 'rf'
        ])
        
        assert args.command == 'train'
        assert args.target == 'logP'
        assert args.model == 'rf'
    
    def test_parser_analyze_command(self):
        """analyzeコマンドパーサー"""
        parser = cli.create_parser()
        
        args = parser.parse_args(['analyze', '-i', 'data.csv'])
        
        assert args.command == 'analyze'
        assert args.input == 'data.csv'
    
    def test_parser_list_command(self):
        """listコマンドパーサー"""
        parser = cli.create_parser()
        
        args = parser.parse_args(['list', 'presets'])
        
        assert args.command == 'list'
        assert args.resource == 'presets'


class TestCmdPredict:
    """predictコマンドテスト"""
    
    @patch('core.services.features.SmartFeatureEngine')
    @patch('core.services.utils.load_model')
    def test_predict_single_smiles(self, mock_load_model, mock_engine_class):
        """単一SMILES予測"""
        # モック設定
        mock_model = MagicMock()
        mock_model.predict.return_value = [1.5]
        mock_load_model.return_value = mock_model
        
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.features = pd.DataFrame([[1, 2, 3]])
        mock_engine.fit_transform.return_value = mock_result
        mock_engine_class.return_value = mock_engine
        
        args = argparse.Namespace(
            smiles='CCO',
            input=None,
            model='test_model',
            output=None,
            uncertainty=False
        )
        
        # 実行
        with patch('builtins.print'):
            result = cli.cmd_predict(args)
        
        assert result == 0
        mock_load_model.assert_called_once_with('test_model')
        mock_engine.fit_transform.assert_called_once_with(['CCO'])
    
    @patch('core.services.features.SmartFeatureEngine')
    @patch('core.services.utils.load_model')
    def test_predict_from_csv(self, mock_load_model, mock_engine_class):
        """CSV入力予測"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # テストCSV作成
            csv_path = os.path.join(tmpdir, 'input.csv')
            pd.DataFrame({'smiles': ['CCO', 'CC']}).to_csv(csv_path, index=False)
            
            # モック設定
            mock_model = MagicMock()
            mock_model.predict.return_value = [1.5, 2.3]
            mock_load_model.return_value = mock_model
            
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.features = pd.DataFrame([[1, 2], [3, 4]])
            mock_engine.fit_transform.return_value = mock_result
            mock_engine_class.return_value = mock_engine
            
            args = argparse.Namespace(
                smiles=None,
                input=csv_path,
                model='test_model',
                output=None,
                uncertainty=False
            )
            
            with patch('builtins.print'):
                result = cli.cmd_predict(args)
            
            assert result == 0
    
    @patch('core.services.utils.load_model')
    def test_predict_no_input(self, mock_load_model):
        """入力なしエラー"""
        args = argparse.Namespace(
            smiles=None,
            input=None,
            model='test_model',
            output=None,
            uncertainty=False
        )
        
        with patch('builtins.print'):
            result = cli.cmd_predict(args)
        
        assert result == 1
    
    @patch('core.services.utils.load_model')
    def test_predict_model_load_error(self, mock_load_model):
        """モデルロードエラー"""
        mock_load_model.side_effect = Exception("Model not found")
        
        args = argparse.Namespace(
            smiles='CCO',
            input=None,
            model='invalid_model',
            output=None,
            uncertainty=False
        )
        
        with patch('builtins.print'):
            result = cli.cmd_predict(args)
        
        assert result == 1


class TestCmdExtract:
    """extractコマンドテスト"""
    
    @patch('core.services.features.SmartFeatureEngine')
    def test_extract_features(self, mock_engine_class):
        """特徴量抽出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 入力CSV作成
            csv_input = os.path.join(tmpdir, 'input.csv')
            pd.DataFrame({'smiles': ['CCO', 'CC']}).to_csv(csv_input, index=False)
            
            # モック設定
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.features = pd.DataFrame([[1, 2], [3, 4]])
            mock_result.n_features = 2
            mock_engine.fit_transform.return_value = mock_result
            mock_engine_class.return_value = mock_engine
            
            csv_output = os.path.join(tmpdir, 'output.csv')
            args = argparse.Namespace(
                input=csv_input,
                output=csv_output,
                smiles_column='smiles',
                preset='general'
            )
            
            with patch('builtins.print'):
                result = cli.cmd_extract(args)
            
            assert result == 0
            assert os.path.exists(csv_output)


class TestCmdTrain:
    """trainコマンドテスト"""
    
    @patch('core.services.utils.save_model')
    @patch('sklearn.model_selection.cross_val_score')
    @patch('core.services.features.SmartFeatureEngine')
    def test_train_rf_model(self, mock_engine_class, mock_cv_score, mock_save):
        """RFモデル学習"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # テストデータ作成
            csv_path = os.path.join(tmpdir, 'train.csv')
            pd.DataFrame({
                'smiles': ['CCO', 'CC', 'CCC'],
                'logP': [1.0, 2.0, 3.0]
            }).to_csv(csv_path, index=False)
            
            # モック設定
            mock_engine = MagicMock()
            mock_result = MagicMock()
            mock_result.features = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
            mock_engine.fit_transform.return_value = mock_result
            mock_engine_class.return_value = mock_engine
            
            mock_cv_score.return_value = np.array([0.8, 0.85, 0.9, 0.88, 0.82])
            mock_save.return_value = '/path/to/model.pkl'
            
            args = argparse.Namespace(
                input=csv_path,
                target='logP',
                model='rf',
                name='test_rf',
                preset='general'
            )
            
            with patch('builtins.print'):
                result = cli.cmd_train(args)
            
            assert result == 0
            mock_save.assert_called_once()
    
    @patch('core.services.features.SmartFeatureEngine')
    def test_train_missing_smiles_column(self, mock_engine_class):
        """SMILESカラムなしエラー"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'train.csv')
            pd.DataFrame({'mol': ['CCO'], 'logP': [1.0]}).to_csv(csv_path, index=False)
            
            args = argparse.Namespace(
                input=csv_path,
                target='logP',
                model='rf',
                name=None,
                preset='general'
            )
            
            with patch('builtins.print'):
                result = cli.cmd_train(args)
            
            assert result == 1
    
    @patch('core.services.features.SmartFeatureEngine')
    def test_train_missing_target_column(self, mock_engine_class):
        """ターゲットカラムなしエラー"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'train.csv')
            pd.DataFrame({'smiles': ['CCO']}).to_csv(csv_path, index=False)
            
            args = argparse.Namespace(
                input=csv_path,
                target='nonexistent',
                model='rf',
                name=None,
                preset='general'
            )
            
            with patch('builtins.print'):
                result = cli.cmd_train(args)
            
            assert result == 1


class TestCmdAnalyze:
    """analyzeコマンドテスト"""
    
    @patch('core.services.features.analyze_scaffolds')
    @patch('core.services.features.analyze_dataset')
    def test_analyze_dataset(self, mock_analyze_ds, mock_analyze_sc):
        """データセット分析"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'data.csv')
            pd.DataFrame({'smiles': ['CCO', 'CC']}).to_csv(csv_path, index=False)
            
            # モック設定
            mock_profile = MagicMock()
            mock_profile.n_valid = 2
            mock_profile.n_total = 2
            mock_profile.mw_mean = 30.5
            mock_profile.recommended_preset = 'general'
            mock_analyze_ds.return_value = mock_profile
            
            mock_scaffold = MagicMock()
            mock_scaffold.n_unique_scaffolds = 2
            mock_scaffold.scaffold_diversity = 1.0
            mock_analyze_sc.return_value = mock_scaffold
            
            args = argparse.Namespace(
                input=csv_path,
                smiles_column='smiles',
                output=None
            )
            
            with patch('builtins.print'):
                result = cli.cmd_analyze(args)
            
            assert result == 0


class TestCmdList:
    """listコマンドテスト"""
    
    @patch('core.services.features.list_presets')
    def test_list_presets(self, mock_list_presets):
        """プリセット一覧"""
        mock_list_presets.return_value = {
            'general': '一般化学物質',
            'drug': '医薬品'
        }
        
        args = argparse.Namespace(resource='presets')
        
        with patch('builtins.print'):
            result = cli.cmd_list(args)
        
        assert result == 0
    
    @patch('core.services.utils.ModelPersistence')
    def test_list_models(self, mock_mp_class):
        """モデル一覧"""
        mock_mp = MagicMock()
        mock_mp.list_models.return_value = ['model1', 'model2']
        mock_mp.list_versions.return_value = ['v1', 'v2']
        mock_mp_class.return_value = mock_mp
        
        args = argparse.Namespace(resource='models')
        
        with patch('builtins.print'):
            result = cli.cmd_list(args)
        
        assert result == 0
    
    @patch('core.services.utils.ModelPersistence')
    def test_list_no_models(self, mock_mp_class):
        """モデルなし"""
        mock_mp = MagicMock()
        mock_mp.list_models.return_value = []
        mock_mp_class.return_value = mock_mp
        
        args = argparse.Namespace(resource='models')
        
        with patch('builtins.print'):
            result = cli.cmd_list(args)
        
        assert result == 0


class TestMain:
    """main関数テスト"""
    
    @patch('core.cli.cmd_list')
    @patch('sys.argv', ['cli.py', 'list', 'presets'])
    def test_main_list_command(self, mock_cmd):
        """mainからlistコマンド実行"""
        mock_cmd.return_value = 0
        
        result = cli.main()
        
        assert result == 0
        mock_cmd.assert_called_once()
    
    @patch('sys.argv', ['cli.py'])
    def test_main_no_command(self):
        """コマンドなし"""
        with patch('builtins.print'):
            result = cli.main()
        
        assert result == 0
