"""
TARTE特徴量抽出器のテスト

テスト方針:
- tarte-ai未インストール時: フォールバック動作を確認
- tarte-aiインストール時: 各モードの動作を確認
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from django.test import SimpleTestCase

from core.services.features.tarte_eng import (
    TarteFeatureExtractor,
    is_tarte_available,
    _check_tarte_available,
)


class TarteAvailabilityTests(SimpleTestCase):
    """tarte-ai利用可能性チェックのテスト"""
    
    def test_is_tarte_available_function(self):
        """is_tarte_available関数が例外を投げないこと"""
        result = is_tarte_available()
        self.assertIsInstance(result, bool)
    
    def test_check_tarte_available_returns_bool(self):
        """_check_tarte_availableがboolを返すこと"""
        result = _check_tarte_available()
        self.assertIsInstance(result, bool)


class TarteFeatureExtractorFallbackTests(SimpleTestCase):
    """tarte-ai未インストール時のフォールバックテスト"""
    
    @patch('core.services.features.tarte_eng._TARTE_AVAILABLE', False)
    def test_init_without_tarte(self):
        """tarte-aiなしでも初期化できること"""
        with patch('core.services.features.tarte_eng._check_tarte_available', return_value=False):
            extractor = TarteFeatureExtractor(mode="featurizer")
            self.assertFalse(extractor.is_available)
    
    @patch('core.services.features.tarte_eng._TARTE_AVAILABLE', False)
    def test_transform_returns_nan_without_tarte(self):
        """tarte-aiなしでNaN埋めのDataFrameを返すこと"""
        with patch('core.services.features.tarte_eng._check_tarte_available', return_value=False):
            extractor = TarteFeatureExtractor(mode="featurizer", embedding_dim=10)
            extractor._tarte_available = False
            
            df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            extractor.fit(df)
            result = extractor.transform(df)
            
            self.assertEqual(len(result), 3)
            self.assertEqual(result.shape[1], 10)  # embedding_dim
            self.assertTrue(result.isna().all().all())  # すべてNaN


class TarteFeatureExtractorModeTests(SimpleTestCase):
    """モード設定のテスト"""
    
    def test_valid_modes(self):
        """有効なモードが受け入れられること"""
        for mode in ["featurizer", "finetuning", "boosting"]:
            with patch('core.services.features.tarte_eng._check_tarte_available', return_value=False):
                extractor = TarteFeatureExtractor(mode=mode)
                self.assertEqual(extractor.mode, mode)
    
    def test_invalid_mode_raises_error(self):
        """無効なモードでValueErrorが発生すること"""
        with self.assertRaises(ValueError):
            TarteFeatureExtractor(mode="invalid_mode")
    
    def test_get_params(self):
        """get_paramsがパラメータを返すこと"""
        with patch('core.services.features.tarte_eng._check_tarte_available', return_value=False):
            extractor = TarteFeatureExtractor(
                mode="finetuning",
                n_epochs=20,
                batch_size=64,
            )
            params = extractor.get_params()
            
            self.assertEqual(params["mode"], "finetuning")
            self.assertEqual(params["n_epochs"], 20)
            self.assertEqual(params["batch_size"], 64)


class TarteFeatureExtractorIntegrationTests(SimpleTestCase):
    """統合テスト（tarte-aiがある場合のみ実行）"""
    
    @unittest.skipIf(not is_tarte_available(), "tarte-ai not installed")
    def test_featurizer_mode_with_real_tarte(self):
        """Featurizerモードの実際の動作テスト"""
        extractor = TarteFeatureExtractor(mode="featurizer")
        
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        
        result = extractor.fit_transform(df)
        
        self.assertEqual(len(result), 5)
        self.assertFalse(result.isna().all().all())  # NaNだけではない
    
    @unittest.skipIf(not is_tarte_available(), "tarte-ai not installed")
    def test_finetuning_mode_with_real_tarte(self):
        """Finetuningモードの実際の動作テスト"""
        extractor = TarteFeatureExtractor(
            mode="finetuning",
            n_epochs=1,  # テスト用に最小
            target_column="target",
        )
        
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5] * 10,  # 50サンプル
            "col2": [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
            "target": [10, 20, 30, 40, 50] * 10,
        })
        
        result = extractor.fit_transform(df)
        
        self.assertEqual(len(result), 50)
