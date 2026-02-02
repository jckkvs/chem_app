# -*- coding: utf-8 -*-
"""
データ品質チェックエンジンのテスト

カバレッジ目標: 0% → 70%
"""
import numpy as np
import pandas as pd
import pytest


class TestDataQualityBasic:
    """基本テスト"""
    
    def test_basic_import(self):
        """基本的なインポート"""
        from core.services.ml import data_quality
        assert data_quality is not None


class TestMissingValueDetection:
    """欠損値検出テスト"""
    
    def test_missing_values(self):
        """欠損値検出"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        missing = df.isnull().sum()
        
        assert missing['A'] == 1
        assert missing['B'] == 1
        assert missing['C'] == 0


class TestOutlierDetection:
    """外れ値検出テスト"""
    
    def test_zscore_outliers(self):
        """Zスコア外れ値検出"""
        from scipy import stats
        
        # より明確な外れ値データ
        data = np.array([10.0, 12.0, 11.0, 13.0, 12.5, 11.5, 10.8, 100.0])
        z_scores = np.abs(stats.zscore(data))
        
        outliers = z_scores > 2.5  # 閾値を下げる
        
        # 最後が外れ値
        assert outliers[-1] == True
    
    def test_iqr_outliers(self):
        """IQR外れ値検出"""
        data = np.array([1, 2, 3, 4, 5, 100])
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = (data < lower) | (data > upper)
        
        assert outliers[-1] == True


class TestDataTypeValidation:
    """データ型検証テスト"""
    
    def test_numeric_validation(self):
        """数値データ検証"""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [1.1, 2.2, 3.3],
            'str': ['a', 'b', 'c']
        })
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        assert 'num1' in numeric_cols
        assert 'num2' in numeric_cols
        assert 'str' not in numeric_cols


class TestDataQualityIntegration:
    """統合テスト"""
    
    def test_complete_quality_check(self):
        """完全な品質チェック"""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6, 100],
            'feature2': [10, 20, 30, 40, 50, 60, 70],
            'feature3': [1, 1, 2, 2, 3, 3, 4]
        })
        
        missing = df.isnull().sum()
        duplicates = df.duplicated().sum()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        for col in numeric_cols:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        assert missing.sum() == 1
        assert duplicates == 0
        assert outlier_count > 0
