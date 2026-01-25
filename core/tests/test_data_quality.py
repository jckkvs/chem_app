"""
Tests for Data Quality Module

Implements: T-DATAQUALITY-001
Target: core/services/ml/data_quality.py
"""

import numpy as np
import pandas as pd
import pytest
from core.services.ml.data_quality import DataQualityAnalyzer, QualityReport

class TestDataQualityAnalyzer:
    def test_analyze_perfect_data(self):
        """Test with clean data"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [5, 4, 3, 2, 1],
            'c': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)
        
        assert report.overall_score == 100
        assert report.missing_count == 0
        assert report.duplicate_rows == 0
        assert report.outlier_count == 0
        assert len(report.constant_columns) == 0

    def test_missing_values(self):
        """Test missing value detection"""
        df = pd.DataFrame({
            'a': [1, np.nan, 3, 4],
            'b': [1, 2, 3, 4]
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)
        
        assert report.missing_count == 1
        assert report.missing_percent == 0.125  # 1/8
        assert 'a' in report.missing_by_column
        assert report.overall_score < 100

    def test_duplicates(self):
        """Test duplicate row detection"""
        df = pd.DataFrame({
            'a': [1, 1, 2],
            'b': [1, 1, 2]
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)
        
        assert report.duplicate_rows == 1
        assert report.duplicate_percent > 0
        assert report.overall_score < 100

    def test_constant_columns(self):
        """Test constant column detection"""
        df = pd.DataFrame({
            'a': [1, 1, 1],
            'b': [1, 2, 3]
        })
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)
        
        assert 'a' in report.constant_columns
        assert report.overall_score < 100
        # Check recommendation
        assert any("情報量なし" in r for r in report.recommendations)

    def test_outliers_iqr(self):
        """Test outlier detection with IQR"""
        # Create data with clear outlier
        data = [1, 2, 3, 4, 5, 100]
        df = pd.DataFrame({'a': data})
        
        analyzer = DataQualityAnalyzer(outlier_method='iqr', outlier_threshold=1.5)
        report = analyzer.analyze(df)
        
        assert report.outlier_count > 0
        assert 'a' in report.outlier_by_column

    def test_outliers_zscore(self):
        """Test outlier detection with Z-score"""
        # Create data with clear outlier
        data = [1, 2, 3, 4, 5, 100]
        df = pd.DataFrame({'a': data})
        
        analyzer = DataQualityAnalyzer(outlier_method='zscore', outlier_threshold=2.0)
        report = analyzer.analyze(df)
        
        assert report.outlier_count > 0

    def test_high_correlation(self):
        """Test high correlation detection"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
        })
        df['b'] = df['a'] * 2  # Perfect correlation
        
        analyzer = DataQualityAnalyzer(correlation_threshold=0.99)
        report = analyzer.analyze(df)
        
        assert len(report.highly_correlated) > 0
        pair = report.highly_correlated[0]
        assert 'a' in pair and 'b' in pair
        assert pair[2] > 0.99

    def test_recommendations(self):
        """Test recommendation generation"""
        df = pd.DataFrame({
            'missing': [1, np.nan, np.nan, np.nan],  # > 50% missing
            'const': [1, 1, 1, 1],
        })
        
        analyzer = DataQualityAnalyzer(missing_threshold=0.5)
        report = analyzer.analyze(df)
        
        recs = " ".join(report.recommendations)
        assert "欠損率" in recs
        assert "情報量なし" in recs

    def test_html_report(self):
        """Test HTML generation"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze(df)
        html = analyzer.get_summary_html(report)
        
        assert "データ品質レポート" in html
        assert "100" in html  # Score

    def test_empty_dataframe(self):
        """Test edge case: empty dataframe"""
        df = pd.DataFrame()
        analyzer = DataQualityAnalyzer()
        # Should handle gracefully (might error depending on logic, let's see if it needs fix)
        # Looking at code: n_samples, n_features = df.shape. 
        # _analyze_missing uses df.size.
        # It handles division by zero checks?
        # missing: total_missing / total_cells if total_cells > 0 else 0. Good.
        # But _analyze_outliers loops columns.
        
        try:
            report = analyzer.analyze(df)
            assert report.n_samples == 0
        except Exception as e:
            pytest.fail(f"Empty dataframe caused error: {e}")
