"""
Tests for Smart Preprocessor

Implements: T-PREP-001
Target: core/services/ml/preprocessor.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from core.services.ml.preprocessor import (
    SmartPreprocessor, 
    ColumnTypeInfo,
    OutlierHandler,
    PreprocessorFactory
)

class TestSmartPreprocessor:
    def setup_method(self, method):
        # Create mixed type dataframe
        self.df = pd.DataFrame({
            'cont': [1.1, 2.2, 3.3, 100.0], # 100.0 is outlier
            'cat': ['a', 'b', 'a', 'c'],
            'binary': [0, 1, 0, 1],
            'int_count': [1, 2, 3, 1], # few unique ints
            'smiles': ['C', 'CC', 'CCC', 'CCCC']
        })
        self.y = pd.Series([1, 0, 1, 0])

    def test_detect_column_types(self):
        prep = SmartPreprocessor(categorical_threshold=5)
        info = prep.detect_column_types(self.df)
        
        assert 'cont' in info.continuous
        assert 'cat' in info.categorical
        assert 'binary' in info.binary
        # 'int_count' has 3 unique values (1,2,3). If cat_threshold=5, it depends on dtype.
        # if using defaults, logic check:
        # isnumeric -> unique <= binary_threshold(2) -> binary
        # else -> int64 & unique <= categorical_threshold -> integer_count
        # else -> continuous
        
        # 'int_count' is likely int64. unique=3. > binary(2). <= cat(5).
        assert 'int_count' in info.integer_count
        
        # SMILES should be skipped
        assert 'smiles' not in info.categorical
        assert 'smiles' not in info.continuous

    def test_fit_transform_default(self):
        prep = SmartPreprocessor()
        X_out = prep.fit_transform(self.df, self.y)
        
        # Check output shape
        # cont -> 1 col (scaler)
        # cat -> 3 unique -> onehot -> 3 cols
        # binary -> 1 col (passthrough or impute)
        # int_count -> 1 col (passthrough or impute)
        # smiles -> removed by default?
        # ColumnTransformer remainder='drop'. 
        
        # So expected cols: 1 + 3 + 1 + 1 = 6
        assert X_out.shape[1] == 6
        assert isinstance(X_out, pd.DataFrame)
        # assert 'cont' in X_out.columns or any(c.startswith('cont') for c in X_out.columns)

    def test_fit_transform_robust(self):
        # Robust preset uses outlier clipping/robust scaler
        prep = PreprocessorFactory.create('robust')
        X_out = prep.fit_transform(self.df)
        
        # Basic check that it runs
        assert X_out is not None

    def test_get_params_summary(self):
        prep = SmartPreprocessor()
        summary = prep.get_params_summary()
        assert '連続変数スケーラー' in summary

class TestOutlierHandler:
    def test_iqr_clipping(self):
        data = np.array([[1], [2], [3], [100]]) # 100 is outlier
        handler = OutlierHandler(method='iqr', iqr_factor=1.5)
        handler.fit(data)
        
        res = handler.transform(data)
        
        # 1,2,3 -> Q1=1.75, Q3=27.25 ?? No
        # percentile is over axis=0
        # 1, 2, 3, 100.
        # 25% = 1.75, 75% = 27.25
        # IQR = 25.5
        # Upper = 27.25 + 1.5*25.5 ~ 65.5
        # So 100 should be clipped to ~65.5
        
        assert res[3][0] < 100
        assert res[3][0] > 3
        
    def test_clip_method(self):
        data = np.array([[0], [50], [100]])
        handler = OutlierHandler(method='clip') # 1-99 percentile
        handler.fit(data)
        res = handler.transform(np.array([[-10], [200]]))
        
        # Should be clipped to 1-99% of training data
        assert res[0][0] >= handler.lower_bounds_[0]
        assert res[1][0] <= handler.upper_bounds_[0]
