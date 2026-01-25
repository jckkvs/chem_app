"""
Tests for UMA Feature Extractor

Implements: T-UMA-001
Target: core/services/features/uma_eng.py
"""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.services.features.uma_eng import UMAFeatureExtractor

class TestUMAFeatureExtractor:
    def setup_method(self, method):
        self.smiles_list = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC']
        
        # Mock base extractor
        self.mock_base = MagicMock()
        # Returns simple dataframe: 5 samples, 2 features
        self.mock_base.transform.return_value = pd.DataFrame({
            'prop1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'prop2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self, method):
        shutil.rmtree(self.tmp_dir)

    def test_fit_transform(self):
        # Mock UMAP to avoid actual expensive computation
        with patch('core.services.features.uma_eng.umap.UMAP') as MockUMAP:
            mock_reducer = MagicMock()
            # transform returns (n_samples, n_components)
            mock_reducer.transform.return_value = np.zeros((5, 2)) 
            MockUMAP.return_value = mock_reducer
            
            extractor = UMAFeatureExtractor(n_components=2, base_extractor=self.mock_base)
            extractor.fit(self.smiles_list)
            
            assert extractor.is_fitted
            assert extractor.reducer is not None
            assert MockUMAP.called
            
            df = extractor.transform(self.smiles_list)
            assert df.shape == (5, 3) # SMILES + 2 components
            assert 'SMILES' in df.columns
            assert 'UMA_0' in df.columns

    def test_supervised_fit(self):
        with patch('core.services.features.uma_eng.umap.UMAP') as MockUMAP:
            mock_reducer = MagicMock()
            MockUMAP.return_value = mock_reducer
            
            extractor = UMAFeatureExtractor(n_components=2, base_extractor=self.mock_base)
            y = [0, 0, 1, 1, 1]
            extractor.fit(self.smiles_list, y=y)
            
            # Check if fit was called with y
            args, kwargs = mock_reducer.fit.call_args
            assert 'y' in kwargs or len(args) > 1

    def test_save_load(self):
        with patch('core.services.features.uma_eng.umap.UMAP') as MockUMAP, \
             patch('core.services.features.uma_eng.joblib') as mock_joblib, \
             patch('os.path.exists', return_value=True):  # Mock exists for load
            
            mock_reducer = MagicMock()
            MockUMAP.return_value = mock_reducer
            
            extractor = UMAFeatureExtractor(n_components=2, base_extractor=self.mock_base)
            extractor.fit(self.smiles_list)
            
            save_path = os.path.join(self.tmp_dir, 'uma_model.pkl')
            extractor.save(save_path)
            
            # Verify usage of joblib without actual pickling
            assert mock_joblib.dump.called
            args, _ = mock_joblib.dump.call_args
            data = args[0]
            assert 'reducer' in data
            assert data['reducer'] == mock_reducer
            
            # Setup mock load return
            mock_joblib.load.return_value = {
                'scaler': MagicMock(),
                'reducer': mock_reducer,
                'n_components': 2,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'feature_columns': ['f1', 'f2']
            }

            # Load
            new_extractor = UMAFeatureExtractor(n_components=2)
            new_extractor.load(save_path)
            
            assert new_extractor.is_fitted
            assert new_extractor.reducer is not None

    def test_transform_not_fitted(self):
        extractor = UMAFeatureExtractor(n_components=2)
        # Should return empty/zeros or handle gracefully
        # Implementation returns "empty" df with zeros
        df = extractor.transform(self.smiles_list)
        assert df.shape == (5, 3)
        assert (df['UMA_0'] == 0).all()

    def test_descriptor_names(self):
        extractor = UMAFeatureExtractor(n_components=3)
        names = extractor.descriptor_names
        assert len(names) == 3
        assert names[0] == 'UMA_0'
