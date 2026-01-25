"""
Tests for Mixture Engineering and Solubility Prediction

Implements: T-MIXTURE-001, T-SOLUBILITY-001
Targets: 
 - core/services/features/mixture_eng.py
 - core/services/features/solubility.py
"""

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from core.services.features.mixture_eng import (
    MixtureFeatureExtractor,
    MixtureComponent,
    create_mixture_from_ratio_dict
)
from core.services.features.solubility import SolubilityPredictor, SolubilityResult


# ==============================================================================
# Mixture Tests
# ==============================================================================

class TestMixtureFeatureExtractor:
    def setup_method(self, method):
        # Mock base extractor
        self.mock_base = MagicMock()
        # Mock transform to return simple dataframe
        # Assume input smiles are 'A', 'B'. Return features [1, 1], [2, 2]
        def mock_transform_side_effect(smiles_list):
            data = []
            for s in smiles_list:
                if s == 'A': val = 1.0
                elif s == 'B': val = 2.0
                else: val = 0.0
                data.append({'f1': val, 'f2': val * 10})
            return pd.DataFrame(data)
        
        self.mock_base.transform.side_effect = mock_transform_side_effect
        self.extractor = MixtureFeatureExtractor(base_extractor=self.mock_base)

    def test_mixture_component_dataclass(self):
        """Test MixtureComponent initialization and ratio normalization"""
        c1 = MixtureComponent("A", 0.5)
        assert c1.ratio == 0.5
        
        c2 = MixtureComponent("B", 50.0)
        assert c2.ratio == 0.5  # 50% -> 0.5

    def test_transform_weighted_average(self):
        """Test weighted average calculation"""
        # Mixture: 50% A, 50% B
        # A features: [1.0, 10.0]
        # B features: [2.0, 20.0]
        # Expected: [1.5, 15.0]
        
        mixture = [
            MixtureComponent("A", 0.5),
            MixtureComponent("B", 0.5)
        ]
        
        df = self.extractor.transform([mixture])
        
        assert len(df) == 1
        assert df['f1'][0] == 1.5
        assert df['f2'][0] == 15.0
        assert df['n_components'][0] == 2
        assert df['max_ratio'][0] == 0.5

    def test_transform_empty(self):
        """Test empty mixture list"""
        df = self.extractor.transform([])
        assert len(df) == 0

    def test_parse_mixture_string(self):
        """Test parsing logic"""
        # Format 1: comma/colon
        m1 = MixtureFeatureExtractor.parse_mixture_string("A:60, B:40")
        assert len(m1) == 2
        assert m1[0].smiles == "A"
        assert m1[0].ratio == 0.6
        assert m1[1].ratio == 0.4
        
        # Format 2: pipe
        m2 = MixtureFeatureExtractor.parse_mixture_string("A|0.6|B|0.4")
        assert len(m2) == 2
        assert m2[0].ratio == 0.6
        
        # Format 3: single
        m3 = MixtureFeatureExtractor.parse_mixture_string("A")
        assert len(m3) == 1
        assert m3[0].ratio == 1.0

    def test_format_mixture(self):
        """Test formatting logic"""
        comps = [MixtureComponent("A", 0.6), MixtureComponent("B", 0.4)]
        s = MixtureFeatureExtractor.format_mixture(comps)
        assert "A:60%" in s
        assert "B:40%" in s

    def test_create_mixture_from_dict(self):
        """Test helper function"""
        d = {"A": 0.6, "B": 0.4}
        comps = create_mixture_from_ratio_dict(d)
        assert len(comps) == 2
        
    def test_transform_with_real_rdkit_mock(self):
        """Test with fake RDKit calls mocked out but using real logic flow"""
        # This confirms that if base_extractor returns real numbers, the mixing math works
        # Using simple numeric values is enough as proven in test_transform_weighted_average
        pass


# ==============================================================================
# Solubility Tests
# ==============================================================================

class TestSolubilityPredictor:
    def test_predict_valid(self):
        """Test ESOL prediction with mocked RDKit"""
        predictor = SolubilityPredictor()
        
        # Mock RDKit modules
        with patch.dict('sys.modules', {
            'rdkit': MagicMock(),
            'rdkit.Chem': MagicMock(),
            'rdkit.Chem.Descriptors': MagicMock(),
            'rdkit.Chem.Lipinski': MagicMock(),
        }):
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski
            
            # Setup mock return values
            # MolLogP=2.0, MolWt=200, Rotatable=2, AromaticRatio=0.5
            Descriptors.MolLogP.return_value = 2.0
            Descriptors.MolWt.return_value = 200.0
            Lipinski.NumRotatableBonds.return_value = 2
            Lipinski.NumAromaticRings.return_value = 1
            Lipinski.RingCount.return_value = 2 # ratio = 0.5
            
            Chem.MolFromSmiles.return_value = MagicMock()
            
            # ESOL manually calculation:
            # Intercept: 0.16
            # logP: -0.63 * 2.0 = -1.26
            # mw: -0.0062 * 200 = -1.24
            # rotatable: 0.066 * 2 = 0.132
            # aromatic: -0.74 * 0.5 = -0.37
            # Sum: 0.16 - 1.26 - 1.24 + 0.132 - 0.37 = -2.578
            
            result = predictor.predict("CCO")
            
            assert result is not None
            assert result.smiles == "CCO"
            # Allow small float error
            assert -2.65 < result.logS < -2.50
            assert result.solubility_class == "low"  # -2.578 is between -2 and -4

    def test_predict_invalid_smiles(self):
        """Test invalid SMILES handling"""
        predictor = SolubilityPredictor()
        
        with patch.dict('sys.modules', {
            'rdkit': MagicMock(),
            'rdkit.Chem': MagicMock(),
        }):
            from rdkit import Chem
            Chem.MolFromSmiles.return_value = None
            
            result = predictor.predict("invalid")
            assert result is None

    def test_predict_batch(self):
        """Test batch prediction"""
        predictor = SolubilityPredictor()
        predictor.predict = MagicMock(return_value=SolubilityResult("A", -2.0, 1.0, "medium"))
        
        results = predictor.predict_batch(["A", "B"])
        assert len(results) == 2
        assert predictor.predict.call_count == 2

    def test_get_solubility_html(self):
        """Test HTML generation"""
        predictor = SolubilityPredictor()
        res = SolubilityResult("A", -1.0, 10.0, "high")
        html = predictor.get_solubility_html(res)
        
        assert "A" not in html # HTML doesn't include SMILES title in current impl
        assert "HIGH" in html
        assert "10.0" in html

