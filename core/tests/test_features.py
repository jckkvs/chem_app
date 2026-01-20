from django.test import SimpleTestCase
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from core.services.features.rdkit_eng import RDKitFeatureExtractor
from core.services.features.xtb_eng import XTBFeatureExtractor
from core.services.features.uma_eng import UMAFeatureExtractor

class RDKitTests(SimpleTestCase):
    def test_transform_benzene(self):
        eng = RDKitFeatureExtractor()
        df = eng.transform(['c1ccccc1'])
        self.assertEqual(len(df), 1)
        # Benzene MolWt ~ 78.11
        self.assertAlmostEqual(df['MolWt'][0], 78.11, delta=0.1)

class XTBTests(SimpleTestCase):
    @patch('core.services.features.xtb_eng.subprocess.run')
    @patch('core.services.features.xtb_eng.XTBFeatureExtractor._check_xtb_available')
    def test_xtb_flow(self, mock_check, mock_run):
        mock_check.return_value = True
        
        # Mock Output
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "| TOTAL ENERGY              -14.0 Eh   |\n| HOMO-LUMO GAP              5.0 eV   |"
        mock_run.return_value = mock_proc
        
        eng = XTBFeatureExtractor(xtb_path='mock_xtb')
        df = eng.transform(['c1ccccc1'])
        
        self.assertEqual(df['energy'][0], -14.0)
        self.assertEqual(df['homo_lumo_gap'][0], 5.0)

class UMATests(SimpleTestCase):
    def test_uma_fit_transform(self):
        # Synthetic descriptors (bypass RDKit for speed/simplicity in this unit test or use real)
        # We'll use real RDKit but small list
        smiles = ['C', 'CC', 'CCC', 'CCCC', 'CCCCC'] * 4 # 20 samples
        eng = UMAFeatureExtractor(n_components=2, n_neighbors=2, min_dist=0.0) # low neighbors for small data
        
        eng.fit(smiles)
        df_emb = eng.transform(smiles)
        
        self.assertEqual(df_emb.shape[1], 3) # 2 components + SMILES
        self.assertIn('UMA_0', df_emb.columns)
        self.assertIn('UMA_1', df_emb.columns)
