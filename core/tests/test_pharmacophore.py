"""
Tests for Pharmacophore Generator

Implements: T-PHARMACOPHORE-001
Target: core/services/features/pharmacophore.py
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from core.services.features.pharmacophore import (
    PharmacophoreGenerator,
    Pharmacophore,
    PharmacophoreFeature
)

class TestPharmacophoreGenerator(unittest.TestCase):
    def test_feature_dataclass(self):
        """Test feature logic"""
        f1 = PharmacophoreFeature('donor', (0,0,0), [1])
        f2 = PharmacophoreFeature('acceptor', (3,0,0), [2])
        
        assert f1.distance_to(f2) == 3.0
        
    def test_pharmacophore_dataclass(self):
        """Test pharmacophore model logic"""
        f1 = PharmacophoreFeature('donor', (0,0,0), [1])
        f2 = PharmacophoreFeature('acceptor', (3,0,0), [2])
        f3 = PharmacophoreFeature('donor', (0,4,0), [3])
        
        pharm = Pharmacophore(smiles="TAG", features=[f1, f2, f3])
        
        counts = pharm.get_feature_counts()
        assert counts['donor'] == 2
        assert counts['acceptor'] == 1
        
        matrix = pharm.get_distance_matrix()
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 3.0
        assert matrix[0, 2] == 4.0
        assert matrix[1, 2] == 5.0 # 3-4-5 triangle

    def test_generate_success(self):
        """Test generation with successful RDKit calls"""
        gen = PharmacophoreGenerator()
        
        # Setup mocks
        mock_rdkit = MagicMock()
        mock_chem = MagicMock()
        mock_allchem = MagicMock()
        mock_mol = MagicMock()
        mock_conf = MagicMock()
        
        # Link hierarchy
        mock_rdkit.Chem = mock_chem
        mock_chem.AllChem = mock_allchem
        
        # Config behavior
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_chem.AddHs.return_value = mock_mol
        mock_chem.MolFromSmarts.return_value = MagicMock() # pattern
        
        mock_mol.GetConformer.return_value = mock_conf
        mock_mol.GetSubstructMatches.return_value = [(0,)]
        
        # Atom Position
        mock_point = MagicMock()
        mock_point.x, mock_point.y, mock_point.z = 1.0, 1.0, 1.0
        mock_conf.GetAtomPosition.return_value = mock_point
        
        # Patch sys.modules
        with patch.dict('sys.modules', {
            'rdkit': mock_rdkit,
            'rdkit.Chem': mock_chem,
            'rdkit.Chem.AllChem': mock_allchem
        }):
            pharm = gen.generate("CCO")
        
        assert pharm is not None
        assert pharm.smiles == "CCO"
        assert len(pharm.features) == 6
        assert mock_allchem.EmbedMolecule.called

    def test_generate_invalid_smiles(self):
        """Test invalid smiles"""
        mock_rdkit = MagicMock()
        mock_chem = MagicMock()
        mock_rdkit.Chem = mock_chem
        
        mock_chem.MolFromSmiles.return_value = None
        
        with patch.dict('sys.modules', {'rdkit': mock_rdkit, 'rdkit.Chem': mock_chem}):
            gen = PharmacophoreGenerator()
            pharm = gen.generate("INVALID")
        
        assert pharm is None

    def test_generate_exception(self):
        """Test exception handling during generation"""
        mock_rdkit = MagicMock()
        mock_chem = MagicMock()
        mock_rdkit.Chem = mock_chem
        
        mock_chem.MolFromSmiles.side_effect = Exception("Boom")
        
        with patch.dict('sys.modules', {'rdkit': mock_rdkit, 'rdkit.Chem': mock_chem}):
            gen = PharmacophoreGenerator()
            pharm = gen.generate("CCO")
        
        assert pharm is None

    def test_fingerprint_generation(self):
        """Test fingerprint logic"""
        gen = PharmacophoreGenerator()
        
        f1 = PharmacophoreFeature('donor', (0,0,0), [1])
        f2 = PharmacophoreFeature('acceptor', (3,0,0), [2])
        pharm = Pharmacophore("SMI", [f1, f2])
        
        fp = gen.get_fingerprint(pharm, n_bits=256)
        
        assert fp.shape == (256,)
        # Check bits set. Donor is idx 0 -> bits 0-9. Count is 1.
        assert fp[0] == 1
        # Acceptor is idx 1 -> bits 10-19. Count is 1.
        assert fp[10] == 1
        
        # Distance 3.0 -> 1.5 -> int 1. offset 60.
        # fp[60 + 0] = 1 (distance in bin 1)
        # Note: logic: fp[60 + i] = int(d / 2) % 10. 
        # i=0 (first pair). d=3.0. int(1.5)=1. 1%10=1.
        assert fp[60] == 1

    def test_similarity(self):
        """Test similarity calculation"""
        gen = PharmacophoreGenerator()
        
        f1 = PharmacophoreFeature('donor', (0,0,0), [1])
        pharm1 = Pharmacophore("SMI1", [f1])
        
        # Identical
        sim = gen.similarity(pharm1, pharm1)
        assert abs(sim - 1.0) < 1e-5
        
        # Orthogonal (different feature types)
        f2 = PharmacophoreFeature('acceptor', (10,10,10), [2])
        pharm2 = Pharmacophore("SMI2", [f2])
        
        # fp1 will have donor bits set, fp2 will have acceptor bits set.
        # Dot product should be 0 (assuming no overlap in other bits)
        # However, distances section might be 0 for both if only 1 feature (no pairs).
        sim = gen.similarity(pharm1, pharm2)
        assert sim == 0.0

    def test_similarity_zero_norm(self):
        """Test zero norm handling"""
        gen = PharmacophoreGenerator()
        pharm1 = Pharmacophore("SMI1", [])
        pharm2 = Pharmacophore("SMI2", [])
        
        sim = gen.similarity(pharm1, pharm2)
        assert sim == 0.0
