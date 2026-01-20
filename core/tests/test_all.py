"""
テストスイート

全モジュールのユニットテスト
"""

import unittest
import sys


class TestFeatures(unittest.TestCase):
    """特徴量モジュールテスト"""
    
    def test_rdkit_import(self):
        from core.services.features.rdkit_eng import RDKitFeatureExtractor
        extractor = RDKitFeatureExtractor()
        self.assertIsNotNone(extractor)
    
    def test_mol_property(self):
        from core.services.features.mol_property import MolecularPropertyCalculator
        calc = MolecularPropertyCalculator()
        props = calc.calculate("CCO")
        self.assertIsNotNone(props)
        self.assertAlmostEqual(props.molecular_weight, 46.07, places=1)
    
    def test_fingerprint(self):
        from core.services.features.fingerprint import FingerprintCalculator
        calc = FingerprintCalculator()
        fp = calc.calculate("CCO")
        self.assertIsNotNone(fp)
        self.assertEqual(len(fp), 2048)
    
    def test_solubility(self):
        from core.services.features.solubility import SolubilityPredictor
        pred = SolubilityPredictor()
        result = pred.predict("CCO")
        self.assertIsNotNone(result)
        self.assertIn(result.solubility_class, ['high', 'medium', 'low', 'insoluble'])
    
    def test_admet(self):
        from core.services.features.admet import ADMETPredictor
        pred = ADMETPredictor()
        result = pred.predict("CCO")
        self.assertIsNotNone(result)
        self.assertGreater(result.hia, 0)
    
    def test_toxicity(self):
        from core.services.features.toxicity import ToxicityPredictor
        pred = ToxicityPredictor()
        result = pred.predict("c1ccc(N)cc1")  # アニリン
        self.assertIsNotNone(result)
        self.assertGreater(len(result.alerts), 0)


class TestML(unittest.TestCase):
    """機械学習モジュールテスト"""
    
    def test_uncertainty(self):
        from core.services.ml.uncertainty import UncertaintyQuantifier
        uq = UncertaintyQuantifier()
        self.assertIsNotNone(uq)
    
    def test_ensemble(self):
        from core.services.ml.ensemble import EnsembleModel
        ens = EnsembleModel()
        self.assertIsNotNone(ens)
    
    def test_active_learning(self):
        from core.services.ml.active_learning import ActiveLearner
        al = ActiveLearner()
        self.assertIsNotNone(al)
    
    def test_clustering(self):
        from core.services.ml.clustering import MolecularClusterer
        clusterer = MolecularClusterer()
        self.assertIsNotNone(clusterer)


class TestVis(unittest.TestCase):
    """可視化モジュールテスト"""
    
    def test_dashboard(self):
        from core.services.vis.dashboard import ExperimentDashboard
        dash = ExperimentDashboard()
        dash.add_experiment("test", {"r2": 0.9})
        html = dash.generate_html()
        self.assertIn("test", html)
    
    def test_experiment_compare(self):
        from core.services.vis.experiment_compare import ExperimentComparator
        comp = ExperimentComparator()
        comp.add_experiment("exp1", {"r2": 0.9})
        comp.add_experiment("exp2", {"r2": 0.85})
        html = comp.generate_comparison()
        self.assertIn("exp1", html)


class TestConfig(unittest.TestCase):
    """設定モジュールテスト"""
    
    def test_config_manager(self):
        from core.services.config import ConfigManager, AppConfig
        config = ConfigManager.get_default()
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.model.model_type, "lightgbm")
    
    def test_config_validate(self):
        from core.services.config import ConfigManager
        config = ConfigManager.get_default()
        self.assertTrue(ConfigManager.validate(config))


def run_tests():
    """テスト実行"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestML))
    suite.addTests(loader.loadTestsFromTestCase(TestVis))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
