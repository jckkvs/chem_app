from django.test import SimpleTestCase
import pandas as pd
import matplotlib.pyplot as plt
from core.services.vis.shap_eng import SHAPEngine


# File path was core/services/vis/plots.py, so class is in core.services.vis.plots 
# Wait, I wrote `core/services/vis/plots.py`.
from core.services.vis.plots import PlotEngine
from core.services.vis.pdp_eng import PDPEngine
from unittest.mock import MagicMock, patch

class VisTests(SimpleTestCase):
    def test_plot_engine(self):
        history = [{'train_loss': 0.1, 'val_loss': 0.2}, {'train_loss': 0.05, 'val_loss': 0.1}]
        fig = PlotEngine.plot_training_curves(history)
        self.assertIsNotNone(fig)
        plt.close(fig)

    @patch('core.services.vis.pdp_eng.PartialDependenceDisplay')
    def test_pdp_engine(self, mock_pdd):
        eng = PDPEngine()
        # Mock estimator
        model = MagicMock()
        X = pd.DataFrame({'a': [1,2], 'b': [3,4]})
        fig = eng.plot_pdp(model, X, ['a'])
        self.assertIsNotNone(fig)
        plt.close(fig)

    @patch('core.services.vis.shap_eng.shap.TreeExplainer')
    @patch('core.services.vis.shap_eng.shap.summary_plot')
    def test_shap_engine(self, mock_plot, mock_explainer):
        mock_ex_inst = MagicMock()
        mock_explainer.return_value = mock_ex_inst
        mock_ex_inst.shap_values.return_value = [[0.1, 0.2]]
        
        eng = SHAPEngine()
        vals, ex = eng.explain(MagicMock(), pd.DataFrame({'a': [1]}))
        self.assertIsNotNone(vals)
        
        fig = eng.plot_summary(vals, pd.DataFrame({'a': [1]}))
        self.assertIsNotNone(fig)
        plt.close(fig)
