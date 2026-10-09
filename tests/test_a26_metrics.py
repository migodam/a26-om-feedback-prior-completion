import unittest
import numpy as np
import torch
from a26.metrics import image_metrics,ssim_delta
from a26.training import ssim_delta as training_ssim

class MetricsTests(unittest.TestCase):
    def test_offline_ssim_matches_training_loss(self):
        rng=np.random.default_rng(2606)
        target=rng.random(576)+.2j*rng.random(576)
        prediction=target+.03*(rng.standard_normal(576)+1j*rng.standard_normal(576))
        actual=ssim_delta(prediction,target)
        expected=training_ssim(torch.tensor(prediction[None]),torch.tensor(target[None])).item()
        self.assertAlmostEqual(actual,expected,places=12)

    def test_exact_image_and_passivity(self):
        bg=np.full(576,.1+.04j);truth=bg.copy();truth[120:170]+=.6+.12j
        result=image_metrics(truth,truth,bg)
        self.assertEqual(result['delta_nrmse'],0.)
        self.assertEqual(result['support_iou'],1.)
        self.assertEqual(result['edge_f1'],1.)
        self.assertTrue(result['passive'])

    def test_undefined_material_denominator_is_not_floored(self):
        bg=np.full(576,.1+.04j)
        with self.assertRaises(ValueError):image_metrics(bg,bg,bg)
