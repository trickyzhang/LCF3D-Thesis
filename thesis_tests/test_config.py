import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('config_utils', Path(__file__).resolve().parents[1] / 'src/config_utils.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
image_score_thresholds = module.image_score_thresholds


class ThresholdTests(unittest.TestCase):
    def test_flat_configuration_is_respected(self):
        self.assertEqual(image_score_thresholds({'num_classes': 3, 'score_thr_2d': [.4, .4, .5]}), [.4, .4, .5])

    def test_nested_configuration_takes_priority(self):
        self.assertEqual(image_score_thresholds({'num_classes': 3, 'score_thr_2d': .4,
                                                'detector2d': {'score_thr': .2}}), [.2] * 3)

    def test_scalar_and_default(self):
        self.assertEqual(image_score_thresholds({'classes': ['a', 'b', 'c']}), [.5] * 3)
        self.assertEqual(image_score_thresholds({'num_classes': 2, 'score_thr_2d': .3}), [.3] * 2)

    def test_invalid_thresholds(self):
        for value in ([.2, .3], float('nan'), 1.2):
            with self.assertRaises(ValueError):
                image_score_thresholds({'num_classes': 3, 'score_thr_2d': value})
