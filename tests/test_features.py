import unittest

import numpy as np

from tiara.features import featurize_block
from tiara.hierarchical.biosignal_multi_expert import biosignal_features


class FeatureTest(unittest.TestCase):
    def test_tfidf_shape_and_norm(self):
        matrix = featurize_block(["ACGTACGT"], 2, np.ones(16, dtype=np.float32))
        self.assertEqual(matrix.shape, (1, 16))
        self.assertAlmostEqual(float(np.linalg.norm(matrix[0])), 1.0, places=6)

    def test_biosignal_dimension(self):
        self.assertEqual(biosignal_features("ACGT" * 300).shape, (60,))


if __name__ == "__main__":
    unittest.main()
