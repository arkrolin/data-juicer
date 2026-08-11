import unittest

import numpy as np
import torch
import torch.distributions as td

from data_juicer.analysis.measure import (
    CrossEntropyMeasure,
    EntropyMeasure,
    JSDivMeasure,
    KLDivMeasure,
    Measure,
    RelatedTTestMeasure,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MeasureConvertTest(DataJuicerTestCaseBase):
    """Supplemental tests for Measure base class conversion methods."""

    def setUp(self):
        super().setUp()
        self.measure = Measure()

    def test_convert_to_tensor_from_list(self):
        result = self.measure._convert_to_tensor([0.1, 0.2, 0.3, 0.4])
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (4,))
        self.assertAlmostEqual(result[0].item(), 0.1, places=5)

    def test_convert_to_tensor_from_tuple(self):
        result = self.measure._convert_to_tensor((1.0, 2.0, 3.0))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3,))
        self.assertAlmostEqual(result[1].item(), 2.0, places=5)

    def test_convert_to_tensor_passthrough(self):
        t = torch.tensor([0.5, 0.5])
        result = self.measure._convert_to_tensor(t)
        self.assertTrue(result is t)

    def test_convert_to_tensor_from_categorical(self):
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        cat = td.Categorical(probs)
        result = self.measure._convert_to_tensor(cat)
        self.assertTrue(torch.allclose(result, cat.probs))

    def test_convert_to_categorical_from_list(self):
        result = self.measure._convert_to_categorical([0.3, 0.7])
        self.assertIsInstance(result, td.Categorical)
        self.assertEqual(result.probs.shape, (2,))

    def test_convert_to_categorical_from_tensor(self):
        t = torch.tensor([0.4, 0.6])
        result = self.measure._convert_to_categorical(t)
        self.assertIsInstance(result, td.Categorical)
        self.assertTrue(torch.allclose(result.probs, t / t.sum()))

    def test_convert_to_categorical_passthrough(self):
        cat = td.Categorical(torch.tensor([0.5, 0.5]))
        result = self.measure._convert_to_categorical(cat)
        self.assertTrue(result is cat)

    def test_convert_to_ndarray_from_list(self):
        result = self.measure._convert_to_ndarray([1.0, 2.0, 3.0])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_convert_to_ndarray_from_tuple(self):
        result = self.measure._convert_to_ndarray((4.0, 5.0))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2,))


class KLDivMeasureSupplementalTest(DataJuicerTestCaseBase):
    """Supplemental edge-case tests for KLDivMeasure."""

    def test_uniform_vs_uniform_is_zero(self):
        uniform = [0.25, 0.25, 0.25, 0.25]
        measure = KLDivMeasure()
        result = measure(uniform, uniform)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_skewed_vs_uniform_positive(self):
        uniform = [0.25, 0.25, 0.25, 0.25]
        skewed = [0.9, 0.05, 0.025, 0.025]
        measure = KLDivMeasure()
        result = measure(skewed, uniform)
        self.assertGreater(result.item(), 0.0)

    def test_kl_not_symmetric(self):
        p = [0.1, 0.9]
        q = [0.5, 0.5]
        measure = KLDivMeasure()
        kl_pq = measure(p, q).item()
        kl_qp = measure(q, p).item()
        # KL divergence is generally not symmetric
        self.assertNotAlmostEqual(kl_pq, kl_qp, places=3)


class JSDivMeasureSupplementalTest(DataJuicerTestCaseBase):
    """Supplemental tests for JSDivMeasure."""

    def test_symmetric(self):
        p = [0.1, 0.2, 0.3, 0.4]
        q = [0.4, 0.3, 0.2, 0.1]
        measure = JSDivMeasure()
        js_pq = measure(p, q).item()
        js_qp = measure(q, p).item()
        self.assertAlmostEqual(js_pq, js_qp, places=5)

    def test_same_dist_is_zero(self):
        p = [0.2, 0.3, 0.5]
        measure = JSDivMeasure()
        result = measure(p, p)
        self.assertAlmostEqual(result.item(), 0.0, places=5)

    def test_js_non_negative(self):
        p = [0.6, 0.4]
        q = [0.3, 0.7]
        measure = JSDivMeasure()
        result = measure(p, q)
        self.assertGreaterEqual(result.item(), 0.0)


class CrossEntropyMeasureSupplementalTest(DataJuicerTestCaseBase):
    """Supplemental tests for CrossEntropyMeasure."""

    def test_cross_entropy_geq_entropy(self):
        """Cross-entropy H(p,q) >= H(p) for any q != p."""
        p = [0.1, 0.2, 0.3, 0.4]
        q = [0.4, 0.3, 0.2, 0.1]
        ce_measure = CrossEntropyMeasure()
        e_measure = EntropyMeasure()
        ce = ce_measure(p, q).item()
        entropy = e_measure(p).item()
        self.assertGreater(ce, entropy)

    def test_cross_entropy_same_dist_equals_entropy(self):
        """Cross-entropy H(p,p) == H(p)."""
        p = [0.1, 0.2, 0.3, 0.4]
        ce_measure = CrossEntropyMeasure()
        e_measure = EntropyMeasure()
        ce = ce_measure(p, p).item()
        entropy = e_measure(p).item()
        self.assertAlmostEqual(ce, entropy, places=4)


class EntropyMeasureSupplementalTest(DataJuicerTestCaseBase):
    """Supplemental tests for EntropyMeasure."""

    def test_uniform_has_max_entropy(self):
        """Uniform distribution maximizes entropy for a given number of classes."""
        uniform = [0.25, 0.25, 0.25, 0.25]
        skewed = [0.7, 0.1, 0.1, 0.1]
        measure = EntropyMeasure()
        h_uniform = measure(uniform).item()
        h_skewed = measure(skewed).item()
        self.assertGreater(h_uniform, h_skewed)

    def test_peaked_has_low_entropy(self):
        """A very peaked distribution has near-zero entropy."""
        peaked = [0.97, 0.01, 0.01, 0.01]
        measure = EntropyMeasure()
        h = measure(peaked).item()
        # Entropy of a near-certain distribution should be close to 0
        self.assertLess(h, 0.3)

    def test_entropy_non_negative(self):
        p = [0.5, 0.3, 0.2]
        measure = EntropyMeasure()
        h = measure(p).item()
        self.assertGreaterEqual(h, 0.0)


class RelatedTTestMeasureSupplementalTest(DataJuicerTestCaseBase):
    """Supplemental tests for RelatedTTestMeasure."""

    def test_stats_to_hist_output_shape(self):
        p = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        q = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        hist_p, hist_q, bin_edges = RelatedTTestMeasure.stats_to_hist(p, q)
        # Number of bins should be at least 10
        self.assertGreaterEqual(len(hist_p), 10)
        self.assertEqual(len(hist_p), len(hist_q))
        # bin_edges has one more element than hist
        self.assertEqual(len(bin_edges), len(hist_p) + 1)

    def test_stats_to_hist_bin_edges_cover_range(self):
        p = [0.5, 1.5, 2.5]
        q = [3.0, 4.0, 5.0]
        _, _, bin_edges = RelatedTTestMeasure.stats_to_hist(p, q)
        self.assertLessEqual(bin_edges[0], 0.5)
        self.assertGreaterEqual(bin_edges[-1], 5.0)

    def test_category_to_hist_counts(self):
        p = ['a', 'a', 'b', 'c']
        q = ['a', 'b', 'b', 'd']
        hist_p, hist_q, count_p, count_q, sorted_cat = \
            RelatedTTestMeasure.category_to_hist(p, q)
        # counts should sum to total items
        self.assertEqual(sum(hist_p), len(p))
        self.assertEqual(sum(hist_q), len(q))
        # all categories present
        self.assertEqual(set(count_p.keys()), {'a', 'b', 'c', 'd'})
        self.assertEqual(count_p['a'], 2)
        self.assertEqual(count_q['b'], 2)

    def test_measure_continuous_returns_ttest(self):
        p = list(range(100))
        q = [x + 0.5 for x in range(100)]
        measure = RelatedTTestMeasure()
        res = measure(p, q)
        # ttest result has statistic and pvalue attributes
        self.assertTrue(hasattr(res, 'statistic'))
        self.assertTrue(hasattr(res, 'pvalue'))

    def test_measure_discrete_returns_ttest(self):
        p = ['cat', 'dog', 'cat', 'bird', 'cat']
        q = ['dog', 'dog', 'cat', 'bird', 'fish']
        measure = RelatedTTestMeasure()
        res = measure(p, q)
        self.assertTrue(hasattr(res, 'statistic'))
        self.assertTrue(hasattr(res, 'pvalue'))

    def test_measure_nested_lists_for_categories(self):
        p = [['a', 'b'], ['c'], 'a', ['d', ['e']]]
        q = ['a', 'b', ['c', 'd'], 'e']
        measure = RelatedTTestMeasure()
        res = measure(p, q)
        self.assertTrue(hasattr(res, 'statistic'))
        self.assertTrue(hasattr(res, 'pvalue'))


if __name__ == '__main__':
    unittest.main()
