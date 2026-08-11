import unittest
import time

from data_juicer.core.monitor import Monitor
from data_juicer.core import NestedDataset
from data_juicer.utils.fingerprint_utils import (
    Hasher,
    generate_fingerprint,
    update_fingerprint,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# =====================================================================
# Monitor supplemental tests
# =====================================================================

class MonitorFieldsTest(DataJuicerTestCaseBase):
    """Test that Monitor class-level constants contain expected keys."""

    def test_dynamic_fields_contains_cpu_util(self):
        self.assertIn('CPU util.', Monitor.DYNAMIC_FIELDS)

    def test_dynamic_fields_contains_mem_util(self):
        self.assertIn('Mem. util.', Monitor.DYNAMIC_FIELDS)

    def test_dynamic_fields_contains_used_mem(self):
        self.assertIn('Used mem.', Monitor.DYNAMIC_FIELDS)

    def test_dynamic_fields_contains_free_mem(self):
        self.assertIn('Free mem.', Monitor.DYNAMIC_FIELDS)

    def test_dynamic_fields_contains_available_mem(self):
        self.assertIn('Available mem.', Monitor.DYNAMIC_FIELDS)

    def test_dynamic_fields_contains_gpu_fields(self):
        self.assertIn('GPU free mem.', Monitor.DYNAMIC_FIELDS)
        self.assertIn('GPU used mem.', Monitor.DYNAMIC_FIELDS)
        self.assertIn('GPU util.', Monitor.DYNAMIC_FIELDS)

    def test_dynamic_fields_is_set(self):
        self.assertIsInstance(Monitor.DYNAMIC_FIELDS, set)

    def test_dynamic_fields_count(self):
        self.assertEqual(len(Monitor.DYNAMIC_FIELDS), 8)


class MonitorCurrentResourcesFormatTest(DataJuicerTestCaseBase):
    """Test that monitor_current_resources returns expected format."""

    def test_returns_dict(self):
        result = Monitor.monitor_current_resources()
        self.assertIsInstance(result, dict)

    def test_contains_timestamp(self):
        result = Monitor.monitor_current_resources()
        self.assertIn('timestamp', result)
        self.assertIsInstance(result['timestamp'], float)

    def test_contains_cpu_count(self):
        result = Monitor.monitor_current_resources()
        self.assertIn('CPU count', result)
        self.assertGreater(result['CPU count'], 0)

    def test_contains_cpu_util(self):
        result = Monitor.monitor_current_resources()
        self.assertIn('CPU util.', result)
        self.assertGreaterEqual(result['CPU util.'], 0.0)
        self.assertLessEqual(result['CPU util.'], 1.0)

    def test_contains_memory_fields(self):
        result = Monitor.monitor_current_resources()
        self.assertIn('Total mem.', result)
        self.assertIn('Used mem.', result)
        self.assertIn('Free mem.', result)
        self.assertIn('Available mem.', result)
        self.assertIn('Mem. util.', result)

    def test_memory_values_positive(self):
        result = Monitor.monitor_current_resources()
        self.assertGreater(result['Total mem.'], 0)
        self.assertGreater(result['Used mem.'], 0)

    def test_mem_util_is_ratio(self):
        result = Monitor.monitor_current_resources()
        self.assertGreater(result['Mem. util.'], 0.0)
        self.assertLessEqual(result['Mem. util.'], 1.0)

    def test_contains_gpu_fields(self):
        result = Monitor.monitor_current_resources()
        self.assertIn('GPU total mem.', result)
        self.assertIn('GPU free mem.', result)
        self.assertIn('GPU used mem.', result)
        self.assertIn('GPU util.', result)


class MonitorAnalyzeSingleResourceUtilTest(DataJuicerTestCaseBase):
    """Test analyze_single_resource_util with synthetic sample data."""

    def _make_resource_util_dict(self, records):
        return {
            'time': 1.0,
            'sampling interval': 0.5,
            'resource': records,
        }

    def test_basic_analysis(self):
        records = [
            {'CPU util.': 0.2, 'Used mem.': 1000.0, 'Free mem.': 3000.0,
             'Available mem.': 3500.0, 'Mem. util.': 0.25},
            {'CPU util.': 0.4, 'Used mem.': 1200.0, 'Free mem.': 2800.0,
             'Available mem.': 3300.0, 'Mem. util.': 0.30},
            {'CPU util.': 0.6, 'Used mem.': 1400.0, 'Free mem.': 2600.0,
             'Available mem.': 3100.0, 'Mem. util.': 0.35},
        ]
        resource_util = self._make_resource_util_dict(records)
        result = Monitor.analyze_single_resource_util(resource_util)

        self.assertIn('resource_analysis', result)
        analysis = result['resource_analysis']
        self.assertIn('CPU util.', analysis)
        cpu = analysis['CPU util.']
        self.assertAlmostEqual(cpu['max'], 0.6)
        self.assertAlmostEqual(cpu['min'], 0.2)
        self.assertAlmostEqual(cpu['avg'], 0.4)

    def test_analysis_with_none_values(self):
        """Fields with None values should be skipped."""
        records = [
            {'CPU util.': 0.5, 'GPU free mem.': None, 'Used mem.': 100.0,
             'Free mem.': 200.0, 'Available mem.': 250.0, 'Mem. util.': 0.3},
            {'CPU util.': 0.7, 'GPU free mem.': None, 'Used mem.': 120.0,
             'Free mem.': 180.0, 'Available mem.': 230.0, 'Mem. util.': 0.4},
        ]
        resource_util = self._make_resource_util_dict(records)
        result = Monitor.analyze_single_resource_util(resource_util)
        analysis = result['resource_analysis']
        # GPU free mem. should not appear because all values are None
        self.assertNotIn('GPU free mem.', analysis)
        # CPU util. should be analyzed
        self.assertIn('CPU util.', analysis)

    def test_analysis_with_list_values(self):
        """GPU metrics can be lists (one value per GPU). They get flattened."""
        records = [
            {'CPU util.': 0.3, 'GPU free mem.': [1000.0, 2000.0],
             'Used mem.': 500.0, 'Free mem.': 1500.0,
             'Available mem.': 1800.0, 'Mem. util.': 0.25,
             'GPU used mem.': [500.0, 300.0], 'GPU util.': [0.4, 0.5]},
            {'CPU util.': 0.5, 'GPU free mem.': [900.0, 1800.0],
             'Used mem.': 600.0, 'Free mem.': 1400.0,
             'Available mem.': 1700.0, 'Mem. util.': 0.30,
             'GPU used mem.': [600.0, 400.0], 'GPU util.': [0.6, 0.7]},
        ]
        resource_util = self._make_resource_util_dict(records)
        result = Monitor.analyze_single_resource_util(resource_util)
        analysis = result['resource_analysis']
        self.assertIn('GPU free mem.', analysis)
        gpu_free = analysis['GPU free mem.']
        # Max should be 2000, min should be 900
        self.assertAlmostEqual(gpu_free['max'], 2000.0)
        self.assertAlmostEqual(gpu_free['min'], 900.0)

    def test_analysis_single_record(self):
        """Single record: max == min == avg."""
        records = [
            {'CPU util.': 0.5, 'Used mem.': 1000.0, 'Free mem.': 2000.0,
             'Available mem.': 2500.0, 'Mem. util.': 0.33},
        ]
        resource_util = self._make_resource_util_dict(records)
        result = Monitor.analyze_single_resource_util(resource_util)
        analysis = result['resource_analysis']
        cpu = analysis['CPU util.']
        self.assertAlmostEqual(cpu['max'], 0.5)
        self.assertAlmostEqual(cpu['min'], 0.5)
        self.assertAlmostEqual(cpu['avg'], 0.5)

    def test_original_dict_is_mutated(self):
        """analyze_single_resource_util mutates the input dict in place."""
        records = [
            {'CPU util.': 0.1, 'Used mem.': 500.0, 'Free mem.': 1500.0,
             'Available mem.': 1800.0, 'Mem. util.': 0.25},
        ]
        resource_util = self._make_resource_util_dict(records)
        result = Monitor.analyze_single_resource_util(resource_util)
        # result is the same object
        self.assertIs(result, resource_util)
        self.assertIn('resource_analysis', resource_util)


class MonitorAnalyzeResourceUtilListTest(DataJuicerTestCaseBase):
    """Test analyze_resource_util_list with multiple items."""

    def test_multiple_items(self):
        items = [
            {
                'time': 1.0,
                'sampling interval': 0.5,
                'resource': [
                    {'CPU util.': 0.2, 'Used mem.': 800.0, 'Free mem.': 1200.0,
                     'Available mem.': 1400.0, 'Mem. util.': 0.4},
                    {'CPU util.': 0.4, 'Used mem.': 900.0, 'Free mem.': 1100.0,
                     'Available mem.': 1300.0, 'Mem. util.': 0.45},
                ],
            },
            {
                'time': 2.0,
                'sampling interval': 0.5,
                'resource': [
                    {'CPU util.': 0.6, 'Used mem.': 1000.0, 'Free mem.': 1000.0,
                     'Available mem.': 1200.0, 'Mem. util.': 0.5},
                    {'CPU util.': 0.8, 'Used mem.': 1100.0, 'Free mem.': 900.0,
                     'Available mem.': 1100.0, 'Mem. util.': 0.55},
                ],
            },
        ]
        results = Monitor.analyze_resource_util_list(items)
        self.assertEqual(len(results), 2)
        # First item
        analysis0 = results[0]['resource_analysis']
        self.assertAlmostEqual(analysis0['CPU util.']['max'], 0.4)
        self.assertAlmostEqual(analysis0['CPU util.']['min'], 0.2)
        # Second item
        analysis1 = results[1]['resource_analysis']
        self.assertAlmostEqual(analysis1['CPU util.']['max'], 0.8)
        self.assertAlmostEqual(analysis1['CPU util.']['min'], 0.6)

    def test_empty_list(self):
        results = Monitor.analyze_resource_util_list([])
        self.assertEqual(results, [])

    def test_single_item_list(self):
        items = [
            {
                'time': 0.5,
                'sampling interval': 0.1,
                'resource': [
                    {'CPU util.': 0.9, 'Used mem.': 2000.0, 'Free mem.': 500.0,
                     'Available mem.': 600.0, 'Mem. util.': 0.8},
                ],
            },
        ]
        results = Monitor.analyze_resource_util_list(items)
        self.assertEqual(len(results), 1)
        self.assertIn('resource_analysis', results[0])


# =====================================================================
# Fingerprint utils supplemental tests
# =====================================================================

class HasherUpdateMultipleValuesTest(DataJuicerTestCaseBase):
    """Test Hasher with multiple update calls produces order-dependent hash."""

    def test_different_order_different_hash(self):
        h1 = Hasher()
        h1.update('a')
        h1.update('b')

        h2 = Hasher()
        h2.update('b')
        h2.update('a')

        self.assertNotEqual(h1.hexdigest(), h2.hexdigest())

    def test_same_order_same_hash(self):
        h1 = Hasher()
        h1.update('x')
        h1.update('y')

        h2 = Hasher()
        h2.update('x')
        h2.update('y')

        self.assertEqual(h1.hexdigest(), h2.hexdigest())

    def test_update_with_various_types(self):
        h = Hasher()
        h.update(42)
        h.update([1, 2, 3])
        h.update({'key': 'value'})
        result = h.hexdigest()
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 16)  # xxh64 produces 16 hex chars


class HasherHashDefaultFingerprintBytesTest(DataJuicerTestCaseBase):
    """Test hash_default uses _fingerprint_bytes when available."""

    def test_object_with_fingerprint_bytes(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

            def _fingerprint_bytes(self):
                return str(self.val).encode()

        obj_a = MyObj(42)
        obj_b = MyObj(42)
        obj_c = MyObj(99)

        self.assertEqual(Hasher.hash(obj_a), Hasher.hash(obj_b))
        self.assertNotEqual(Hasher.hash(obj_a), Hasher.hash(obj_c))

    def test_object_without_fingerprint_bytes_uses_dill(self):
        class SimpleObj:
            def __init__(self, val):
                self.val = val

        obj_a = SimpleObj(10)
        obj_b = SimpleObj(10)
        # dill serialization of two different instances with same state
        # should produce same hash
        self.assertEqual(Hasher.hash(obj_a), Hasher.hash(obj_b))


class HasherFindOpOwnerNoFingerprintBytesTest(DataJuicerTestCaseBase):
    """Test _find_op_owner when __self__ exists but lacks _fingerprint_bytes."""

    def test_bound_method_without_fingerprint_bytes(self):
        class RegularObj:
            def do_stuff(self, x):
                return x

        obj = RegularObj()
        result_obj, result_name = Hasher._find_op_owner(obj.do_stuff)
        # __self__ exists but no _fingerprint_bytes, so returns (None, None)
        self.assertIsNone(result_obj)
        self.assertIsNone(result_name)

    def test_wrapped_chain_without_fingerprint_bytes(self):
        """Walk __wrapped__ chain, but no _fingerprint_bytes anywhere."""
        import functools

        class RegularObj:
            def compute(self, x):
                return x

        obj = RegularObj()
        bound = obj.compute

        @functools.wraps(bound)
        def wrapper(*args, **kwargs):
            return bound(*args, **kwargs)

        wrapper.__wrapped__ = bound

        result_obj, result_name = Hasher._find_op_owner(wrapper)
        self.assertIsNone(result_obj)
        self.assertIsNone(result_name)


class UpdateFingerprintArgOrderTest(DataJuicerTestCaseBase):
    """Test that update_fingerprint sorts args by key for determinism."""

    def test_arg_order_does_not_matter(self):
        # Different insertion order but same key-value pairs
        fp1 = update_fingerprint('base', 'transform', {'z': 1, 'a': 2, 'm': 3})
        fp2 = update_fingerprint('base', 'transform', {'a': 2, 'm': 3, 'z': 1})
        self.assertEqual(fp1, fp2)

    def test_different_base_fingerprint_different_result(self):
        fp1 = update_fingerprint('base_a', 'transform', {'x': 1})
        fp2 = update_fingerprint('base_b', 'transform', {'x': 1})
        self.assertNotEqual(fp1, fp2)


class GenerateFingerprintEdgeCasesTest(DataJuicerTestCaseBase):
    """Test generate_fingerprint with edge cases."""

    def test_empty_dataset(self):
        dataset = NestedDataset.from_list([{'text': ''}])
        fp = generate_fingerprint(dataset)
        self.assertIsInstance(fp, str)
        self.assertGreater(len(fp), 0)

    def test_same_dataset_same_fingerprint(self):
        dataset = NestedDataset.from_list([{'text': 'hello'}])
        fp1 = generate_fingerprint(dataset)
        fp2 = generate_fingerprint(dataset)
        self.assertEqual(fp1, fp2)

    def test_different_dataset_different_fingerprint(self):
        ds1 = NestedDataset.from_list([{'text': 'hello'}])
        ds2 = NestedDataset.from_list([{'text': 'world'}])
        fp1 = generate_fingerprint(ds1)
        fp2 = generate_fingerprint(ds2)
        self.assertNotEqual(fp1, fp2)

    def test_with_kwargs(self):
        dataset = NestedDataset.from_list([{'text': 'test'}])
        fp_no_kwargs = generate_fingerprint(dataset)
        fp_with_kwargs = generate_fingerprint(dataset, num_proc=4)
        # Adding kwargs should change the fingerprint
        self.assertNotEqual(fp_no_kwargs, fp_with_kwargs)

    def test_fingerprint_is_valid_hex(self):
        dataset = NestedDataset.from_list([{'text': 'data'}])
        fp = generate_fingerprint(dataset)
        # Should be a valid hex string
        int(fp, 16)


class HasherHashBytesEdgeCasesTest(DataJuicerTestCaseBase):
    """Edge cases for Hasher.hash_bytes."""

    def test_empty_bytes(self):
        result = Hasher.hash_bytes(b'')
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 16)

    def test_empty_list_of_bytes(self):
        result = Hasher.hash_bytes([])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 16)

    def test_large_bytes(self):
        data = b'x' * 1000000
        result = Hasher.hash_bytes(data)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 16)

    def test_single_item_list_same_as_direct(self):
        # hash_bytes(b"hello") vs hash_bytes([b"hello"]) should be the same
        r1 = Hasher.hash_bytes(b'hello')
        r2 = Hasher.hash_bytes([b'hello'])
        self.assertEqual(r1, r2)


if __name__ == '__main__':
    unittest.main()
