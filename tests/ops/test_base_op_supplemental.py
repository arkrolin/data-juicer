"""
Supplemental tests for data_juicer/ops/base_op.py.

Covers additional edge cases and scenarios beyond test_base_op.py.
"""
import unittest

import pyarrow as pa

from data_juicer.ops.base_op import (
    OP,
    Filter,
    Mapper,
    catch_map_batches_exception,
    catch_map_single_exception,
    convert_arrow_to_python,
    convert_dict_list_to_list_dict,
    convert_list_dict_to_dict_list,
    sample_to_dict,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------


class SimpleMapper(Mapper):
    _batched_op = False

    def process_single(self, sample):
        return sample


class SimpleFilter(Filter):
    _batched_op = False

    def compute_stats_single(self, sample, context=False):
        sample[Fields.stats] = {
            **sample.get(Fields.stats, {}),
            'length': len(sample.get('text', '')),
        }
        return sample

    def process_single(self, sample):
        return sample[Fields.stats].get('length', 0) >= 5


# ---------------------------------------------------------------------------
# convert_list_dict_to_dict_list: additional edge cases
# ---------------------------------------------------------------------------


class ConvertListDictToDictListSupplementalTest(DataJuicerTestCaseBase):

    def test_empty_values(self):
        """Test with empty string and None values."""
        samples = [{'a': '', 'b': None}, {'a': 'x', 'b': 42}]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result, {'a': ['', 'x'], 'b': [None, 42]})

    def test_nested_dict_values(self):
        """Test with nested dict values (like stats)."""
        samples = [
            {'text': 'hello', 'stats': {'len': 5}},
            {'text': 'world', 'stats': {'len': 5}},
        ]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result['text'], ['hello', 'world'])
        self.assertEqual(result['stats'], [{'len': 5}, {'len': 5}])

    def test_list_values(self):
        """Test with list-type values in samples."""
        samples = [
            {'text': 'a', 'images': ['img1.jpg', 'img2.jpg']},
            {'text': 'b', 'images': ['img3.jpg']},
        ]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result['text'], ['a', 'b'])
        self.assertEqual(result['images'],
                         [['img1.jpg', 'img2.jpg'], ['img3.jpg']])

    def test_many_keys(self):
        """Test with many keys."""
        samples = [
            {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
            {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10},
        ]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result['a'], [1, 6])
        self.assertEqual(result['e'], [5, 10])

    def test_three_samples(self):
        """Test with three samples."""
        samples = [{'x': i} for i in range(3)]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result, {'x': [0, 1, 2]})


# ---------------------------------------------------------------------------
# convert_dict_list_to_list_dict: additional edge cases
# ---------------------------------------------------------------------------


class ConvertDictListToListDictSupplementalTest(DataJuicerTestCaseBase):

    def test_empty_lists(self):
        """Empty lists produce empty list of dicts."""
        samples = {'a': [], 'b': []}
        result = convert_dict_list_to_list_dict(samples)
        self.assertEqual(result, [])

    def test_three_items(self):
        """Three items per key."""
        samples = {'x': [10, 20, 30], 'y': ['a', 'b', 'c']}
        result = convert_dict_list_to_list_dict(samples)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], {'x': 10, 'y': 'a'})
        self.assertEqual(result[2], {'x': 30, 'y': 'c'})

    def test_roundtrip_with_nested_data(self):
        """Roundtrip with nested structures."""
        original = [
            {'text': 'hi', 'meta': {'source': 'web'}},
            {'text': 'lo', 'meta': {'source': 'book'}},
        ]
        dict_list = convert_list_dict_to_dict_list(original)
        roundtripped = convert_dict_list_to_list_dict(dict_list)
        self.assertEqual(roundtripped, original)

    def test_single_key_multiple_values(self):
        """Single key with multiple values."""
        samples = {'text': ['alpha', 'beta', 'gamma']}
        result = convert_dict_list_to_list_dict(samples)
        self.assertEqual(result, [
            {'text': 'alpha'},
            {'text': 'beta'},
            {'text': 'gamma'},
        ])


# ---------------------------------------------------------------------------
# convert_arrow_to_python: supplemental tests
# ---------------------------------------------------------------------------


class ConvertArrowToPythonSupplementalTest(DataJuicerTestCaseBase):

    def test_preserves_function_name(self):
        """Wrapped function retains original function name."""
        @convert_arrow_to_python
        def my_function(sample):
            return sample

        self.assertEqual(my_function.__name__, 'my_function')

    def test_with_extra_args_and_kwargs(self):
        """Decorator passes extra args and kwargs through."""
        @convert_arrow_to_python
        def fn(sample, multiplier=1):
            sample['val'] = [v * multiplier for v in sample['val']]
            return sample

        table = pa.table({'val': [1, 2, 3]})
        result = fn(table, multiplier=10)
        self.assertEqual(result['val'], [10, 20, 30])

    def test_with_multi_column_table(self):
        """Multi-column arrow table is fully converted."""
        @convert_arrow_to_python
        def fn(sample):
            return sample

        table = pa.table({
            'text': ['hello', 'world'],
            'score': [0.9, 0.1],
            'labels': [['a'], ['b', 'c']],
        })
        result = fn(table)
        self.assertEqual(result['text'], ['hello', 'world'])
        self.assertEqual(result['score'], [0.9, 0.1])
        self.assertEqual(result['labels'], [['a'], ['b', 'c']])


# ---------------------------------------------------------------------------
# catch_map_batches_exception: supplemental tests
# ---------------------------------------------------------------------------


class CatchMapBatchesExceptionSupplementalTest(DataJuicerTestCaseBase):

    def test_op_name_defaults_to_method_name(self):
        """When op_name is None, it defaults to method.__name__."""
        def my_batch_fn(samples):
            return samples

        wrapped = catch_map_batches_exception(my_batch_fn)
        self.assertEqual(wrapped.__name__, 'my_batch_fn')

    def test_custom_op_name(self):
        """Custom op_name does not change function name attribute."""
        def fn(samples):
            return samples

        wrapped = catch_map_batches_exception(fn, op_name='custom_op')
        # The wrapper preserves original function name via @wraps
        self.assertEqual(wrapped.__name__, 'fn')

    def test_arrow_table_with_multiple_columns(self):
        """Arrow table input with multiple columns is properly converted."""
        def fn(samples):
            return samples

        wrapped = catch_map_batches_exception(fn)
        table = pa.table({
            'text': ['a', 'b'],
            'num': [1, 2],
            Fields.stats: [None, None],
            Fields.source_file: ['f1', 'f2'],
        })
        result = wrapped(table)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['text'], ['a', 'b'])

    def test_error_with_many_keys_returns_all_empty(self):
        """On error with skip, all original keys get empty lists."""
        def fn(samples):
            raise RuntimeError('fail')

        wrapped = catch_map_batches_exception(fn, skip_op_error=True,
                                              op_name='test')
        result = wrapped({
            'text': ['hello'],
            'images': [['img.jpg']],
            'score': [0.5],
        })
        self.assertEqual(result['text'], [])
        self.assertEqual(result['images'], [])
        self.assertEqual(result['score'], [])
        self.assertEqual(result[Fields.stats], [])
        self.assertEqual(result[Fields.source_file], [])

    def test_modifies_samples_in_place(self):
        """Function that modifies samples returns modified version."""
        def fn(samples):
            samples['count'] = [len(t) for t in samples['text']]
            return samples

        wrapped = catch_map_batches_exception(fn)
        result = wrapped({'text': ['abc', 'de']})
        self.assertEqual(result['count'], [3, 2])


# ---------------------------------------------------------------------------
# catch_map_single_exception: supplemental tests
# ---------------------------------------------------------------------------


class CatchMapSingleExceptionSupplementalTest(DataJuicerTestCaseBase):

    def test_non_batched_dict_is_not_unwrapped(self):
        """Non-batched (non-list values) dict goes directly to method."""
        call_log = []

        def fn(sample):
            call_log.append(sample)
            sample['processed'] = True
            return sample

        wrapped = catch_map_single_exception(fn)
        result = wrapped({'text': 'hello', 'num': 42})
        self.assertEqual(result['processed'], True)
        self.assertEqual(call_log[0]['text'], 'hello')

    def test_batched_single_element_unwrap(self):
        """A batch of size 1 gets unwrapped to a single dict then re-wrapped."""
        def fn(sample):
            # sample should be a flat dict, not batched
            self.assertIsInstance(sample['text'], str)
            sample['text'] = sample['text'].upper()
            sample['upper'] = sample['text']
            return sample

        wrapped = catch_map_single_exception(fn, return_sample=True)
        result = wrapped({'text': ['foo'], 'n': [1]})
        self.assertEqual(result['text'], ['FOO'])
        self.assertEqual(result['upper'], ['FOO'])

    def test_arrow_table_single_row(self):
        """Arrow table with single row is treated as batched."""
        def fn(sample):
            sample['doubled'] = sample['val'] * 2
            return sample

        wrapped = catch_map_single_exception(fn, return_sample=True)
        table = pa.table({'val': [5], 'key': ['a']})
        result = wrapped(table)
        self.assertEqual(result['doubled'], [10])

    def test_preserves_function_name(self):
        """Wrapper preserves original function name."""
        def my_single_fn(sample):
            return sample

        wrapped = catch_map_single_exception(my_single_fn)
        self.assertEqual(wrapped.__name__, 'my_single_fn')

    def test_return_sample_false_batched(self):
        """With return_sample=False, raw return values are wrapped in list."""
        def fn(sample):
            return sample['val'] > 3

        wrapped = catch_map_single_exception(fn, return_sample=False)
        result = wrapped({'val': [5], 'key': ['a']})
        self.assertEqual(result, [True])

    def test_error_in_non_batched_propagates(self):
        """Errors in non-batched mode always propagate (no fault tolerance)."""
        def fn(sample):
            raise ValueError('oops')

        wrapped = catch_map_single_exception(fn, skip_op_error=True)
        # Non-batched mode does not catch errors
        with self.assertRaises(ValueError):
            wrapped({'text': 'hello', 'n': 42})


# ---------------------------------------------------------------------------
# sample_to_dict: supplemental tests
# ---------------------------------------------------------------------------


class SampleToDictSupplementalTest(DataJuicerTestCaseBase):

    def test_arrow_table_multiple_columns(self):
        """Multi-column arrow table converts to dict of lists."""
        table = pa.table({
            'text': ['a', 'b', 'c'],
            'score': [1.0, 2.0, 3.0],
        })
        result = sample_to_dict(table)
        self.assertEqual(result['text'], ['a', 'b', 'c'])
        self.assertEqual(result['score'], [1.0, 2.0, 3.0])

    def test_arrow_table_empty(self):
        """Empty arrow table converts to dict with empty lists."""
        table = pa.table({'text': pa.array([], type=pa.string())})
        result = sample_to_dict(table)
        self.assertEqual(result['text'], [])

    def test_integer_input_raises(self):
        """Integer input raises ValueError."""
        with self.assertRaises(ValueError):
            sample_to_dict(123)

    def test_string_input_raises(self):
        """String input raises ValueError."""
        with self.assertRaises(ValueError):
            sample_to_dict("not a dict")


# ---------------------------------------------------------------------------
# OP initialization: supplemental tests
# ---------------------------------------------------------------------------


class OPInitSupplementalTest(DataJuicerTestCaseBase):

    def test_name_from_class_attr(self):
        """_name attribute is accessible on instances."""
        op = SimpleMapper()
        # _name is a class attribute, defaults to empty string
        self.assertIsInstance(op._name, str)

    def test_accelerator_class_default(self):
        """Accelerator defaults to 'cpu' from class attribute."""
        self.assertEqual(OP._accelerator, 'cpu')
        op = SimpleMapper()
        self.assertEqual(op.accelerator, 'cpu')

    def test_batched_op_class_default(self):
        """_batched_op defaults to False."""
        self.assertEqual(OP._batched_op, False)

    def test_is_batched_op_returns_false_for_simple_mapper(self):
        """SimpleMapper is not a batched op."""
        op = SimpleMapper()
        self.assertFalse(op.is_batched_op())

    def test_use_cuda_returns_false_on_cpu(self):
        """use_cuda returns False when accelerator is cpu."""
        op = SimpleMapper(accelerator='cpu')
        self.assertFalse(op.use_cuda())

    def test_num_proc_default_auto(self):
        """num_proc defaults to -1 (auto) when auto_op_parallelism is True."""
        op = SimpleMapper()
        self.assertEqual(op.num_proc, -1)

    def test_num_proc_manual(self):
        """num_proc can be set manually."""
        op = SimpleMapper(auto_op_parallelism=False, num_proc=4)
        self.assertEqual(op.num_proc, 4)

    def test_system_key_default(self):
        """system_key defaults to 'system'."""
        op = SimpleMapper()
        self.assertEqual(op.system_key, 'system')

    def test_instruction_key_default(self):
        """instruction_key defaults to 'instruction'."""
        op = SimpleMapper()
        self.assertEqual(op.instruction_key, 'instruction')

    def test_prompt_key_default(self):
        """prompt_key defaults to 'prompt'."""
        op = SimpleMapper()
        self.assertEqual(op.prompt_key, 'prompt')

    def test_image_bytes_key_default(self):
        """image_bytes_key defaults to 'image_bytes'."""
        op = SimpleMapper()
        self.assertEqual(op.image_bytes_key, 'image_bytes')

    def test_batch_mode_none_default(self):
        """batch_mode defaults to None."""
        op = SimpleMapper()
        self.assertIsNone(op.batch_mode)

    def test_cuda_accelerator_batch_size_default(self):
        """CUDA accelerator gets batch_size=10 by default."""
        op = SimpleMapper(accelerator='cuda')
        self.assertEqual(op.batch_size, 10)


# ---------------------------------------------------------------------------
# Filter.get_keep_boolean: supplemental tests
# ---------------------------------------------------------------------------


class FilterGetKeepBooleanSupplementalTest(DataJuicerTestCaseBase):

    def test_both_none_always_true(self):
        """No min or max means always keep."""
        op = SimpleFilter()
        self.assertTrue(op.get_keep_boolean(0, None, None))
        self.assertTrue(op.get_keep_boolean(-100, None, None))
        self.assertTrue(op.get_keep_boolean(9999, None, None))

    def test_min_only_closed(self):
        """With only min_val, closed interval."""
        op = SimpleFilter(min_closed_interval=True)
        self.assertTrue(op.get_keep_boolean(5, 5, None))
        self.assertFalse(op.get_keep_boolean(4, 5, None))

    def test_min_only_open(self):
        """With only min_val, open interval."""
        op = SimpleFilter(min_closed_interval=False)
        self.assertFalse(op.get_keep_boolean(5, 5, None))
        self.assertTrue(op.get_keep_boolean(6, 5, None))

    def test_max_only_closed(self):
        """With only max_val, closed interval."""
        op = SimpleFilter(max_closed_interval=True)
        self.assertTrue(op.get_keep_boolean(10, None, 10))
        self.assertFalse(op.get_keep_boolean(11, None, 10))

    def test_max_only_open(self):
        """With only max_val, open interval."""
        op = SimpleFilter(max_closed_interval=False)
        self.assertFalse(op.get_keep_boolean(10, None, 10))
        self.assertTrue(op.get_keep_boolean(9, None, 10))

    def test_reversed_range_excludes_middle(self):
        """Reversed range excludes the middle of [min, max]."""
        op = SimpleFilter(reversed_range=True)
        # Normal [3, 7] would keep values in [3,7]
        # Reversed keeps values outside that range
        self.assertTrue(op.get_keep_boolean(2, 3, 7))
        self.assertTrue(op.get_keep_boolean(8, 3, 7))
        # Boundary: reversed flips closed to open
        self.assertTrue(op.get_keep_boolean(3, 3, 7))
        self.assertTrue(op.get_keep_boolean(7, 3, 7))
        self.assertFalse(op.get_keep_boolean(5, 3, 7))

    def test_float_values(self):
        """Works with float values."""
        op = SimpleFilter()
        self.assertTrue(op.get_keep_boolean(0.5, 0.0, 1.0))
        self.assertFalse(op.get_keep_boolean(1.5, 0.0, 1.0))

    def test_negative_values(self):
        """Works with negative values."""
        op = SimpleFilter()
        self.assertTrue(op.get_keep_boolean(-5, -10, 0))
        self.assertFalse(op.get_keep_boolean(-15, -10, 0))


# ---------------------------------------------------------------------------
# Mapper.__init_subclass__: supplemental tests
# ---------------------------------------------------------------------------


class MapperInitSubclassSupplementalTest(DataJuicerTestCaseBase):

    def test_valid_subclass_with_process_single(self):
        """Subclass with process_single is allowed."""
        class ValidMapper(Mapper):
            def process_single(self, sample):
                return sample

        op = ValidMapper()
        self.assertIsNotNone(op)

    def test_valid_subclass_with_process_batched(self):
        """Subclass with process_batched is allowed."""
        class ValidBatchedMapper(Mapper):
            _batched_op = True

            def process_batched(self, samples):
                return samples

        op = ValidBatchedMapper()
        self.assertIsNotNone(op)

    def test_override_process_raises_type_error(self):
        """Subclass overriding 'process' raises TypeError."""
        with self.assertRaises(TypeError):
            class BadMapper(Mapper):
                def process(self, sample):
                    return sample


# ---------------------------------------------------------------------------
# Filter.__init_subclass__: supplemental tests
# ---------------------------------------------------------------------------


class FilterInitSubclassSupplementalTest(DataJuicerTestCaseBase):

    def test_valid_subclass_with_required_methods(self):
        """Subclass with compute_stats_single and process_single is fine."""
        class ValidFilter(Filter):
            def compute_stats_single(self, sample, context=False):
                return sample

            def process_single(self, sample):
                return True

        op = ValidFilter()
        self.assertIsNotNone(op)

    def test_override_compute_stats_raises(self):
        """Subclass overriding 'compute_stats' raises TypeError."""
        with self.assertRaises(TypeError):
            class BadFilter(Filter):
                def compute_stats(self, sample):
                    return sample

    def test_override_process_raises(self):
        """Subclass overriding 'process' raises TypeError."""
        with self.assertRaises(TypeError):
            class BadFilter(Filter):
                def process(self, sample):
                    return True


# ---------------------------------------------------------------------------
# Integration: end-to-end simple filter pipeline
# ---------------------------------------------------------------------------


class FilterEndToEndTest(DataJuicerTestCaseBase):

    def test_simple_filter_compute_and_process(self):
        """Test filter compute_stats + process in sequence."""
        op = SimpleFilter()
        sample = {'text': 'hello world', Fields.stats: {}}

        # compute_stats adds the length
        result = op.compute_stats(sample)
        self.assertIn('length', result[Fields.stats])
        self.assertEqual(result[Fields.stats]['length'], 11)

        # process_single decides to keep
        keep = op.process_single(result)
        self.assertTrue(keep)

    def test_simple_filter_rejects_short(self):
        """Short text is rejected by filter."""
        op = SimpleFilter()
        sample = {'text': 'hi', Fields.stats: {}}

        result = op.compute_stats(sample)
        self.assertEqual(result[Fields.stats]['length'], 2)

        keep = op.process_single(result)
        self.assertFalse(keep)


if __name__ == '__main__':
    unittest.main()
