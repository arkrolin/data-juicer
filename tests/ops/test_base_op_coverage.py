"""
Additional coverage tests for data_juicer/ops/base_op.py.

Targets lines not covered by the existing test_base_op.py:
- OP.__init__ branches (deprecated params, accelerator, batch_mode, etc.)
- _fingerprint_bytes with nested OP instances
- Filter.get_keep_boolean with reversed_range
- Filter.__init_subclass__ enforcement
- Mapper.__init_subclass__ enforcement
- OP.run (meta field addition, stats field addition, index_key)
- Filter.run, Mapper.run, Deduplicator.run, Selector.run, Grouper.run, Aggregator.run
- OP helper methods (use_auto_proc, is_batched_op, runtime_np, etc.)
"""

import unittest

import numpy as np
from datasets import Dataset

from data_juicer.ops.base_op import (
    OP,
    Aggregator,
    Deduplicator,
    Filter,
    Grouper,
    Mapper,
    Selector,
)
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------

class SimpleMapper(Mapper):
    _batched_op = False

    def process_single(self, sample):
        sample['text'] = sample['text'].strip()
        return sample


class BatchMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        samples['text'] = [t.upper() for t in samples['text']]
        return samples


class SimpleFilter(Filter):
    _batched_op = False

    def compute_stats_single(self, sample, context=False):
        sample[Fields.stats] = {
            **sample.get(Fields.stats, {}),
            'length': len(sample['text']),
        }
        return sample

    def process_single(self, sample):
        return sample[Fields.stats]['length'] >= 3


class SimpleDeduplicator(Deduplicator):
    _batched_op = False

    def compute_hash(self, sample):
        import hashlib
        sample[HashKeys.hash] = hashlib.md5(
            sample['text'].encode()).hexdigest()
        return sample

    def process(self, dataset, show_num=0):
        seen = set()
        keep_indices = []
        for i, sample in enumerate(dataset):
            h = sample[HashKeys.hash]
            if h not in seen:
                seen.add(h)
                keep_indices.append(i)
        return dataset.select(keep_indices), []


class SimpleSelector(Selector):
    def process(self, dataset):
        return dataset.select(range(min(2, len(dataset))))


class SimpleGrouper(Grouper):
    def process(self, dataset):
        batch = {key: list(dataset[key]) for key in dataset.features}
        return [batch]


class SimpleAggregator(Aggregator):
    _batched_op = False

    def process_single(self, sample):
        if isinstance(sample.get('text'), list):
            sample['text'] = ' '.join(sample['text'])
        return sample


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class OPInitBranchesTest(DataJuicerTestCaseBase):

    def test_deprecated_cpu_required(self):
        op = SimpleMapper(cpu_required=2)
        self.assertEqual(op.num_cpus, 2)

    def test_deprecated_gpu_required(self):
        op = SimpleMapper(gpu_required=1)
        self.assertEqual(op.num_gpus, 1)

    def test_deprecated_mem_required_string(self):
        op = SimpleMapper(mem_required='4GB')
        self.assertIsNotNone(op.memory)
        self.assertAlmostEqual(op.memory, 4.0, places=0)

    def test_memory_string_conversion(self):
        op = SimpleMapper(memory='2GB')
        self.assertAlmostEqual(op.memory, 2.0, places=0)

    def test_accelerator_override(self):
        op = SimpleMapper(accelerator='cpu')
        self.assertEqual(op.accelerator, 'cpu')

    def test_batch_size_default_for_cuda(self):
        op = SimpleMapper(accelerator='cuda')
        self.assertEqual(op.batch_size, 10)

    def test_batch_size_custom(self):
        op = SimpleMapper(batch_size=32)
        self.assertEqual(op.batch_size, 32)

    def test_num_proc_auto(self):
        op = SimpleMapper()
        self.assertEqual(op.num_proc, -1)

    def test_num_proc_explicit(self):
        op = SimpleMapper(num_proc=4)
        self.assertEqual(op.num_proc, 4)

    def test_auto_op_parallelism_false(self):
        op = SimpleMapper(auto_op_parallelism=False, num_proc=2)
        self.assertEqual(op.num_proc, 2)
        self.assertFalse(op.auto_op_parallelism)

    def test_skip_op_error(self):
        op = SimpleMapper(skip_op_error=True)
        self.assertTrue(op.skip_op_error)

    def test_work_dir(self):
        op = SimpleMapper(work_dir='/tmp/test_work')
        self.assertEqual(op.work_dir, '/tmp/test_work')

    def test_turbo_flag(self):
        op = SimpleMapper(turbo=True)
        self.assertTrue(op.turbo)

    def test_index_key(self):
        op = SimpleMapper(index_key='idx')
        self.assertEqual(op.index_key, 'idx')

    def test_ray_execution_mode_actor(self):
        op = SimpleMapper(ray_execution_mode='actor')
        self.assertEqual(op.ray_execution_mode, 'actor')
        self.assertTrue(op.use_ray_actor())

    def test_ray_execution_mode_task(self):
        op = SimpleMapper(ray_execution_mode='task')
        self.assertEqual(op.ray_execution_mode, 'task')
        self.assertFalse(op.use_ray_actor())

    def test_ray_execution_mode_invalid(self):
        with self.assertRaises(AssertionError):
            SimpleMapper(ray_execution_mode='invalid')


class OPFingerprintTest(DataJuicerTestCaseBase):

    def test_fingerprint_basic(self):
        op1 = SimpleMapper(text_key='text')
        op2 = SimpleMapper(text_key='text')
        self.assertEqual(op1._fingerprint_bytes(), op2._fingerprint_bytes())

    def test_fingerprint_differs_with_params(self):
        op1 = SimpleMapper(text_key='text')
        op2 = SimpleMapper(text_key='content')
        self.assertNotEqual(
            op1._fingerprint_bytes(), op2._fingerprint_bytes())

    def test_fingerprint_excludes_work_dir(self):
        op1 = SimpleMapper(work_dir='/tmp/a')
        op2 = SimpleMapper(work_dir='/tmp/b')
        self.assertEqual(op1._fingerprint_bytes(), op2._fingerprint_bytes())


class OPHelperMethodsTest(DataJuicerTestCaseBase):

    def test_is_batched_op_false(self):
        op = SimpleMapper()
        self.assertFalse(op.is_batched_op())

    def test_is_batched_op_true(self):
        op = BatchMapper()
        self.assertTrue(op.is_batched_op())

    def test_batch_mode_override(self):
        op = SimpleMapper(batch_mode=True)
        self.assertTrue(op.is_batched_op())

    def test_batch_mode_conflict_raises(self):
        with self.assertRaises(ValueError):
            BatchMapper(batch_mode=False)

    def test_use_auto_proc_default(self):
        op = SimpleMapper()
        self.assertTrue(op.use_auto_proc())

    def test_use_auto_proc_explicit_num(self):
        op = SimpleMapper(num_proc=4)
        self.assertFalse(op.use_auto_proc())

    def test_runtime_np(self):
        op = SimpleMapper(num_proc=2, auto_op_parallelism=False)
        self.assertEqual(op.runtime_np(), 2)

    def test_use_cuda_without_gpu(self):
        op = SimpleMapper(accelerator='cpu')
        self.assertFalse(op.use_cuda())

    def test_remove_extra_parameters(self):
        op = SimpleMapper()
        params = {'self': op, 'text_key': 'text', '_private': 1, 'batch_size': 10}
        cleaned = op.remove_extra_parameters(params)
        self.assertNotIn('self', cleaned)
        self.assertNotIn('_private', cleaned)
        self.assertIn('text_key', cleaned)

    def test_remove_extra_parameters_with_keys(self):
        op = SimpleMapper()
        params = {'a': 1, 'b': 2, 'c': 3}
        cleaned = op.remove_extra_parameters(params, keys=['b'])
        self.assertIn('a', cleaned)
        self.assertNotIn('b', cleaned)
        self.assertIn('c', cleaned)

    def test_add_parameters(self):
        op = SimpleMapper()
        init_params = {'param1': 'val1'}
        result = op.add_parameters(init_params, extra='extra_val')
        self.assertEqual(result['param1'], 'val1')
        self.assertEqual(result['extra'], 'extra_val')
        self.assertNotIn('extra', init_params)

    def test_empty_history(self):
        op = SimpleMapper()
        hist = op.empty_history()
        self.assertIsInstance(hist, np.ndarray)
        self.assertEqual(hist.shape, (0, 0))


class FilterKeepBooleanTest(DataJuicerTestCaseBase):

    def test_default_closed_intervals(self):
        f = SimpleFilter()
        self.assertTrue(f.get_keep_boolean(5, min_val=5, max_val=10))
        self.assertTrue(f.get_keep_boolean(10, min_val=5, max_val=10))
        self.assertFalse(f.get_keep_boolean(4, min_val=5, max_val=10))
        self.assertFalse(f.get_keep_boolean(11, min_val=5, max_val=10))

    def test_open_intervals(self):
        f = SimpleFilter(min_closed_interval=False, max_closed_interval=False)
        self.assertFalse(f.get_keep_boolean(5, min_val=5, max_val=10))
        self.assertFalse(f.get_keep_boolean(10, min_val=5, max_val=10))
        self.assertTrue(f.get_keep_boolean(6, min_val=5, max_val=10))

    def test_reversed_range(self):
        f = SimpleFilter(reversed_range=True)
        # reversed: keep values OUTSIDE [5, 10]
        self.assertTrue(f.get_keep_boolean(4, min_val=5, max_val=10))
        self.assertTrue(f.get_keep_boolean(11, min_val=5, max_val=10))
        self.assertFalse(f.get_keep_boolean(7, min_val=5, max_val=10))

    def test_no_bounds(self):
        f = SimpleFilter()
        self.assertTrue(f.get_keep_boolean(999))

    def test_min_only(self):
        f = SimpleFilter()
        self.assertTrue(f.get_keep_boolean(5, min_val=3))
        self.assertFalse(f.get_keep_boolean(2, min_val=3))


class MapperInitSubclassTest(DataJuicerTestCaseBase):

    def test_cannot_override_process(self):
        with self.assertRaises(TypeError):
            class BadMapper(Mapper):
                def process(self, sample):
                    return sample


class FilterInitSubclassTest(DataJuicerTestCaseBase):

    def test_cannot_override_process(self):
        with self.assertRaises(TypeError):
            class BadFilter(Filter):
                def process(self, sample):
                    return True

    def test_cannot_override_compute_stats(self):
        with self.assertRaises(TypeError):
            class BadFilter2(Filter):
                def compute_stats(self, sample):
                    return sample


class MapperRunTest(DataJuicerTestCaseBase):

    def test_mapper_run_basic(self):
        ds = Dataset.from_dict({'text': ['  hello  ', '  world  ']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        mapper = SimpleMapper()
        result = mapper.run(ds)
        self.assertEqual(list(result['text']), ['hello', 'world'])

    def test_mapper_run_batched(self):
        ds = Dataset.from_dict({'text': ['hello', 'world']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        mapper = BatchMapper()
        result = mapper.run(ds)
        self.assertEqual(list(result['text']), ['HELLO', 'WORLD'])

    def test_mapper_run_with_index_key(self):
        ds = Dataset.from_dict({'text': ['a', 'b', 'c']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        mapper = SimpleMapper(index_key='idx')
        result = mapper.run(ds)
        self.assertIn('idx', result.features)
        self.assertEqual(list(result['idx']), [0, 1, 2])


class FilterRunTest(DataJuicerTestCaseBase):

    def test_filter_run_basic(self):
        ds = Dataset.from_dict({
            'text': ['hi', 'hello world', 'ok'],
            Fields.stats: [{}, {}, {}],
        })
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        f = SimpleFilter()
        result = f.run(ds)
        texts = list(result['text'])
        self.assertIn('hello world', texts)
        self.assertNotIn('hi', texts)
        self.assertNotIn('ok', texts)

    def test_filter_run_no_reduce(self):
        ds = Dataset.from_dict({
            'text': ['hi', 'hello world'],
            Fields.stats: [{}, {}],
        })
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        f = SimpleFilter()
        result = f.run(ds, reduce=False)
        self.assertEqual(len(result), 2)

    def test_filter_adds_stats_column(self):
        ds = Dataset.from_dict({'text': ['hello', 'world']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        f = SimpleFilter()
        result = f.run(ds)
        self.assertIn(Fields.stats, result.features)


class DeduplicatorRunTest(DataJuicerTestCaseBase):

    def test_deduplicator_run_basic(self):
        ds = Dataset.from_dict({'text': ['hello', 'world', 'hello']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        dedup = SimpleDeduplicator()
        result = dedup.run(ds)
        texts = list(result['text'])
        self.assertEqual(len(texts), 2)
        self.assertIn('hello', texts)
        self.assertIn('world', texts)


class SelectorRunTest(DataJuicerTestCaseBase):

    def test_selector_run(self):
        ds = Dataset.from_dict({'text': ['a', 'b', 'c', 'd']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        sel = SimpleSelector()
        result = sel.run(ds)
        self.assertEqual(len(result), 2)


class GrouperRunTest(DataJuicerTestCaseBase):

    def test_grouper_run(self):
        ds = Dataset.from_dict({'text': ['hello', 'world']})
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        grouper = SimpleGrouper()
        result = grouper.run(ds)
        self.assertEqual(len(result), 1)


class AggregatorRunTest(DataJuicerTestCaseBase):

    def test_aggregator_run(self):
        ds = Dataset.from_dict({
            'text': [['hello', 'world'], ['foo', 'bar']],
        })
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset(ds)
        agg = SimpleAggregator()
        result = agg.run(ds)
        self.assertIn(Fields.batch_meta, result.features)
        texts = list(result['text'])
        self.assertEqual(texts[0], 'hello world')
        self.assertEqual(texts[1], 'foo bar')


class MapperCallableTest(DataJuicerTestCaseBase):

    def test_mapper_callable(self):
        mapper = SimpleMapper()
        result = mapper({'text': '  hello  '})
        self.assertEqual(result['text'], 'hello')

    def test_filter_callable(self):
        f = SimpleFilter()
        result = f({
            'text': 'hello world',
            Fields.stats: {},
        })
        self.assertIn(Fields.stats, result)


if __name__ == '__main__':
    unittest.main()
