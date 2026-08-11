import unittest

from data_juicer.ops.fused_sequential_batch_op import FusedSequentialBatchOp
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.clean_email_mapper import CleanEmailMapper
from data_juicer.ops.mapper.punctuation_normalization_mapper import (
    PunctuationNormalizationMapper,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FusedSequentialBatchOpTest(DataJuicerTestCaseBase):

    def test_init_with_fused_ops(self):
        """Init with pre-built op instances via fused_ops."""
        pn = PunctuationNormalizationMapper()
        ce = CleanEmailMapper()
        fused = FusedSequentialBatchOp(fused_ops=[pn, ce])
        self.assertEqual(fused.group_name, 'fused')
        self.assertEqual(len(fused._fused_ops_input), 2)
        self.assertEqual(fused.op_specs, [])

    def test_init_with_op_specs(self):
        """Init with op_specs triggers lazy construction."""
        specs = [
            {'class_name': 'punctuation_normalization_mapper', 'kwargs': {}},
            {'class_name': 'clean_email_mapper', 'kwargs': {}},
        ]
        fused = FusedSequentialBatchOp(op_specs=specs)
        self.assertIsNone(fused._fused_ops_input)
        self.assertEqual(len(fused.op_specs), 2)
        # Ops not yet built
        self.assertIsNone(fused._ops)

    def test_both_fused_ops_and_op_specs_raises_value_error(self):
        """Passing both fused_ops and op_specs raises ValueError."""
        pn = PunctuationNormalizationMapper()
        specs = [{'class_name': 'clean_email_mapper', 'kwargs': {}}]
        with self.assertRaises(ValueError):
            FusedSequentialBatchOp(fused_ops=[pn], op_specs=specs)

    def test_process_batched_chain_of_mappers(self):
        """process_batched applies a chain of real mappers sequentially."""
        pn = PunctuationNormalizationMapper()
        ce = CleanEmailMapper()
        fused = FusedSequentialBatchOp(fused_ops=[pn, ce])

        samples = {
            'text': [
                'Hello！ Contact test@example.com for info',
                'World？ No email here',
            ]
        }
        result = fused.process_batched(samples)

        self.assertEqual(len(result['text']), 2)
        # Punctuation normalized and email cleaned
        self.assertEqual(result['text'][0], 'Hello! Contact  for info')
        self.assertEqual(result['text'][1], 'World? No email here')

    def test_process_batched_mapper_and_filter_drops_rows(self):
        """Filter sub-op drops rows that don't pass the filter."""
        pn = PunctuationNormalizationMapper()
        tf = TextLengthFilter(min_len=5, max_len=50)
        fused = FusedSequentialBatchOp(fused_ops=[pn, tf])

        samples = {
            'text': [
                'Hi！',  # after normalization: "Hi!" (3 chars) -> filtered out
                'Hello World！ This is a longer text',  # kept
            ]
        }
        result = fused.process_batched(samples)

        self.assertEqual(len(result['text']), 1)
        self.assertEqual(result['text'][0], 'Hello World! This is a longer text')

    def test_process_batched_empty_batch(self):
        """Empty batch returns immediately without error."""
        pn = PunctuationNormalizationMapper()
        fused = FusedSequentialBatchOp(fused_ops=[pn])

        samples = {'text': []}
        result = fused.process_batched(samples)

        self.assertEqual(result['text'], [])

    def test_cleanup_columns_removes_specified_columns(self):
        """cleanup_columns removes the listed columns from the output."""
        pn = PunctuationNormalizationMapper()
        fused = FusedSequentialBatchOp(
            fused_ops=[pn], cleanup_columns=['extra_col', 'another_col']
        )

        samples = {
            'text': ['Hello！ World'],
            'extra_col': ['should be removed'],
            'another_col': ['also removed'],
        }
        result = fused.process_batched(samples)

        self.assertIn('text', result)
        self.assertNotIn('extra_col', result)
        self.assertNotIn('another_col', result)
        self.assertEqual(result['text'][0], 'Hello! World')

    def test_op_specs_lazy_construction(self):
        """op_specs mode lazily constructs ops on first process_batched call."""
        specs = [
            {'class_name': 'punctuation_normalization_mapper', 'kwargs': {}},
            {'class_name': 'clean_email_mapper', 'kwargs': {}},
        ]
        fused = FusedSequentialBatchOp(op_specs=specs)

        # Before processing, _ops is None (lazy)
        self.assertIsNone(fused._ops)

        samples = {
            'text': ['Hello！ user@test.com', 'World？ OK']
        }
        result = fused.process_batched(samples)

        # After processing, _ops is built
        self.assertIsNotNone(fused._ops)
        self.assertEqual(len(fused._ops), 2)
        self.assertEqual(result['text'][0], 'Hello! ')
        self.assertEqual(result['text'][1], 'World? OK')

    def test_op_specs_strips_ray_sched_kwargs(self):
        """Ray scheduling kwargs are stripped from sub-op kwargs."""
        specs = [
            {
                'class_name': 'punctuation_normalization_mapper',
                'kwargs': {
                    'num_gpus': 2,
                    'num_proc': 4,
                    'num_cpus': 8,
                    'memory': 1024,
                },
            },
        ]
        # Should not raise; Ray kwargs are stripped before constructing the op
        fused = FusedSequentialBatchOp(op_specs=specs)
        samples = {'text': ['Hello！']}
        result = fused.process_batched(samples)
        self.assertEqual(result['text'][0], 'Hello!')

    def test_filter_drops_all_rows(self):
        """When all rows are filtered out, result has empty lists."""
        tf = TextLengthFilter(min_len=100, max_len=200)
        fused = FusedSequentialBatchOp(fused_ops=[tf])

        samples = {'text': ['short', 'also short']}
        result = fused.process_batched(samples)

        self.assertEqual(len(result['text']), 0)

    def test_custom_group_name(self):
        """Custom group_name is preserved."""
        pn = PunctuationNormalizationMapper()
        fused = FusedSequentialBatchOp(
            fused_ops=[pn], group_name='my_custom_group'
        )
        self.assertEqual(fused.group_name, 'my_custom_group')


if __name__ == '__main__':
    unittest.main()
