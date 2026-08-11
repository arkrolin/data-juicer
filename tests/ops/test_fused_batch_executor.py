import unittest

from data_juicer.ops.base_op import (
    NON_STATS_FILTERS,
    TAGGING_OPS,
    Deduplicator,
    Filter,
    Mapper,
)
from data_juicer.ops.filter import TextLengthFilter
from data_juicer.ops.fused_batch_executor import (
    GENERAL_FUSED_EXECUTION_POLICY,
    SequentialBatchExecutionPolicy,
    _cleanup_context_rows,
    _ensure_meta_if_needed,
    _ensure_stats_if_needed,
    _needs_meta,
    _needs_stats,
    _run_filter,
    _run_mapper,
    _uses_cuda,
    _validate_batch,
    execute_sequential_batch,
    get_batch_size,
)
from data_juicer.ops.mapper import CleanEmailMapper, PunctuationNormalizationMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class GetBatchSizeTest(DataJuicerTestCaseBase):
    """Tests for get_batch_size helper."""

    def test_empty_dict(self):
        self.assertEqual(get_batch_size({}), 0)

    def test_none_input(self):
        self.assertEqual(get_batch_size(None), 0)

    def test_single_key(self):
        samples = {'text': ['a', 'b', 'c']}
        self.assertEqual(get_batch_size(samples), 3)

    def test_multiple_keys(self):
        samples = {'text': ['a', 'b'], 'ids': [1, 2]}
        self.assertEqual(get_batch_size(samples), 2)

    def test_empty_values(self):
        samples = {'text': []}
        self.assertEqual(get_batch_size(samples), 0)


class SequentialBatchExecutionPolicyTest(DataJuicerTestCaseBase):
    """Tests for SequentialBatchExecutionPolicy dataclass."""

    def test_default_values(self):
        policy = SequentialBatchExecutionPolicy()
        self.assertFalse(policy.copy_input)
        self.assertFalse(policy.shared_context)
        self.assertTrue(policy.use_op_wrappers)
        self.assertTrue(policy.validate)
        self.assertTrue(policy.ensure_fields)

    def test_general_fused_policy_values(self):
        policy = GENERAL_FUSED_EXECUTION_POLICY
        self.assertTrue(policy.copy_input)
        self.assertTrue(policy.shared_context)
        self.assertFalse(policy.use_op_wrappers)
        self.assertFalse(policy.validate)
        self.assertFalse(policy.ensure_fields)

    def test_frozen_dataclass(self):
        policy = SequentialBatchExecutionPolicy()
        with self.assertRaises(Exception):
            policy.copy_input = True


class ValidateBatchTest(DataJuicerTestCaseBase):
    """Tests for _validate_batch."""

    def test_none_result_raises(self):
        op = CleanEmailMapper()
        with self.assertRaises(ValueError) as ctx:
            _validate_batch(None, op, 'owner', 'process', True)
        self.assertIn('returned None', str(ctx.exception))
        self.assertIn('clean_email_mapper', str(ctx.exception))

    def test_non_dict_with_validate_raises(self):
        op = CleanEmailMapper()
        with self.assertRaises(ValueError) as ctx:
            _validate_batch([1, 2, 3], op, 'owner', 'process', True)
        self.assertIn('unsupported batch type', str(ctx.exception))
        self.assertIn('list', str(ctx.exception))

    def test_non_dict_without_validate_passes(self):
        op = CleanEmailMapper()
        result = _validate_batch([1, 2, 3], op, 'owner', 'process', False)
        self.assertEqual(result, [1, 2, 3])

    def test_valid_dict_passes(self):
        op = CleanEmailMapper()
        batch = {'text': ['hello']}
        result = _validate_batch(batch, op, 'owner', 'process', True)
        self.assertEqual(result, batch)

    def test_none_result_even_with_validate_false_raises(self):
        op = CleanEmailMapper()
        with self.assertRaises(ValueError):
            _validate_batch(None, op, 'owner', 'process', False)


class UsesCudaTest(DataJuicerTestCaseBase):
    """Tests for _uses_cuda."""

    def test_cpu_op_with_wrappers(self):
        op = CleanEmailMapper()
        self.assertFalse(_uses_cuda(op, True))

    def test_cpu_op_without_wrappers(self):
        op = CleanEmailMapper()
        self.assertFalse(_uses_cuda(op, False))


class NeedsMetaTest(DataJuicerTestCaseBase):
    """Tests for _needs_meta."""

    def test_meta_already_in_samples(self):
        samples = {'text': ['hello'], Fields.meta: [{}]}
        op = CleanEmailMapper()
        self.assertTrue(_needs_meta(samples, op))

    def test_meta_not_needed_for_basic_mapper(self):
        samples = {'text': ['hello']}
        op = CleanEmailMapper()
        self.assertFalse(_needs_meta(samples, op))

    def test_meta_not_needed_for_basic_filter(self):
        samples = {'text': ['hello']}
        op = TextLengthFilter(min_len=1)
        self.assertFalse(_needs_meta(samples, op))


class NeedsStatsTest(DataJuicerTestCaseBase):
    """Tests for _needs_stats."""

    def test_stats_already_in_samples(self):
        samples = {'text': ['hello'], Fields.stats: [{}]}
        op = TextLengthFilter(min_len=1)
        self.assertTrue(_needs_stats(samples, op))

    def test_stats_needed_for_regular_filter(self):
        # TextLengthFilter is NOT in NON_STATS_FILTERS
        samples = {'text': ['hello']}
        op = TextLengthFilter(min_len=1)
        self.assertTrue(_needs_stats(samples, op))

    def test_stats_not_needed_for_mapper(self):
        samples = {'text': ['hello']}
        op = CleanEmailMapper()
        self.assertFalse(_needs_stats(samples, op))


class EnsureMetaIfNeededTest(DataJuicerTestCaseBase):
    """Tests for _ensure_meta_if_needed."""

    def test_adds_meta_when_present_in_samples_but_empty(self):
        samples = {'text': ['a', 'b'], Fields.meta: []}
        op = CleanEmailMapper()
        # meta key is present so _needs_meta returns True
        result = _ensure_meta_if_needed(samples, op, 'test')
        self.assertEqual(len(result[Fields.meta]), 2)
        self.assertEqual(result[Fields.meta], [{}, {}])

    def test_replaces_none_entries(self):
        samples = {'text': ['a', 'b'], Fields.meta: [None, {'k': 'v'}]}
        op = CleanEmailMapper()
        result = _ensure_meta_if_needed(samples, op, 'test')
        self.assertEqual(result[Fields.meta][0], {})
        self.assertEqual(result[Fields.meta][1], {'k': 'v'})

    def test_raises_on_length_mismatch(self):
        samples = {'text': ['a', 'b', 'c'], Fields.meta: [{}, {}]}
        op = CleanEmailMapper()
        with self.assertRaises(ValueError) as ctx:
            _ensure_meta_if_needed(samples, op, 'test')
        self.assertIn('does not match batch size', str(ctx.exception))

    def test_no_change_when_not_needed(self):
        samples = {'text': ['hello']}
        op = CleanEmailMapper()
        result = _ensure_meta_if_needed(samples, op, 'test')
        self.assertNotIn(Fields.meta, result)


class EnsureStatsIfNeededTest(DataJuicerTestCaseBase):
    """Tests for _ensure_stats_if_needed."""

    def test_adds_stats_for_filter(self):
        samples = {'text': ['a', 'b']}
        op = TextLengthFilter(min_len=1)
        result = _ensure_stats_if_needed(samples, op, 'test')
        self.assertEqual(result[Fields.stats], [{}, {}])

    def test_raises_on_length_mismatch(self):
        samples = {'text': ['a', 'b', 'c'], Fields.stats: [{}, {}]}
        op = TextLengthFilter(min_len=1)
        with self.assertRaises(ValueError) as ctx:
            _ensure_stats_if_needed(samples, op, 'test')
        self.assertIn('does not match batch size', str(ctx.exception))


class RunMapperTest(DataJuicerTestCaseBase):
    """Tests for _run_mapper."""

    def test_clean_email_mapper(self):
        op = CleanEmailMapper()
        batch = {'text': ['contact user@example.com for info', 'no email']}
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True
        )
        result = _run_mapper(op, batch, None, policy, 'test')
        self.assertEqual(result['text'][0], 'contact  for info')
        self.assertEqual(result['text'][1], 'no email')

    def test_punctuation_normalization_mapper(self):
        op = PunctuationNormalizationMapper()
        batch = {'text': ['Hello。World！']}
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True
        )
        result = _run_mapper(op, batch, None, policy, 'test')
        self.assertEqual(result['text'][0], 'Hello.World!')


class RunFilterTest(DataJuicerTestCaseBase):
    """Tests for _run_filter."""

    def test_text_length_filter_keeps_valid(self):
        op = TextLengthFilter(min_len=5, max_len=20)
        batch = {
            'text': ['hi', 'hello world', 'x' * 30],
            Fields.stats: [{}, {}, {}],
        }
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True
        )
        result = _run_filter(op, batch, None, policy, 'test')
        self.assertEqual(result['text'], ['hello world'])

    def test_text_length_filter_removes_all(self):
        op = TextLengthFilter(min_len=100, max_len=200)
        batch = {
            'text': ['short', 'also short'],
            Fields.stats: [{}, {}],
        }
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True
        )
        result = _run_filter(op, batch, None, policy, 'test')
        self.assertEqual(result['text'], [])

    def test_text_length_filter_keeps_all(self):
        op = TextLengthFilter(min_len=1, max_len=1000)
        batch = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True
        )
        result = _run_filter(op, batch, None, policy, 'test')
        self.assertEqual(result['text'], ['hello', 'world'])


class ExecuteSequentialBatchTest(DataJuicerTestCaseBase):
    """Tests for execute_sequential_batch."""

    def test_single_mapper(self):
        ops = [CleanEmailMapper()]
        samples = {'text': ['user@test.com hello', 'no email here']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        self.assertEqual(result['text'], [' hello', 'no email here'])
        # Original not modified
        self.assertEqual(samples['text'], ['user@test.com hello', 'no email here'])

    def test_single_filter(self):
        ops = [TextLengthFilter(min_len=5, max_len=50)]
        samples = {'text': ['hi', 'hello world', 'good morning everyone']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        self.assertEqual(result['text'], ['hello world', 'good morning everyone'])

    def test_chained_mapper_and_filter(self):
        ops = [
            CleanEmailMapper(),
            PunctuationNormalizationMapper(),
            TextLengthFilter(min_len=5, max_len=100),
        ]
        samples = {'text': [
            'email: user@test.com。 This is fine！',
            'x',
            'Hello world, normal text.',
        ]}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        # 'x' is filtered out (too short after processing)
        self.assertIn('email: . This is fine!', result['text'])
        self.assertIn('Hello world, normal text.', result['text'])
        self.assertEqual(len(result['text']), 2)

    def test_copy_input_false_modifies_original(self):
        ops = [CleanEmailMapper()]
        samples = {'text': ['user@test.com hello']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=False, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        self.assertIs(result, samples)
        self.assertEqual(samples['text'], [' hello'])

    def test_copy_input_true_preserves_original(self):
        ops = [CleanEmailMapper()]
        original_text = ['user@test.com hello']
        samples = {'text': original_text[:]}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        self.assertEqual(samples['text'], ['user@test.com hello'])
        self.assertEqual(result['text'], [' hello'])

    def test_shared_context_added_and_removed(self):
        ops = [CleanEmailMapper()]
        samples = {'text': ['hello@test.com world']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, shared_context=True,
            use_op_wrappers=False, validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        self.assertNotIn(Fields.context, result)
        self.assertEqual(result['text'], [' world'])

    def test_cleanup_columns(self):
        ops = [CleanEmailMapper()]
        samples = {'text': ['user@test.com hi'], 'extra': ['data']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy,
            cleanup_columns=['extra'],
        )
        self.assertNotIn('extra', result)
        self.assertIn('text', result)

    def test_on_op_complete_callback(self):
        timings = []

        def on_complete(op, ms):
            timings.append((op._name, ms))

        ops = [CleanEmailMapper(), PunctuationNormalizationMapper()]
        samples = {'text': ['user@test.com。Hello']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy,
            on_op_complete=on_complete,
        )
        self.assertEqual(len(timings), 2)
        self.assertEqual(timings[0][0], 'clean_email_mapper')
        self.assertEqual(timings[1][0], 'punctuation_normalization_mapper')
        # Timings should be positive
        self.assertGreater(timings[0][1], 0)
        self.assertGreater(timings[1][1], 0)

    def test_early_termination_on_empty_batch(self):
        """If a filter removes all samples, subsequent ops are skipped."""
        call_count = []

        class CountingMapper(Mapper):
            _name = 'counting_mapper'
            _batched_op = True

            def process_batched(self, samples, *args, **kwargs):
                call_count.append(1)
                return samples

        ops = [
            TextLengthFilter(min_len=100, max_len=200),
            CountingMapper(),
        ]
        samples = {'text': ['short', 'tiny']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        self.assertEqual(result['text'], [])
        # CountingMapper should not have been called
        self.assertEqual(len(call_count), 0)

    def test_unsupported_op_type_raises(self):
        class FakeDedup(Deduplicator):
            _name = 'fake_dedup'

            def compute_hash(self, sample):
                return sample

            def process(self, dataset, show_num=0):
                return dataset, []

        ops = [FakeDedup()]
        samples = {'text': ['hello']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            execute_sequential_batch(
                samples, ops, owner_name='test', policy=policy
            )
        self.assertIn('does not support op', str(ctx.exception))
        self.assertIn('fake_dedup', str(ctx.exception))

    def test_general_fused_execution_policy(self):
        ops = [CleanEmailMapper(), PunctuationNormalizationMapper()]
        samples = {'text': ['user@test.com。Hello！']}
        result = execute_sequential_batch(
            samples, ops, owner_name='general',
            policy=GENERAL_FUSED_EXECUTION_POLICY,
        )
        self.assertEqual(result['text'], ['.Hello!'])
        self.assertNotIn(Fields.context, result)

    def test_empty_ops_list(self):
        samples = {'text': ['hello', 'world']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, [], owner_name='test', policy=policy
        )
        self.assertEqual(result['text'], ['hello', 'world'])

    def test_multiple_filters_progressive_reduction(self):
        ops = [
            TextLengthFilter(min_len=3, max_len=100),
            TextLengthFilter(min_len=10, max_len=100),
        ]
        samples = {'text': ['hi', 'hello', 'hello world everyone']}
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=True,
        )
        result = execute_sequential_batch(
            samples, ops, owner_name='test', policy=policy
        )
        # 'hi' filtered by first (len 2 < 3)
        # 'hello' filtered by second (len 5 < 10)
        # 'hello world everyone' passes both
        self.assertEqual(result['text'], ['hello world everyone'])


class CleanupContextRowsTest(DataJuicerTestCaseBase):
    """Tests for _cleanup_context_rows."""

    def test_empty_list(self):
        # Should not raise
        _cleanup_context_rows([])

    def test_non_dict_entries(self):
        # Should handle gracefully
        _cleanup_context_rows([None, 'string', 42])

    def test_dict_entries_no_av(self):
        # Should not raise when no av objects
        _cleanup_context_rows([{'key': 'value'}, {'other': [1, 2, 3]}])

    def test_deduplicates_by_id(self):
        # Same object referenced multiple times should only be processed once
        shared_obj = {'data': 'shared'}
        contexts = [{'a': shared_obj}, {'b': shared_obj}]
        # Should not raise
        _cleanup_context_rows(contexts)


if __name__ == '__main__':
    unittest.main()
