"""Tests for data_juicer.ops.fused_batch_executor module."""

import unittest
from unittest.mock import MagicMock, patch

from data_juicer.ops.base_op import NON_STATS_FILTERS, TAGGING_OPS, Filter, Mapper
from data_juicer.ops.fused_batch_executor import (
    GENERAL_FUSED_EXECUTION_POLICY,
    SequentialBatchExecutionPolicy,
    _cleanup_context_rows,
    _ensure_dict_column,
    _needs_meta,
    _needs_stats,
    _validate_batch,
    execute_sequential_batch,
    get_batch_size,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


class SimpleTestMapper(Mapper):
    _batched_op = True
    _name = 'simple_test_mapper'

    def process_batched(self, samples, **kwargs):
        samples['text'] = [t.upper() for t in samples['text']]
        return samples


class ContextAwareMapper(Mapper):
    _batched_op = True
    _name = 'context_aware_mapper'

    def process_batched(self, samples, context=False, **kwargs):
        if context and Fields.context in samples:
            for i, ctx in enumerate(samples[Fields.context]):
                ctx['processed'] = True
        samples['text'] = [t.upper() for t in samples['text']]
        return samples


class SimpleTestFilter(Filter):
    _batched_op = True
    _name = 'simple_test_filter'

    def compute_stats_batched(self, samples, **kwargs):
        if Fields.stats not in samples:
            samples[Fields.stats] = [{} for _ in samples['text']]
        for i, t in enumerate(samples['text']):
            samples[Fields.stats][i]['length'] = len(t)
        return samples

    def process_batched(self, samples):
        stats = samples.get(Fields.stats, [{}] * len(samples['text']))
        return [s.get('length', 0) > 3 for s in stats]


class NoneReturningMapper(Mapper):
    _batched_op = True
    _name = 'none_returning_mapper'

    def process_batched(self, samples, **kwargs):
        return None


class NonDictReturningMapper(Mapper):
    _batched_op = True
    _name = 'non_dict_returning_mapper'

    def process_batched(self, samples, **kwargs):
        return "not a dict"


class FilterAllMapper(Mapper):
    """A mapper that empties the batch."""
    _batched_op = True
    _name = 'filter_all_mapper'

    def process_batched(self, samples, **kwargs):
        for key in samples:
            samples[key] = []
        return samples


class TestSequentialBatchExecutionPolicy(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_default_values(self):
        policy = SequentialBatchExecutionPolicy()
        self.assertFalse(policy.copy_input)
        self.assertFalse(policy.shared_context)
        self.assertTrue(policy.use_op_wrappers)
        self.assertTrue(policy.validate)
        self.assertTrue(policy.ensure_fields)

    @TEST_TAG("standalone")
    def test_custom_values(self):
        policy = SequentialBatchExecutionPolicy(
            copy_input=True,
            shared_context=True,
            use_op_wrappers=False,
            validate=False,
            ensure_fields=False,
        )
        self.assertTrue(policy.copy_input)
        self.assertTrue(policy.shared_context)
        self.assertFalse(policy.use_op_wrappers)
        self.assertFalse(policy.validate)
        self.assertFalse(policy.ensure_fields)

    @TEST_TAG("standalone")
    def test_frozen(self):
        policy = SequentialBatchExecutionPolicy()
        with self.assertRaises(Exception):
            policy.copy_input = True

    @TEST_TAG("standalone")
    def test_general_fused_execution_policy(self):
        self.assertTrue(GENERAL_FUSED_EXECUTION_POLICY.copy_input)
        self.assertTrue(GENERAL_FUSED_EXECUTION_POLICY.shared_context)
        self.assertFalse(GENERAL_FUSED_EXECUTION_POLICY.use_op_wrappers)
        self.assertFalse(GENERAL_FUSED_EXECUTION_POLICY.validate)
        self.assertFalse(GENERAL_FUSED_EXECUTION_POLICY.ensure_fields)


class TestGetBatchSize(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_empty_dict(self):
        self.assertEqual(get_batch_size({}), 0)

    @TEST_TAG("standalone")
    def test_normal_dict(self):
        samples = {'text': ['hello', 'world', 'foo'], 'id': [1, 2, 3]}
        self.assertEqual(get_batch_size(samples), 3)

    @TEST_TAG("standalone")
    def test_none(self):
        self.assertEqual(get_batch_size(None), 0)

    @TEST_TAG("standalone")
    def test_single_element(self):
        samples = {'text': ['hello']}
        self.assertEqual(get_batch_size(samples), 1)


class TestExecuteSequentialBatch(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_simple_mapper(self):
        samples = {'text': ['hello', 'world']}
        mapper = SimpleTestMapper()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=False
        )
        result = execute_sequential_batch(
            samples, [mapper], policy=policy, owner_name="test"
        )
        self.assertEqual(result['text'], ['HELLO', 'WORLD'])

    @TEST_TAG("standalone")
    def test_simple_filter(self):
        samples = {'text': ['hi', 'hello', 'ab', 'world']}
        filt = SimpleTestFilter()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=True
        )
        result = execute_sequential_batch(
            samples, [filt], policy=policy, owner_name="test"
        )
        # Only 'hello' (5) and 'world' (5) pass length > 3
        self.assertEqual(result['text'], ['hello', 'world'])

    @TEST_TAG("standalone")
    def test_mixed_ops(self):
        samples = {'text': ['hi', 'hello', 'ab', 'world']}
        mapper = SimpleTestMapper()
        filt = SimpleTestFilter()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=True
        )
        # Mapper uppercases, then filter keeps length > 3
        result = execute_sequential_batch(
            samples, [mapper, filt], policy=policy, owner_name="test"
        )
        # After mapper: ['HI', 'HELLO', 'AB', 'WORLD']
        # Filter keeps length > 3: 'HELLO' (5), 'WORLD' (5)
        self.assertEqual(result['text'], ['HELLO', 'WORLD'])

    @TEST_TAG("standalone")
    def test_cleanup_columns(self):
        samples = {'text': ['hello'], 'extra': ['data']}
        mapper = SimpleTestMapper()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=False
        )
        result = execute_sequential_batch(
            samples, [mapper], policy=policy,
            cleanup_columns=['extra'], owner_name="test"
        )
        self.assertNotIn('extra', result)
        self.assertEqual(result['text'], ['HELLO'])

    @TEST_TAG("standalone")
    def test_on_op_complete_callback(self):
        samples = {'text': ['hello', 'world']}
        mapper = SimpleTestMapper()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=False
        )
        callback_calls = []

        def on_complete(op, elapsed_ms):
            callback_calls.append((op, elapsed_ms))

        result = execute_sequential_batch(
            samples, [mapper], policy=policy,
            on_op_complete=on_complete, owner_name="test"
        )
        self.assertEqual(len(callback_calls), 1)
        self.assertIs(callback_calls[0][0], mapper)
        self.assertIsInstance(callback_calls[0][1], float)
        self.assertGreaterEqual(callback_calls[0][1], 0.0)

    @TEST_TAG("standalone")
    def test_unsupported_op_type_raises(self):
        samples = {'text': ['hello']}
        # Create a mock op that is neither Mapper nor Filter
        mock_op = MagicMock()
        mock_op._name = 'unsupported_op'
        # Make it not an instance of Mapper or Filter
        mock_op.__class__ = type('SomeOtherOp', (), {})

        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=False
        )
        with self.assertRaises(NotImplementedError) as ctx:
            execute_sequential_batch(
                samples, [mock_op], policy=policy, owner_name="test_fused"
            )
        self.assertIn('test_fused', str(ctx.exception))
        self.assertIn('unsupported_op', str(ctx.exception))

    @TEST_TAG("standalone")
    def test_batch_becomes_empty_early_break(self):
        """When batch becomes empty mid-pipeline, remaining ops are skipped."""
        samples = {'text': ['hello', 'world']}
        filter_all = FilterAllMapper()
        # This mapper should never be reached
        mapper = SimpleTestMapper()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=False
        )
        callback_calls = []

        def on_complete(op, elapsed_ms):
            callback_calls.append(op)

        result = execute_sequential_batch(
            samples, [filter_all, mapper], policy=policy,
            on_op_complete=on_complete, owner_name="test"
        )
        # Only filter_all should have been called
        self.assertEqual(len(callback_calls), 1)
        self.assertIs(callback_calls[0], filter_all)
        self.assertEqual(get_batch_size(result), 0)

    @TEST_TAG("standalone")
    def test_copy_input_policy(self):
        """With copy_input=True, original samples should not be mutated."""
        samples = {'text': ['hello', 'world']}
        original_texts = samples['text'][:]
        mapper = SimpleTestMapper()
        policy = SequentialBatchExecutionPolicy(
            copy_input=True, use_op_wrappers=False,
            validate=True, ensure_fields=False
        )
        result = execute_sequential_batch(
            samples, [mapper], policy=policy, owner_name="test"
        )
        # Original should be unchanged
        self.assertEqual(samples['text'], original_texts)
        # Result should be uppercased
        self.assertEqual(result['text'], ['HELLO', 'WORLD'])

    @TEST_TAG("standalone")
    def test_shared_context_added_and_cleaned(self):
        """With shared_context=True, context is added and cleaned up after."""
        samples = {'text': ['hello']}
        mapper = SimpleTestMapper()
        policy = SequentialBatchExecutionPolicy(
            shared_context=True, use_op_wrappers=False,
            validate=True, ensure_fields=False
        )
        result = execute_sequential_batch(
            samples, [mapper], policy=policy, owner_name="test"
        )
        # Context should be removed from the output
        self.assertNotIn(Fields.context, result)

    @TEST_TAG("standalone")
    def test_none_result_raises(self):
        """Mapper returning None raises ValueError."""
        samples = {'text': ['hello']}
        mapper = NoneReturningMapper()
        policy = SequentialBatchExecutionPolicy(
            use_op_wrappers=False, validate=True, ensure_fields=False
        )
        with self.assertRaises(ValueError) as ctx:
            execute_sequential_batch(
                samples, [mapper], policy=policy, owner_name="test"
            )
        self.assertIn('None', str(ctx.exception))


class TestValidateBatch(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_none_result_raises(self):
        op = MagicMock()
        op._name = 'test_op'
        with self.assertRaises(ValueError) as ctx:
            _validate_batch(None, op, "owner", "process", validate=True)
        self.assertIn('None', str(ctx.exception))
        self.assertIn('test_op', str(ctx.exception))

    @TEST_TAG("standalone")
    def test_non_dict_result_raises_when_validate_true(self):
        op = MagicMock()
        op._name = 'test_op'
        with self.assertRaises(ValueError) as ctx:
            _validate_batch("not a dict", op, "owner", "process", validate=True)
        self.assertIn('unsupported batch type', str(ctx.exception))

    @TEST_TAG("standalone")
    def test_non_dict_result_passes_when_validate_false(self):
        op = MagicMock()
        op._name = 'test_op'
        result = _validate_batch("not a dict", op, "owner", "process", validate=False)
        self.assertEqual(result, "not a dict")

    @TEST_TAG("standalone")
    def test_dict_result_passes(self):
        op = MagicMock()
        op._name = 'test_op'
        data = {'text': ['hello']}
        result = _validate_batch(data, op, "owner", "process", validate=True)
        self.assertIs(result, data)

    @TEST_TAG("standalone")
    def test_none_result_raises_even_when_validate_false(self):
        """None always raises regardless of validate flag."""
        op = MagicMock()
        op._name = 'test_op'
        with self.assertRaises(ValueError):
            _validate_batch(None, op, "owner", "process", validate=False)


class TestEnsureDictColumn(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_adds_column_when_missing(self):
        samples = {'text': ['a', 'b', 'c']}
        op = MagicMock()
        op._name = 'test_op'
        result = _ensure_dict_column(samples, Fields.stats, op, "owner")
        self.assertIn(Fields.stats, result)
        self.assertEqual(len(result[Fields.stats]), 3)
        self.assertEqual(result[Fields.stats], [{}, {}, {}])

    @TEST_TAG("standalone")
    def test_adds_column_when_empty_list(self):
        samples = {'text': ['a', 'b'], Fields.stats: []}
        op = MagicMock()
        op._name = 'test_op'
        result = _ensure_dict_column(samples, Fields.stats, op, "owner")
        self.assertEqual(len(result[Fields.stats]), 2)

    @TEST_TAG("standalone")
    def test_fixes_none_values(self):
        samples = {'text': ['a', 'b', 'c'], Fields.stats: [{'x': 1}, None, {'y': 2}]}
        op = MagicMock()
        op._name = 'test_op'
        result = _ensure_dict_column(samples, Fields.stats, op, "owner")
        self.assertEqual(result[Fields.stats][0], {'x': 1})
        self.assertEqual(result[Fields.stats][1], {})
        self.assertEqual(result[Fields.stats][2], {'y': 2})

    @TEST_TAG("standalone")
    def test_raises_on_length_mismatch(self):
        samples = {'text': ['a', 'b', 'c'], Fields.stats: [{}]}
        op = MagicMock()
        op._name = 'test_op'
        with self.assertRaises(ValueError) as ctx:
            _ensure_dict_column(samples, Fields.stats, op, "owner")
        self.assertIn('length', str(ctx.exception))
        self.assertIn('does not match batch size', str(ctx.exception))


class TestNeedsMeta(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_meta_already_in_samples(self):
        samples = {Fields.meta: [{}, {}], 'text': ['a', 'b']}
        op = MagicMock()
        op._name = 'some_op'
        op._requires_meta = False
        op._output_columns = []
        self.assertTrue(_needs_meta(samples, op))

    @TEST_TAG("standalone")
    def test_requires_meta_attribute(self):
        samples = {'text': ['a']}
        op = MagicMock()
        op._name = 'some_op'
        op._requires_meta = True
        op._output_columns = []
        self.assertTrue(_needs_meta(samples, op))

    @TEST_TAG("standalone")
    def test_op_in_tagging_ops(self):
        samples = {'text': ['a']}
        op = MagicMock()
        op._name = 'tagging_op'
        op._requires_meta = False
        op._output_columns = []
        # Register the op name in TAGGING_OPS
        TAGGING_OPS._modules['tagging_op'] = True
        try:
            self.assertTrue(_needs_meta(samples, op))
        finally:
            del TAGGING_OPS._modules['tagging_op']

    @TEST_TAG("standalone")
    def test_output_columns_with_meta_prefix(self):
        samples = {'text': ['a']}
        op = MagicMock()
        op._name = 'some_op'
        op._requires_meta = False
        op._output_columns = [Fields.meta + 'some_field']
        self.assertTrue(_needs_meta(samples, op))

    @TEST_TAG("standalone")
    def test_no_meta_needed(self):
        samples = {'text': ['a']}
        op = MagicMock()
        op._name = 'some_op'
        op._requires_meta = False
        op._output_columns = ['other_field']
        self.assertFalse(_needs_meta(samples, op))


class TestNeedsStats(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_stats_already_in_samples(self):
        samples = {Fields.stats: [{}], 'text': ['a']}
        op = MagicMock(spec=Filter)
        op._name = 'some_filter'
        op._output_columns = []
        self.assertTrue(_needs_stats(samples, op))

    @TEST_TAG("standalone")
    def test_filter_not_in_non_stats(self):
        samples = {'text': ['a']}
        op = MagicMock(spec=Filter)
        op._name = 'some_filter'
        op._output_columns = []
        # Make sure not in NON_STATS_FILTERS
        if 'some_filter' in NON_STATS_FILTERS._modules:
            del NON_STATS_FILTERS._modules['some_filter']
        self.assertTrue(_needs_stats(samples, op))

    @TEST_TAG("standalone")
    def test_filter_in_non_stats(self):
        samples = {'text': ['a']}
        op = MagicMock(spec=Filter)
        op._name = 'non_stats_filter'
        op._output_columns = []
        NON_STATS_FILTERS._modules['non_stats_filter'] = True
        try:
            self.assertFalse(_needs_stats(samples, op))
        finally:
            del NON_STATS_FILTERS._modules['non_stats_filter']

    @TEST_TAG("standalone")
    def test_output_columns_with_stats_prefix(self):
        samples = {'text': ['a']}
        op = MagicMock()
        op._name = 'some_op'
        op._output_columns = [Fields.stats + 'score']
        # Not a Filter instance, stats not in samples
        self.assertTrue(_needs_stats(samples, op))

    @TEST_TAG("standalone")
    def test_no_stats_needed(self):
        samples = {'text': ['a']}
        op = MagicMock()
        op._name = 'some_op'
        op._output_columns = ['other_field']
        self.assertFalse(_needs_stats(samples, op))


class TestCleanupContextRows(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_non_dict_items_skipped(self):
        context_rows = [None, "string", 42, {}, []]
        # Should not raise
        _cleanup_context_rows(context_rows)

    @TEST_TAG("standalone")
    def test_no_av_values_returns_early(self):
        context_rows = [{'key': 'value'}, {'num': 42}]
        # Should not raise, no av values
        _cleanup_context_rows(context_rows)

    @TEST_TAG("standalone")
    def test_empty_list(self):
        _cleanup_context_rows([])

    @TEST_TAG("standalone")
    def test_dict_with_non_av_objects(self):
        class FakeModule:
            __module__ = 'some.other.module'

        obj = FakeModule()
        context_rows = [{'obj': obj}]
        _cleanup_context_rows(context_rows)

    @TEST_TAG("standalone")
    def test_deduplicates_by_id(self):
        """Same object referenced in multiple context dicts is only processed once."""
        shared_obj = {'data': 'shared'}
        context_rows = [
            {'a': shared_obj, 'b': shared_obj},
            {'c': shared_obj}
        ]
        # Should not raise - just verifying deduplication logic
        _cleanup_context_rows(context_rows)


if __name__ == '__main__':
    unittest.main()
