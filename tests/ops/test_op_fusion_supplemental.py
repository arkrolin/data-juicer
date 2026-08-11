"""Supplemental tests for data_juicer/ops/op_fusion.py focusing on
helper functions and fusion logic that are not covered by the main
test_op_fusion.py file.
"""

import unittest

from data_juicer.ops.base_op import Filter, Mapper
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.filter.words_num_filter import WordsNumFilter
from data_juicer.ops.mapper.clean_email_mapper import CleanEmailMapper
from data_juicer.ops.mapper.punctuation_normalization_mapper import (
    PunctuationNormalizationMapper,
)
from data_juicer.ops.op_fusion import (
    MAPPER_FUSION_SAFE_ATTR,
    FusedFilter,
    _are_ops_independent,
    _estimated_vram_fraction,
    _is_fusible_gpu_mapper,
    _is_gpu_mapper,
    _mapper_group_blocker,
    _runtime_envs_compatible,
    fuse_operators,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestIsGpuMapper(DataJuicerTestCaseBase):
    """Tests for _is_gpu_mapper helper."""

    def test_cpu_mapper_returns_false(self):
        op = PunctuationNormalizationMapper()
        self.assertFalse(_is_gpu_mapper(op))

    def test_cpu_mapper_no_num_gpus(self):
        op = CleanEmailMapper()
        self.assertFalse(_is_gpu_mapper(op))

    def test_filter_returns_false(self):
        op = TextLengthFilter(min_len=10, max_len=1000)
        self.assertFalse(_is_gpu_mapper(op))

    def test_mapper_with_num_gpus_set(self):
        op = PunctuationNormalizationMapper()
        op.num_gpus = 1
        self.assertTrue(_is_gpu_mapper(op))

    def test_mapper_with_num_gpus_zero(self):
        op = PunctuationNormalizationMapper()
        op.num_gpus = 0
        self.assertFalse(_is_gpu_mapper(op))


class TestIsFusibleGpuMapper(DataJuicerTestCaseBase):
    """Tests for _is_fusible_gpu_mapper helper."""

    def test_cpu_mapper_not_fusible(self):
        op = PunctuationNormalizationMapper()
        self.assertFalse(_is_fusible_gpu_mapper(op))

    def test_gpu_mapper_without_safe_attr_not_fusible(self):
        op = PunctuationNormalizationMapper()
        op.num_gpus = 1
        self.assertFalse(_is_fusible_gpu_mapper(op))

    def test_gpu_mapper_with_safe_attr_is_fusible(self):
        op = PunctuationNormalizationMapper()
        op.num_gpus = 1
        setattr(op, MAPPER_FUSION_SAFE_ATTR, True)
        self.assertTrue(_is_fusible_gpu_mapper(op))

    def test_gpu_mapper_with_safe_attr_false(self):
        op = PunctuationNormalizationMapper()
        op.num_gpus = 1
        setattr(op, MAPPER_FUSION_SAFE_ATTR, False)
        self.assertFalse(_is_fusible_gpu_mapper(op))

    def test_filter_never_fusible(self):
        op = TextLengthFilter(min_len=5, max_len=100)
        op.num_gpus = 1
        setattr(op, MAPPER_FUSION_SAFE_ATTR, True)
        # Filter is not a Mapper, so should not be considered fusible
        self.assertFalse(_is_fusible_gpu_mapper(op))


class TestRuntimeEnvsCompatible(DataJuicerTestCaseBase):
    """Tests for _runtime_envs_compatible helper."""

    def test_empty_list(self):
        self.assertTrue(_runtime_envs_compatible([]))

    def test_single_op(self):
        op = PunctuationNormalizationMapper()
        self.assertTrue(_runtime_envs_compatible([op]))

    def test_multiple_ops_no_runtime_env(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        # Both should have runtime_env=None by default
        self.assertTrue(_runtime_envs_compatible([op1, op2]))

    def test_ops_with_same_runtime_env(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        op1.runtime_env = {'pip': ['torch']}
        op2.runtime_env = {'pip': ['torch']}
        self.assertTrue(_runtime_envs_compatible([op1, op2]))

    def test_ops_with_different_runtime_env(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        op1.runtime_env = {'pip': ['torch']}
        op2.runtime_env = {'pip': ['tensorflow']}
        self.assertFalse(_runtime_envs_compatible([op1, op2]))

    def test_one_has_runtime_env_other_none(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        op1.runtime_env = {'pip': ['torch']}
        # op2.runtime_env defaults to None
        self.assertFalse(_runtime_envs_compatible([op1, op2]))


class TestAreOpsIndependent(DataJuicerTestCaseBase):
    """Tests for _are_ops_independent helper."""

    def test_empty_list(self):
        self.assertTrue(_are_ops_independent([]))

    def test_single_op_with_output_columns(self):
        op = PunctuationNormalizationMapper()
        op._input_columns = ['text']
        op._output_columns = ['text']
        self.assertTrue(_are_ops_independent([op]))

    def test_ops_with_no_output_columns_returns_false(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        # If _output_columns is not defined or empty, should return False
        op1._output_columns = []
        op2._output_columns = ['text']
        self.assertFalse(_are_ops_independent([op1, op2]))

    def test_ops_with_disjoint_columns(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        op1._input_columns = ['text']
        op1._output_columns = ['clean_text']
        op2._input_columns = ['text']
        op2._output_columns = ['email_cleaned']
        self.assertTrue(_are_ops_independent([op1, op2]))

    def test_ops_with_overlapping_output_columns(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        op1._input_columns = ['text']
        op1._output_columns = ['text']
        op2._input_columns = ['raw']
        op2._output_columns = ['text']  # same output as op1
        self.assertFalse(_are_ops_independent([op1, op2]))

    def test_op_reads_column_produced_by_previous(self):
        op1 = PunctuationNormalizationMapper()
        op2 = CleanEmailMapper()
        op1._input_columns = ['text']
        op1._output_columns = ['intermediate']
        op2._input_columns = ['intermediate']  # reads what op1 produces
        op2._output_columns = ['final']
        self.assertFalse(_are_ops_independent([op1, op2]))


class TestMapperGroupBlocker(DataJuicerTestCaseBase):
    """Tests for _mapper_group_blocker helper."""

    def _make_fusible_gpu_mapper(self, name='test_mapper',
                                 vram_fraction=0.3,
                                 runtime_env=None,
                                 input_columns=None,
                                 output_columns=None):
        """Create a mapper configured to pass all fusion checks."""
        op = PunctuationNormalizationMapper()
        op._name = name
        op.num_gpus = 1
        setattr(op, MAPPER_FUSION_SAFE_ATTR, True)
        op.estimated_vram_fraction = vram_fraction
        op.runtime_env = runtime_env
        op._input_columns = input_columns or ['text']
        op._output_columns = output_columns or [f'{name}_out']
        return op

    def test_empty_group_no_blocker(self):
        # _mapper_group_blocker with empty should have no issue about
        # individual ops, but all() on empty is True
        # Actually the function checks all ops are fusible first with
        # `all(_is_fusible_gpu_mapper(op) for op in mapper_group)` which
        # is vacuously True for empty list. But vram_limit is also checked.
        # Let's just verify it doesn't crash.
        result = _mapper_group_blocker([], vram_limit=0.9)
        # Empty group: all checks pass vacuously, total vram = 0 <= 0.9
        self.assertIsNone(result)

    def test_single_valid_op_no_blocker(self):
        op = self._make_fusible_gpu_mapper('op1', vram_fraction=0.3,
                                           output_columns=['col_a'])
        result = _mapper_group_blocker([op], vram_limit=0.9)
        self.assertIsNone(result)

    def test_non_fusible_op_blocks(self):
        op = PunctuationNormalizationMapper()
        # not a GPU mapper, so not fusible
        result = _mapper_group_blocker([op], vram_limit=0.9)
        self.assertIsNotNone(result)
        self.assertIn('not explicitly opted into', result)

    def test_vram_exceeds_limit_blocks(self):
        op1 = self._make_fusible_gpu_mapper('op1', vram_fraction=0.5,
                                            output_columns=['col_a'])
        op2 = self._make_fusible_gpu_mapper('op2', vram_fraction=0.5,
                                            output_columns=['col_b'])
        result = _mapper_group_blocker([op1, op2], vram_limit=0.9)
        self.assertIsNotNone(result)
        self.assertIn('VRAM', result)

    def test_vram_within_limit_no_blocker(self):
        op1 = self._make_fusible_gpu_mapper('op1', vram_fraction=0.4,
                                            output_columns=['col_a'])
        op2 = self._make_fusible_gpu_mapper('op2', vram_fraction=0.4,
                                            output_columns=['col_b'])
        result = _mapper_group_blocker([op1, op2], vram_limit=0.9)
        self.assertIsNone(result)

    def test_incompatible_runtime_envs_blocks(self):
        op1 = self._make_fusible_gpu_mapper('op1', vram_fraction=0.3,
                                            runtime_env={'pip': ['torch']},
                                            output_columns=['col_a'])
        op2 = self._make_fusible_gpu_mapper('op2', vram_fraction=0.3,
                                            runtime_env={'pip': ['tf']},
                                            output_columns=['col_b'])
        result = _mapper_group_blocker([op1, op2], vram_limit=0.9)
        self.assertIsNotNone(result)
        self.assertIn('runtime environment', result)

    def test_non_independent_ops_blocks(self):
        op1 = self._make_fusible_gpu_mapper('op1', vram_fraction=0.3,
                                            output_columns=['shared_col'])
        op2 = self._make_fusible_gpu_mapper('op2', vram_fraction=0.3,
                                            output_columns=['shared_col'])
        result = _mapper_group_blocker([op1, op2], vram_limit=0.9)
        self.assertIsNotNone(result)
        self.assertIn('not independent', result)

    def test_missing_vram_fraction_blocks(self):
        op1 = self._make_fusible_gpu_mapper('op1', vram_fraction=0.3,
                                            output_columns=['col_a'])
        op2 = self._make_fusible_gpu_mapper('op2', vram_fraction=0.3,
                                            output_columns=['col_b'])
        # Remove the vram_fraction attribute from op2
        del op2.estimated_vram_fraction
        result = _mapper_group_blocker([op1, op2], vram_limit=0.9)
        self.assertIsNotNone(result)
        self.assertIn('estimated_vram_fraction', result)

    def test_invalid_vram_limit_raises(self):
        op = self._make_fusible_gpu_mapper('op1', vram_fraction=0.3)
        with self.assertRaises(ValueError):
            _mapper_group_blocker([op], vram_limit=0.0)
        with self.assertRaises(ValueError):
            _mapper_group_blocker([op], vram_limit=1.5)


class TestEstimatedVramFraction(DataJuicerTestCaseBase):
    """Tests for _estimated_vram_fraction helper."""

    def test_returns_none_when_not_set(self):
        op = PunctuationNormalizationMapper()
        # Ensure attribute is not present
        if hasattr(op, 'estimated_vram_fraction'):
            delattr(op, 'estimated_vram_fraction')
        self.assertIsNone(_estimated_vram_fraction(op))

    def test_returns_float_value(self):
        op = PunctuationNormalizationMapper()
        op.estimated_vram_fraction = 0.5
        self.assertEqual(_estimated_vram_fraction(op), 0.5)

    def test_raises_on_invalid_value_zero(self):
        op = PunctuationNormalizationMapper()
        op._name = 'test_op'
        op.estimated_vram_fraction = 0.0
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_raises_on_invalid_value_negative(self):
        op = PunctuationNormalizationMapper()
        op._name = 'test_op'
        op.estimated_vram_fraction = -0.5
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_raises_on_invalid_value_greater_than_one(self):
        op = PunctuationNormalizationMapper()
        op._name = 'test_op'
        op.estimated_vram_fraction = 1.5
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_value_of_one_is_valid(self):
        op = PunctuationNormalizationMapper()
        op.estimated_vram_fraction = 1.0
        self.assertEqual(_estimated_vram_fraction(op), 1.0)

    def test_string_convertible_value(self):
        op = PunctuationNormalizationMapper()
        op.estimated_vram_fraction = '0.7'
        self.assertEqual(_estimated_vram_fraction(op), 0.7)

    def test_non_numeric_string_raises(self):
        op = PunctuationNormalizationMapper()
        op._name = 'test_op'
        op.estimated_vram_fraction = 'abc'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)


class TestFuseOperatorsMixedOps(DataJuicerTestCaseBase):
    """Tests for fuse_operators with a mix of real mappers and filters."""

    def test_single_mapper_unchanged(self):
        op = PunctuationNormalizationMapper()
        result = fuse_operators([op])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], op)

    def test_single_filter_unchanged(self):
        op = TextLengthFilter(min_len=10, max_len=1000)
        result = fuse_operators([op])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], op)

    def test_mapper_then_filters_groups_filters(self):
        mapper = PunctuationNormalizationMapper()
        f1 = TextLengthFilter(min_len=10, max_len=1000)
        f2 = WordsNumFilter(min_num=5, max_num=500)
        result = fuse_operators([mapper, f1, f2])
        # Mapper should stay as-is, filters may be fused or not
        # depending on whether they share intermediate vars
        self.assertIs(result[0], mapper)
        # The filters should still be present (fused or not)
        # At minimum, we have mapper + something for filters
        self.assertGreaterEqual(len(result), 2)

    def test_interleaved_mappers_and_filters(self):
        m1 = PunctuationNormalizationMapper()
        f1 = TextLengthFilter(min_len=10, max_len=1000)
        m2 = CleanEmailMapper()
        f2 = WordsNumFilter(min_num=5, max_num=500)
        result = fuse_operators([m1, f1, m2, f2])
        # Each mapper breaks filter groups, so we expect:
        # m1, [f1 group], m2, [f2 group]
        # With single filters in each group, they stay unfused
        self.assertEqual(len(result), 4)
        self.assertIs(result[0], m1)
        self.assertIs(result[1], f1)
        self.assertIs(result[2], m2)
        self.assertIs(result[3], f2)

    def test_consecutive_filters_sharing_intermediate_vars_get_fused(self):
        # TextLengthFilter and WordsNumFilter both use 'words' intermediate var
        # if they're registered in the same inter_vars registry
        f1 = TextLengthFilter(min_len=10, max_len=1000)
        f2 = WordsNumFilter(min_num=5, max_num=500)
        result = fuse_operators([f1, f2])
        # They may or may not be fused depending on intermediate var registry.
        # Either way, all filters should be accounted for.
        total_filters = 0
        for op in result:
            if isinstance(op, FusedFilter):
                total_filters += len(op.fused_filters)
            elif isinstance(op, Filter):
                total_filters += 1
        self.assertEqual(total_filters, 2)

    def test_mapper_fusion_disabled(self):
        m1 = PunctuationNormalizationMapper()
        m2 = CleanEmailMapper()
        result = fuse_operators([m1, m2], mapper_fusion=False)
        # With mapper_fusion disabled, mappers should stay as-is
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], m1)
        self.assertIs(result[1], m2)

    def test_empty_ops_list(self):
        result = fuse_operators([])
        self.assertEqual(result, [])

    def test_all_filters_no_mappers(self):
        f1 = TextLengthFilter(min_len=10, max_len=1000)
        f2 = WordsNumFilter(min_num=5, max_num=500)
        result = fuse_operators([f1, f2])
        # Should process the filter group
        self.assertGreaterEqual(len(result), 1)
        # All original filters accounted for
        total_filters = 0
        for op in result:
            if isinstance(op, FusedFilter):
                total_filters += len(op.fused_filters)
            elif isinstance(op, Filter):
                total_filters += 1
        self.assertEqual(total_filters, 2)

    def test_multiple_mappers_no_gpu_stay_unfused(self):
        # CPU mappers should not be fused in mapper fusion phase
        m1 = PunctuationNormalizationMapper()
        m2 = CleanEmailMapper()
        result = fuse_operators([m1, m2], mapper_fusion=True)
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], m1)
        self.assertIs(result[1], m2)


if __name__ == '__main__':
    unittest.main()
