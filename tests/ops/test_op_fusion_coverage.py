"""Tests for data_juicer/ops/op_fusion.py mapper fusion logic."""
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.ops.op_fusion import (
    _are_ops_independent,
    _estimated_vram_fraction,
    _is_fusible_gpu_mapper,
    _is_gpu_mapper,
    _mapper_group_blocker,
    _runtime_envs_compatible,
    fuse_consecutive_mappers,
    fuse_mapper_group,
    fuse_operators,
    MAPPER_FUSION_SAFE_ATTR,
)


def _make_filter(name='filter1', speed=None):
    from data_juicer.ops.base_op import Filter
    op = MagicMock(spec=Filter)
    op.__class__ = Filter
    op._name = name
    op._op_cfg = {name: {}}
    op.accelerator = 'cpu'
    op.runtime_np.return_value = 4
    return op


def _make_mapper(name='mapper1', num_gpus=0, fusible=False, vram=None,
                 input_cols=None, output_cols=None, runtime_env=None,
                 num_cpus=None, batch_size=1, num_proc=4):
    from data_juicer.ops.base_op import Mapper
    op = MagicMock(spec=Mapper)
    op.__class__ = Mapper
    op._name = name
    op._op_cfg = {name: {}}
    op.num_gpus = num_gpus
    op.accelerator = 'cuda' if num_gpus else 'cpu'
    op.runtime_np.return_value = num_proc
    op.batch_size = batch_size
    op.num_cpus = num_cpus
    op.runtime_env = runtime_env
    op._input_columns = input_cols or []
    op._output_columns = output_cols or []
    if fusible:
        setattr(op, MAPPER_FUSION_SAFE_ATTR, True)
    else:
        setattr(op, MAPPER_FUSION_SAFE_ATTR, False)
    if vram is not None:
        op.estimated_vram_fraction = vram
    else:
        op.estimated_vram_fraction = None
    return op


class TestIsGpuMapper(unittest.TestCase):

    def test_gpu_mapper(self):
        op = _make_mapper(num_gpus=1)
        self.assertTrue(_is_gpu_mapper(op))

    def test_cpu_mapper(self):
        op = _make_mapper(num_gpus=0)
        self.assertFalse(_is_gpu_mapper(op))

    def test_filter_not_mapper(self):
        op = _make_filter()
        self.assertFalse(_is_gpu_mapper(op))


class TestIsFusibleGpuMapper(unittest.TestCase):

    def test_fusible(self):
        op = _make_mapper(num_gpus=1, fusible=True)
        self.assertTrue(_is_fusible_gpu_mapper(op))

    def test_not_fusible(self):
        op = _make_mapper(num_gpus=1, fusible=False)
        self.assertFalse(_is_fusible_gpu_mapper(op))

    def test_cpu_not_fusible(self):
        op = _make_mapper(num_gpus=0, fusible=True)
        self.assertFalse(_is_fusible_gpu_mapper(op))


class TestEstimatedVramFraction(unittest.TestCase):

    def test_valid_fraction(self):
        op = _make_mapper(vram=0.5)
        self.assertEqual(_estimated_vram_fraction(op), 0.5)

    def test_none_fraction(self):
        op = _make_mapper(vram=None)
        self.assertIsNone(_estimated_vram_fraction(op))

    def test_invalid_zero(self):
        op = _make_mapper(vram=0.0)
        op._name = 'bad_op'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_invalid_greater_than_one(self):
        op = _make_mapper(vram=1.5)
        op._name = 'bad_op'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_invalid_type(self):
        op = _make_mapper()
        op.estimated_vram_fraction = "not_a_number"
        op._name = 'bad_op'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)


class TestRuntimeEnvsCompatible(unittest.TestCase):

    def test_empty_list(self):
        self.assertTrue(_runtime_envs_compatible([]))

    def test_same_env(self):
        ops = [_make_mapper(runtime_env={'pip': ['torch']}),
               _make_mapper(runtime_env={'pip': ['torch']})]
        self.assertTrue(_runtime_envs_compatible(ops))

    def test_different_envs(self):
        ops = [_make_mapper(runtime_env={'pip': ['torch']}),
               _make_mapper(runtime_env={'pip': ['jax']})]
        self.assertFalse(_runtime_envs_compatible(ops))

    def test_none_envs(self):
        ops = [_make_mapper(runtime_env=None), _make_mapper(runtime_env=None)]
        self.assertTrue(_runtime_envs_compatible(ops))


class TestAreOpsIndependent(unittest.TestCase):

    def test_disjoint_outputs(self):
        ops = [_make_mapper(output_cols=['col_a']),
               _make_mapper(output_cols=['col_b'])]
        self.assertTrue(_are_ops_independent(ops))

    def test_overlapping_outputs(self):
        ops = [_make_mapper(output_cols=['col_a']),
               _make_mapper(output_cols=['col_a'])]
        self.assertFalse(_are_ops_independent(ops))

    def test_reads_produced_col(self):
        ops = [_make_mapper(output_cols=['col_a']),
               _make_mapper(input_cols=['col_a'], output_cols=['col_b'])]
        self.assertFalse(_are_ops_independent(ops))

    def test_no_output_cols_declared(self):
        op = _make_mapper()
        op._output_columns = []
        self.assertFalse(_are_ops_independent([op]))


class TestMapperGroupBlocker(unittest.TestCase):

    def test_no_blocker(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a']),
               _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['b'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIsNone(result)

    def test_not_fusible(self):
        ops = [_make_mapper(num_gpus=1, fusible=False, vram=0.3, output_cols=['a'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("not explicitly opted", result)

    def test_not_independent(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a']),
               _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("not independent", result)

    def test_different_runtime_envs(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a'],
                            runtime_env={'pip': ['torch']}),
               _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['b'],
                            runtime_env={'pip': ['jax']})]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("runtime environments", result)

    def test_missing_vram_estimate(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=None, output_cols=['a'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("estimated_vram_fraction", result)

    def test_vram_over_limit(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.6, output_cols=['a']),
               _make_mapper(num_gpus=1, fusible=True, vram=0.6, output_cols=['b'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("exceeds", result)

    def test_invalid_vram_limit(self):
        with self.assertRaises(ValueError):
            _mapper_group_blocker([], 0.0)
        with self.assertRaises(ValueError):
            _mapper_group_blocker([], 1.5)


class TestFuseMapperGroup(unittest.TestCase):

    def test_empty_group(self):
        result = fuse_mapper_group([])
        self.assertEqual(result, [])

    def test_blocked_returns_original(self):
        ops = [_make_mapper(num_gpus=1, fusible=False, vram=0.3, output_cols=['a'])]
        result = fuse_mapper_group(ops)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ops[0])

    def test_successful_fusion(self):
        mock_fused_cls = MagicMock()
        mock_fused_instance = MagicMock()
        mock_fused_cls.return_value = mock_fused_instance
        mock_module = MagicMock()
        mock_module.FusedSequentialBatchOp = mock_fused_cls
        ops = [_make_mapper(name='op1', num_gpus=1, fusible=True, vram=0.3,
                            output_cols=['a'], num_cpus=2, batch_size=4),
               _make_mapper(name='op2', num_gpus=1, fusible=True, vram=0.3,
                            output_cols=['b'], num_cpus=4, batch_size=2)]
        with patch.dict('sys.modules', {'data_juicer.ops.fused_sequential_batch_op': mock_module}):
            result = fuse_mapper_group(ops, vram_limit=0.9)
        self.assertEqual(len(result), 1)
        mock_fused_cls.assert_called_once()


class TestFuseConsecutiveMappers(unittest.TestCase):

    def test_no_gpu_mappers(self):
        ops = [_make_mapper(num_gpus=0), _make_mapper(num_gpus=0)]
        result = fuse_consecutive_mappers(ops)
        self.assertEqual(len(result), 2)

    def test_single_gpu_mapper_passes_through(self):
        op = _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])
        result = fuse_consecutive_mappers([op])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], op)

    def test_invalid_vram_limit_raises(self):
        with self.assertRaises(ValueError):
            fuse_consecutive_mappers([], vram_limit=0.0)

    @patch('data_juicer.ops.op_fusion.fuse_mapper_group')
    def test_consecutive_fusible_mappers(self, mock_fuse):
        mock_fuse.return_value = [MagicMock()]
        op1 = _make_mapper(name='g1', num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])
        op2 = _make_mapper(name='g2', num_gpus=1, fusible=True, vram=0.3, output_cols=['b'])
        result = fuse_consecutive_mappers([op1, op2])
        mock_fuse.assert_called_once()

    def test_non_gpu_breaks_group(self):
        op1 = _make_mapper(name='g1', num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])
        cpu_op = _make_mapper(name='cpu', num_gpus=0)
        op2 = _make_mapper(name='g2', num_gpus=1, fusible=True, vram=0.3, output_cols=['b'])
        result = fuse_consecutive_mappers([op1, cpu_op, op2])
        # op1 alone (< 2 so passes through), cpu_op passes through, op2 alone passes through
        self.assertEqual(len(result), 3)


class TestFuseOperators(unittest.TestCase):

    def test_empty_ops(self):
        result = fuse_operators([])
        self.assertEqual(result, [])

    def test_non_filter_passes_through(self):
        mapper = _make_mapper(name='m1')
        result = fuse_operators([mapper])
        self.assertEqual(len(result), 1)

    def test_single_filter_no_fusion(self):
        f = _make_filter(name='f1')
        # Need to properly set up the filter for fuse_filter_group
        # Since fuse_filter_group checks INTER_VARS, and the mock won't be in any,
        # it just appends directly
        result = fuse_operators([f], probe_res=[None])
        self.assertEqual(len(result), 1)

    def test_mapper_fusion_disabled(self):
        mapper = _make_mapper(name='m1')
        result = fuse_operators([mapper], mapper_fusion=False)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
