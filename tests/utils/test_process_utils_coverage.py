"""Tests for data_juicer/utils/process_utils.py to boost coverage."""
import math
import sys
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.process_utils import (
    _find_optimal_concurrency,
    calculate_np,
    setup_mp,
    setup_worker_threads,
)


class TestSetupWorkerThreads(unittest.TestCase):

    def setUp(self):
        import data_juicer.utils.process_utils as mod
        self._orig = mod._WORKER_THREADS_CONFIGURED
        mod._WORKER_THREADS_CONFIGURED = False

    def tearDown(self):
        import data_juicer.utils.process_utils as mod
        mod._WORKER_THREADS_CONFIGURED = self._orig

    @patch('data_juicer.utils.process_utils.logger')
    def test_sets_torch_threads(self, mock_logger):
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            setup_worker_threads(num_threads=2)
        mock_torch.set_num_threads.assert_called_once_with(2)
        mock_torch.set_num_interop_threads.assert_called_once_with(2)

    @patch('data_juicer.utils.process_utils.logger')
    def test_import_error_torch(self, mock_logger):
        import data_juicer.utils.process_utils as mod
        mod._WORKER_THREADS_CONFIGURED = False
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                setup_worker_threads(num_threads=1)

    @patch('data_juicer.utils.process_utils.logger')
    def test_runtime_error_torch(self, mock_logger):
        import data_juicer.utils.process_utils as mod
        mod._WORKER_THREADS_CONFIGURED = False
        mock_torch = MagicMock()
        mock_torch.set_num_interop_threads.side_effect = RuntimeError("already set")
        with patch.dict('sys.modules', {'torch': mock_torch}):
            setup_worker_threads(num_threads=1)
        mock_torch.set_num_threads.assert_called_once_with(1)

    def test_only_configures_once(self):
        import data_juicer.utils.process_utils as mod
        mod._WORKER_THREADS_CONFIGURED = True
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            setup_worker_threads(num_threads=4)
        mock_torch.set_num_threads.assert_not_called()


class TestSetupMp(unittest.TestCase):

    @patch('data_juicer.utils.process_utils.mp')
    def test_non_main_process_returns_early(self, mock_mp):
        mock_mp.current_process.return_value.name = "Worker-1"
        setup_mp()
        mock_mp.set_start_method.assert_not_called()

    @patch('data_juicer.utils.process_utils.mp')
    def test_sets_first_available_method(self, mock_mp):
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['fork', 'spawn']
        setup_mp(method='fork')
        mock_mp.set_start_method.assert_called_once_with('fork', force=True)

    @patch('data_juicer.utils.process_utils.mp')
    def test_env_method_override(self, mock_mp):
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['fork', 'spawn', 'forkserver']
        with patch.dict('os.environ', {'MP_START_METHOD': 'spawn'}):
            setup_mp(method=['fork', 'spawn'])
        mock_mp.set_start_method.assert_called_once_with('spawn', force=True)

    @patch('data_juicer.utils.process_utils.mp')
    def test_runtime_error_on_set(self, mock_mp):
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['spawn']
        mock_mp.set_start_method.side_effect = RuntimeError("context already set")
        setup_mp(method='spawn')

    @patch('data_juicer.utils.process_utils.mp')
    def test_method_not_available(self, mock_mp):
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['spawn']
        setup_mp(method='forkserver')
        mock_mp.set_start_method.assert_not_called()

    @patch('data_juicer.utils.process_utils.mp')
    def test_default_method_list(self, mock_mp):
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['fork', 'forkserver', 'spawn']
        setup_mp()
        mock_mp.set_start_method.assert_called_once_with('fork', force=True)


class TestCalculateNp(unittest.TestCase):

    @patch('data_juicer.utils.process_utils.cpu_count', return_value=16)
    @patch('data_juicer.utils.process_utils.available_memories', return_value=[16384, 16384])
    def test_cpu_only_with_memory(self, mock_mem, mock_cpu):
        result = calculate_np("test_op", memory=4, num_cpus=2, use_cuda=False, num_gpus=0)
        self.assertGreater(result, 0)

    @patch('data_juicer.utils.process_utils.cpu_count', return_value=8)
    @patch('data_juicer.utils.process_utils.available_memories', return_value=[8192])
    def test_cpu_only_no_memory(self, mock_mem, mock_cpu):
        result = calculate_np("test_op", memory=0, num_cpus=2, use_cuda=False, num_gpus=0)
        self.assertEqual(result, 4)

    @patch('data_juicer.utils.process_utils.cpu_count', return_value=4)
    @patch('data_juicer.utils.process_utils.available_memories', return_value=[1024])
    def test_cpu_insufficient_resources(self, mock_mem, mock_cpu):
        result = calculate_np("test_op", memory=8, num_cpus=8, use_cuda=False, num_gpus=0)
        self.assertEqual(result, 1)

    @patch('data_juicer.utils.process_utils.cpu_count', return_value=16)
    @patch('data_juicer.utils.process_utils.cuda_device_count', return_value=4)
    @patch('data_juicer.utils.process_utils.available_gpu_memories', return_value=[8192, 8192, 8192, 8192])
    def test_cuda_with_memory_and_gpus(self, mock_gpu_mem, mock_cuda_count, mock_cpu):
        result = calculate_np("test_op", memory=2, num_cpus=2, use_cuda=True, num_gpus=1)
        self.assertGreater(result, 0)

    @patch('data_juicer.utils.process_utils.cpu_count', return_value=16)
    @patch('data_juicer.utils.process_utils.cuda_device_count', return_value=4)
    @patch('data_juicer.utils.process_utils.available_gpu_memories', return_value=[8192, 8192, 8192, 8192])
    def test_cuda_no_memory_no_gpus(self, mock_gpu_mem, mock_cuda_count, mock_cpu):
        result = calculate_np("test_op", memory=0, num_cpus=0, use_cuda=True, num_gpus=0)
        self.assertEqual(result, 4)

    def test_use_cuda_false_but_num_gpus_raises(self):
        with self.assertRaises(ValueError) as ctx:
            calculate_np("test_op", memory=0, num_cpus=0, use_cuda=False, num_gpus=2)
        self.assertIn("GPU resources", str(ctx.exception))


class TestFindOptimalConcurrency(unittest.TestCase):

    def test_empty_input(self):
        result = _find_optimal_concurrency([], 1.0)
        self.assertEqual(result, (None, 0, 0))

    def test_all_zero_ratios(self):
        result = _find_optimal_concurrency([0, 0, 0], 1.0)
        self.assertEqual(result, (None, 0, 0))

    def test_single_operator(self):
        combo, usage, std = _find_optimal_concurrency([0.25], 1.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 1)
        self.assertGreater(usage, 0)

    def test_two_operators_equal_resource(self):
        combo, usage, std = _find_optimal_concurrency([0.2, 0.2], 1.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 2)
        self.assertLessEqual(usage, 1.0 + 1e-10)

    def test_resource_constraint_respected(self):
        combo, usage, std = _find_optimal_concurrency([0.5, 0.5], 0.8)
        if combo is not None:
            total_used = sum(c * r for c, r in zip(combo, [0.5, 0.5]))
            self.assertLessEqual(total_used, 0.8 + 1e-10)

    def test_unequal_ratios(self):
        combo, usage, std = _find_optimal_concurrency([0.1, 0.3, 0.2], 1.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 3)

    def test_very_small_resource(self):
        combo, usage, std = _find_optimal_concurrency([0.9, 0.9], 0.5)
        if combo is not None:
            total_used = sum(c * r for c, r in zip(combo, [0.9, 0.9]))
            self.assertLessEqual(total_used, 0.5 + 1e-10)


class TestCalculateRayNp(unittest.TestCase):

    def _make_mock_op(self, name='test_op', num_cpus=1, num_gpus=0, memory=0,
                      use_cuda_val=False, auto_proc=True, ray_actor=False, num_proc=-1):
        op = MagicMock()
        op._name = name
        op.num_cpus = num_cpus
        op.num_gpus = num_gpus
        op.memory = memory
        op.use_cuda.return_value = use_cuda_val
        op.use_auto_proc.return_value = auto_proc
        op.use_ray_actor.return_value = ray_actor
        op.num_proc = num_proc
        return op

    def _patch_ray_resources(self, cpu=16, gpu=0, mem=None, gpu_mem=None, cuda=False):
        """Helper to patch ray resource functions at their source modules."""
        if mem is None:
            mem = [32768]
        if gpu_mem is None:
            gpu_mem = []
        patches = [
            patch('data_juicer.utils.ray_utils.ray_cpu_count', return_value=cpu),
            patch('data_juicer.utils.ray_utils.ray_gpu_count', return_value=gpu),
            patch('data_juicer.utils.ray_utils.ray_available_memories', return_value=mem),
            patch('data_juicer.utils.ray_utils.ray_available_gpu_memories', return_value=gpu_mem),
            patch('data_juicer.utils.resource_utils.is_cuda_available', return_value=cuda),
        ]
        return patches

    def _run_with_patches(self, operators, cpu=16, gpu=0, mem=None, gpu_mem=None, cuda=False):
        from data_juicer.utils.process_utils import calculate_ray_np
        if mem is None:
            mem = [32768]
        if gpu_mem is None:
            gpu_mem = []

        mock_ray_utils = MagicMock()
        mock_ray_utils.ray_cpu_count.return_value = cpu
        mock_ray_utils.ray_gpu_count.return_value = gpu
        mock_ray_utils.ray_available_memories.return_value = mem
        mock_ray_utils.ray_available_gpu_memories.return_value = gpu_mem

        mock_resource_utils = MagicMock()
        mock_resource_utils.is_cuda_available.return_value = cuda

        with patch.dict('sys.modules', {
            'data_juicer.utils.ray_utils': mock_ray_utils,
        }):
            with patch('data_juicer.utils.resource_utils.is_cuda_available', return_value=cuda):
                return calculate_ray_np(operators)

    def test_cpu_only_operators(self):
        op = self._make_mock_op(num_cpus=2, memory=4)
        result = self._run_with_patches([op], cpu=16, mem=[32768])
        self.assertEqual(len(result), 1)

    def test_gpu_operator_with_memory(self):
        op = self._make_mock_op(name='gpu_op', num_cpus=1, num_gpus=1, memory=2,
                                use_cuda_val=True, ray_actor=True)
        result = self._run_with_patches([op], cpu=16, gpu=4, mem=[32768],
                                        gpu_mem=[8192, 8192, 8192, 8192], cuda=True)
        self.assertEqual(len(result), 1)

    def test_gpu_op_no_memory_no_gpus_warns(self):
        op = self._make_mock_op(name='gpu_op_bare', num_cpus=0, num_gpus=0, memory=0,
                                use_cuda_val=True, ray_actor=True)
        result = self._run_with_patches([op], cpu=16, gpu=4, mem=[32768],
                                        gpu_mem=[8192, 8192, 8192, 8192], cuda=True)
        self.assertEqual(len(result), 1)

    def test_gpu_request_without_cuda_raises(self):
        from data_juicer.utils.process_utils import calculate_ray_np
        op = self._make_mock_op(name='bad_op', num_gpus=1, use_cuda_val=True)
        with self.assertRaises(ValueError):
            self._run_with_patches([op], cpu=16, gpu=0, mem=[32768], cuda=False)

    def test_zero_cpu_raises(self):
        op = self._make_mock_op()
        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_patches([op], cpu=0, mem=[32768])
        self.assertIn("no CPU", str(ctx.exception))

    def test_zero_memory_raises(self):
        op = self._make_mock_op()
        with self.assertRaises(RuntimeError) as ctx:
            self._run_with_patches([op], cpu=16, mem=[0])
        self.assertIn("no memory", str(ctx.exception))

    def test_fixed_num_proc_operator(self):
        op = self._make_mock_op(num_cpus=2, auto_proc=False, num_proc=4)
        result = self._run_with_patches([op], cpu=16, mem=[32768])
        self.assertEqual(len(result), 1)

    def test_task_op_none_concurrency(self):
        op = self._make_mock_op(num_cpus=0, memory=0, auto_proc=False, num_proc=None)
        result = self._run_with_patches([op], cpu=16, mem=[32768])
        self.assertEqual(len(result), 1)

    def test_mixed_auto_and_fixed(self):
        op1 = self._make_mock_op(name='op1', num_cpus=2, auto_proc=True)
        op2 = self._make_mock_op(name='op2', num_cpus=4, auto_proc=False, num_proc=2)
        result = self._run_with_patches([op1, op2], cpu=16, mem=[32768])
        self.assertEqual(len(result), 2)

    def test_actor_fixed_num_proc_tuple(self):
        op = self._make_mock_op(name='actor_op', num_cpus=1, num_gpus=1, memory=2,
                                use_cuda_val=True, ray_actor=True, auto_proc=False, num_proc=[1, 4])
        result = self._run_with_patches([op], cpu=16, gpu=4, mem=[32768],
                                        gpu_mem=[8192, 8192, 8192, 8192], cuda=True)
        self.assertEqual(len(result), 1)

    def test_actor_resource_insufficient_raises(self):
        op1 = self._make_mock_op(name='big_op1', num_cpus=2, num_gpus=2, memory=4,
                                 use_cuda_val=True, ray_actor=True)
        op2 = self._make_mock_op(name='big_op2', num_cpus=2, num_gpus=2, memory=4,
                                 use_cuda_val=True, ray_actor=True)
        with self.assertRaises(ValueError):
            self._run_with_patches([op1, op2], cpu=4, gpu=2, mem=[8192],
                                   gpu_mem=[4096, 4096], cuda=True)

    def test_cpu_op_no_spec_no_auto_proc(self):
        op = self._make_mock_op(num_cpus=0, memory=0, auto_proc=True)
        result = self._run_with_patches([op], cpu=16, mem=[32768])
        self.assertEqual(len(result), 1)

    def test_op_num_proc_set_to_none_for_task(self):
        op = self._make_mock_op(num_cpus=1, auto_proc=False, num_proc=-1, ray_actor=False)
        self._run_with_patches([op], cpu=16, mem=[32768])
        self.assertIsNone(op.num_proc)


if __name__ == '__main__':
    unittest.main()
