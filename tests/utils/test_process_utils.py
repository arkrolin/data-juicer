import os
import multiprocess as mp
import unittest
from unittest.mock import patch, MagicMock

import torch
import ray

from data_juicer.utils.process_utils import setup_mp, get_min_cuda_memory, calculate_np, calculate_ray_np
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from data_juicer.utils.constant import RAY_JOB_ENV_VAR

class ProcessUtilsTest(DataJuicerTestCaseBase):

    def test_setup_mp(self):
        all_methods = mp.get_all_start_methods()
        setup_mp()
        self.assertIn(mp.get_start_method(), all_methods)

        setup_mp('spawn')
        self.assertEqual(mp.get_start_method(), 'spawn')

        setup_mp(['spawn', 'forkserver', 'fork'])
        self.assertEqual(mp.get_start_method(), 'spawn')

    def test_get_min_cuda_memory(self):
        if torch.cuda.is_available():
            self.assertIsInstance(get_min_cuda_memory(), int)
        else:
            with self.assertRaises(AssertionError):
                get_min_cuda_memory()


class CalculateNpTest(DataJuicerTestCaseBase):

    def setUp(self):
        self._patch_module = 'data_juicer.utils.process_utils'
        self._patch_ray_module = 'data_juicer.utils.ray_utils'
        self._ori_ray_job_env_value = os.environ.get(RAY_JOB_ENV_VAR, '0')
        super().setUp()
    
    def tearDown(self):
        os.environ[RAY_JOB_ENV_VAR] = self._ori_ray_job_env_value
        super().tearDown()

    def enable_ray_mode(self):
        os.environ[RAY_JOB_ENV_VAR] = '1'
        ray.init(address='auto', ignore_reinit_error=True)

    @TEST_TAG('ray')
    def test_cuda_memory_zero_and_num_proc_not_given(self):
        logger = MagicMock()
        with patch(f"{self._patch_ray_module}.get_ray_nodes_info") as mock_ray_nodes_info, \
            patch(f"{self._patch_module}.cuda_device_count") as mock_cuda_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_ray_nodes_info.return_value = {
                'node1_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                'node2_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                }
            mock_cuda_count.return_value = 2
            self.enable_ray_mode()
            result = calculate_np("test_op", memory=0, num_cpus=0, use_cuda=True)
            self.assertEqual(result, 2)
            logger.warning.assert_called_with(
                "The required cuda memory and gpu of Op[test_op] has not been specified. "
                "Please specify the memory field or num_gpus field in the config file. "
                "You can reference data_juicer/config/config_all.yaml.Set the auto "
                "`num_proc` to number of GPUs 2."
            )
    @TEST_TAG('ray')
    def test_cuda_auto_less_than_device_count(self):
        logger = MagicMock()
        with patch(f"{self._patch_ray_module}.get_ray_nodes_info") as mock_ray_nodes_info, \
            patch(f"{self._patch_module}.logger", logger):
            mock_ray_nodes_info.return_value = {
                'node1_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                'node2_id': {'free_memory': 512, 'cpu_count': 8, 'free_gpus_memory': [2 * 1024]},
                }
            self.enable_ray_mode()
            result = calculate_np("test_op", memory=3, num_cpus=0, use_cuda=True)
            self.assertEqual(result, 2)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 2 of Op[test_op] based on the required cuda memory: 3GB required gpu: 0 and required cpu: 0."
            )

    @TEST_TAG('ray')
    def test_cuda_num_proc_exceeds_auto(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_gpu_memories") as mock_avail_gpu, \
            patch(f"{self._patch_module}.cuda_device_count") as mock_cuda_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_gpu.return_value = [5 * 1024, 5 * 1024]  # 5GB per GPU
            mock_cuda_count.return_value = 2
            # auto_num_proc = (5//2) * 2 = 4
            self.enable_ray_mode()
            result = calculate_np("test_op", memory=2, num_cpus=0, use_cuda=True)
            self.assertEqual(result, 4)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 4 of Op[test_op] based on the required cuda memory: 2GB required gpu: 0 and required cpu: 0."
            )

    def test_cpu_default_num_proc(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_memories") as mock_avail_mem, \
            patch(f"{self._patch_module}.cpu_count") as mock_cpu_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_mem.return_value = [8 * 1024 + 1]  # 8GB, add eps
            mock_cpu_count.return_value = 4
            result = calculate_np("test_op", memory=2, num_cpus=0, use_cuda=False)
            # auto_proc = 8//2 =4
            self.assertEqual(result, 4)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 4 of Op[test_op] based on the required memory: 2GB and required cpu: 0."
            )

    def test_cpu_insufficient_memory(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_memories") as mock_avail_mem, \
            patch(f"{self._patch_module}.cpu_count") as mock_cpu_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_mem.return_value = [2 * 1024]  # 2GB
            mock_cpu_count.return_value = 8
            result = calculate_np("test_op", memory=3, num_cpus=2, use_cuda=False)
            # auto_proc = 0,  max(min(5,0),1) =1
            self.assertEqual(result, 1)
            logger.warning.assert_called_with(
                "The required CPU number: 2 "
                "and memory: 3GB might "
                "be more than the available CPU: 8 "
                "and memory: [2.0]GB."
                "This Op [test_op] might "
                "require more resource to run. "
                "Set the auto `num_proc` to available nodes number 1."
            )

    def test_cpu_num_proc_unset_and_mem_unlimited(self):
        logger = MagicMock()
        with patch(f"{self._patch_module}.available_memories") as mock_avail_mem, \
            patch(f"{self._patch_module}.cpu_count") as mock_cpu_count, \
            patch(f"{self._patch_module}.logger", logger):
            mock_avail_mem.return_value = [8 * 1024]
            mock_cpu_count.return_value = 4
            result = calculate_np("test_op", memory=0, num_cpus=0, use_cuda=False)
            # auto_proc = 8/(接近0) ≈无限大，取默认 num_proc=4
            self.assertEqual(result, 4)
            logger.info.assert_called_with(
                "Set the auto `num_proc` to 4 of Op[test_op] based on the required memory: 0GB and required cpu: 0."
            )


class CalculateRayNPTest(DataJuicerTestCaseBase):

    def setUp(self):

        def _use_auto_proc(num_proc, use_cuda, ray_execution_mode=None):
            if ray_execution_mode:
                use_actor = ray_execution_mode == "actor"
            else:
                use_actor = use_cuda
    
            if not use_actor:  # ray task
                return num_proc == -1
            else:
                return not num_proc or num_proc == -1
            
        def _use_ray_actor(use_cuda, ray_execution_mode=None):
            if ray_execution_mode:
                return ray_execution_mode == "actor"

            return use_cuda

        def create_mock_op(use_cuda, num_proc=-1, ray_execution_mode=None):
            op = MagicMock(
                num_cpus=None,
                memory=None,
                num_gpus=None,
                num_proc=num_proc,
                _name="test_op",
                use_cuda=lambda: use_cuda,
            )
            op.use_auto_proc = lambda: _use_auto_proc(op.num_proc, ray_execution_mode)
            op.use_ray_actor = lambda: _use_ray_actor(use_cuda, ray_execution_mode)
            return op

        self.mock_op = create_mock_op
        
        # Common patchers
        self.ray_cpu_patcher = patch(
            'data_juicer.utils.ray_utils.ray_cpu_count')
        self.ray_gpu_patcher = patch(
            'data_juicer.utils.ray_utils.ray_gpu_count')
        self.ray_mem_patcher = patch(
            'data_juicer.utils.ray_utils.ray_available_memories')
        self.ray_gpu_mem_patcher = patch(
            'data_juicer.utils.ray_utils.ray_available_gpu_memories')
        self.cuda_available_patcher = patch(
            'data_juicer.utils.resource_utils.is_cuda_available')
        
        self.mock_cpu = self.ray_cpu_patcher.start()
        self.mock_gpu = self.ray_gpu_patcher.start()
        self.mock_mem = self.ray_mem_patcher.start()
        self.mock_gpu_mem = self.ray_gpu_mem_patcher.start()
        self.mock_cuda_available = self.cuda_available_patcher.start()

        # Default cluster resources (64 CPUs, 256GB RAM, 8 GPU 32GB)
        self.mock_cpu.return_value = 64
        self.mock_gpu.return_value = 8
        self.mock_mem.return_value = [256 * 1024]  # 256GB
        self.mock_gpu_mem.return_value = [32 * 1024] * 8 # 32GB * 8
        self.mock_cuda_available.return_value = True

    def tearDown(self):
        self.ray_cpu_patcher.stop()
        self.ray_gpu_patcher.stop()
        self.ray_mem_patcher.stop()
        self.ray_gpu_mem_patcher.stop()
        self.cuda_available_patcher.stop()

    def test_cpu_op_auto_scaling(self):
        """Test CPU operator with auto scaling"""
        op = self.mock_op(use_cuda=False)
        op.num_cpus = 1
        
        calculate_ray_np([op])
        self.assertEqual(op.num_proc, None)
        self.assertEqual(op.num_cpus, 1)
        self.assertEqual(op.num_gpus, None)

    def test_gpu_op_auto_scaling(self):
        """Test GPU operator with auto scaling"""
        op = self.mock_op(use_cuda=True)
        op.num_gpus = 1
        
        calculate_ray_np([op])
        self.assertEqual(op.num_proc, 8)  # Only 1 op and 8 GPU available
        self.assertEqual(op.num_gpus, 1)
        self.assertEqual(op.num_cpus, None)

    def test_gpu_op_in_task_mode_preserves_requested_gpu(self):
        """A CUDA Ray task must retain its GPU scheduling requirement."""
        op = self.mock_op(
            use_cuda=True,
            num_proc=8,
            ray_execution_mode="task",
        )
        op.num_cpus = 1
        op.num_gpus = 1

        calculate_ray_np([op])

        self.assertEqual(op.num_proc, 8)
        self.assertEqual(op.num_cpus, 1)
        self.assertEqual(op.num_gpus, 1)

    def test_user_specified_num_proc(self):
        """Test user-specified num_proc takes priority"""
        op = self.mock_op(use_cuda=False, num_proc=2)
        op.num_cpus = 1
        
        calculate_ray_np([op])
        self.assertEqual(op.num_proc, 2)
        self.assertEqual(op.num_cpus, 1)
        self.assertEqual(op.num_gpus, None)
    
    def test_user_specified_num_proc_to_none_in_task(self):
        """Test user-specified num_proc takes priority"""
        op = self.mock_op(use_cuda=False, num_proc=None)
        op.num_cpus = 1
        
        calculate_ray_np([op])
        self.assertEqual(op.num_proc, None)
        self.assertEqual(op.num_cpus, 1)
        self.assertEqual(op.num_gpus, None)

    @unittest.skip("Disabled num_proc tuple check for ray task mode")
    def test_num_proc_check(self):
        op = self.mock_op(use_cuda=False, num_proc=(1, 2))
        op._name = 'op1'
        op.num_cpus = 1
        
        with self.assertRaises(ValueError) as cm:
            calculate_ray_np([op])

        self.assertEqual(str(cm.exception), 
                         "Op[op1] is running in ray task mode, ``num_proc`` is expected to be set as an integer but got: (1, 2).")

    def test_mixed_ops_resource_allocation(self):
        """Test mixed operators with fixed and auto scaling"""
        fixed_op = self.mock_op(use_cuda=False, num_proc=4)  # concurrency max=4, min=1
        fixed_op._name = 'op1'
        fixed_op.num_cpus = 1
        
        auto_op = self.mock_op(use_cuda=False)
        auto_op._name = 'op2'
        auto_op.num_cpus = 1
        
        calculate_ray_np([fixed_op, auto_op])

        self.assertEqual(fixed_op.num_cpus, 1)
        self.assertEqual(fixed_op.num_proc, 4)
        self.assertEqual(auto_op.num_proc, None)
        self.assertEqual(auto_op.num_cpus, 1)

    def test_insufficient_resources_cpu(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False, num_proc=5)
        op1._name = 'op1'
        op1.num_cpus = 2
        
        op2 = self.mock_op(use_cuda=False)
        op2._name = 'op2'
        op2.num_cpus = 3
        
        self.mock_cpu.return_value = 4  # Only 4 cores available
        
        calculate_ray_np([op1, op2])

        # removing resource restriction errors. Errors should not be reported at batch mode  and  resource flexibility
        # with self.assertRaises(ValueError) as cm:
        #     calculate_ray_np([op1, op2])

        # self.assertEqual(str(cm.exception),
        #                  "Insufficient cpu resources: At least 5.0 cpus are required,  but only 4 are available. "
        #                  "Please add resources to ray cluster or reduce operator requirements.")

        self.assertEqual(op1.num_proc, 5)
        self.assertEqual(op1.num_cpus, 2)
        self.assertEqual(op1.num_gpus, None)
        self.assertEqual(op1.memory, None)

        self.assertEqual(op2.num_proc, None)
        self.assertEqual(op2.num_cpus, 3)
        self.assertEqual(op2.num_gpus, None)
        self.assertEqual(op2.memory, None)

    def test_insufficient_resources_gpu(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False, num_proc=5)
        op1._name = 'op1'
        op1.num_cpus = 2
        
        op2 = self.mock_op(use_cuda=True)
        op2._name = 'op2'

        op3 = self.mock_op(use_cuda=True)
        op3._name = 'op3'

        self.mock_gpu.return_value = 1
        self.mock_cpu.return_value = 5

        with self.assertRaises(ValueError) as cm:
            calculate_ray_np([op1, op2, op3])

        self.assertEqual(str(cm.exception), 
                         "GPU resource is not enough for the current operators configuration. "
                         "At least 2.0 gpus are required, but only 1 gpus are available. "
                         "Please consider configuring the 'num_gpus' of cuda operators to "
                         "a smaller value or increase the number of GPUs.")
    
    def test_batch_mode(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False, num_proc=5)
        op1._name = 'op1'
        op1.num_cpus = 3
        
        op2 = self.mock_op(use_cuda=False)
        op2._name = 'op2'
        op2.num_cpus = 4

        op3 = self.mock_op(use_cuda=True)
        op3._name = 'op3'

        op4 = self.mock_op(use_cuda=True, num_proc=2)
        op4._name = 'op4'

        self.mock_gpu.return_value = 3
        self.mock_cpu.return_value = 5

        calculate_ray_np([op1, op2, op3, op4])

        self.assertEqual(op1.num_proc, 5)
        self.assertEqual(op1.num_cpus, 3)
        self.assertEqual(op1.num_gpus, None)
        self.assertEqual(op1.memory, None)

        self.assertEqual(op2.num_proc, None)
        self.assertEqual(op2.num_cpus, 4)
        self.assertEqual(op2.num_gpus, None)
        self.assertEqual(op2.memory, None)

        self.assertEqual(op3.num_proc, 1)
        self.assertEqual(op3.num_cpus, None)
        self.assertEqual(op3.num_gpus, 1)
        self.assertEqual(op3.memory, None)

        self.assertEqual(op4.num_proc, 2)
        self.assertEqual(op4.num_cpus, None)
        self.assertEqual(op4.num_gpus, 1)
        self.assertEqual(op4.memory, None)

    def test_gpu_op_without_cuda(self):
        """Test GPU operator when CUDA is unavailable"""
        self.mock_cuda_available.return_value = False
        op = self.mock_op(use_cuda=True)
        op.num_gpus = 1
        
        with self.assertRaises(ValueError) as cm:
            calculate_ray_np([op])

        self.assertEqual(str(cm.exception), 
                         "Op[test_op] attempted to request GPU resources (num_gpus=1), "
                         "but the gpu is unavailable. Please check whether your environment is installed correctly"
                         " and whether there is a gpu in the resource pool.")

    def test_multi_ops_with_cpu_gpu(self):
        """Test operator with no resource requirements"""

        op1_cuda = self.mock_op(use_cuda=True)
        op1_cuda.memory = 2
        op1_cuda.num_cpus = 1
        op1_cuda._name = 'op1_cuda'

        op2_cuda = self.mock_op(use_cuda=True)
        op2_cuda.num_gpus = 0.5
        op2_cuda._name = 'op2_cuda'

        op3_cuda = self.mock_op(use_cuda=True, num_proc=(5, 10))
        op3_cuda.num_gpus = 0.2
        op3_cuda._name = 'op3_cuda'

        op1_cpu = self.mock_op(use_cuda=False)
        op1_cpu.memory = 8
        op1_cpu._name = 'op1_cpu'

        op2_cpu = self.mock_op(use_cuda=False)
        op2_cpu.num_cpus = 5
        op2_cpu._name = 'op2_cpu'

        op3_cpu = self.mock_op(use_cuda=False, num_proc=10)  # concurrency max=10, min=1
        op3_cpu.num_cpus = 0.2
        op3_cpu._name = 'op3_cpu'

        op4_cpu = self.mock_op(use_cuda=False)
        op4_cpu._name = 'op4_cpu'

        self.mock_cpu.return_value = 100
        self.mock_gpu.return_value = 5
        self.mock_mem.return_value = [131072]  # 128 GB
        self.mock_gpu_mem.return_value = [10240] * 5  # 10GB * 5

        calculate_ray_np([op1_cuda, op2_cuda, op3_cuda, op1_cpu, op2_cpu, op3_cpu, op4_cpu])

        # fixed cpu: 
        #   op3_cpu: 0.2
        # fixed gpu: 
        #   op3_cuda: (1, 2) # (5*0.2, 10*0.2)

        # remaining gpu: (3, 4)

        # auto gpu: 0.2: 0.5  remaining min gpu = 3
        # find_optimal_concurrency([0.2, 0.5], 3) = [2, 5]

        self.assertEqual(op1_cuda.num_proc, (2, 20)) # min=2, max=4/(2/10)
        self.assertEqual(op1_cuda.num_cpus, 1)
        self.assertEqual(op1_cuda.num_gpus, 0.2)  # 2GB / 10GB * 1.0
        self.assertEqual(op1_cuda.memory, 2)

        self.assertEqual(op2_cuda.num_proc, (5, 8))  # min=4, max=4/0.5
        self.assertEqual(op2_cuda.num_cpus, None)
        self.assertEqual(op2_cuda.num_gpus, 0.5)
        self.assertEqual(op2_cuda.memory, None)

        # fixed gpu
        self.assertEqual(op3_cuda.num_proc, (5, 10))
        self.assertEqual(op3_cuda.num_cpus, None)
        self.assertEqual(op3_cuda.num_gpus, 0.2)
        self.assertEqual(op3_cuda.memory, None)

        self.assertEqual(op1_cpu.num_proc, None)
        self.assertEqual(op1_cpu.num_cpus, None)
        self.assertEqual(op1_cpu.num_gpus, None)
        self.assertEqual(op1_cpu.memory, 8) 

        self.assertEqual(op2_cpu.num_proc, None)
        self.assertEqual(op2_cpu.num_cpus, 5)
        self.assertEqual(op2_cpu.num_gpus, None)
        self.assertEqual(op2_cpu.memory, None)

        # fixed cpu
        self.assertEqual(op3_cpu.num_proc, 10)
        self.assertEqual(op3_cpu.num_cpus, 0.2)
        self.assertEqual(op3_cpu.num_gpus, None)
        self.assertEqual(op3_cpu.memory, None)

        self.assertEqual(op4_cpu.num_proc, None)
        self.assertEqual(op4_cpu.num_cpus, None)
        self.assertEqual(op4_cpu.num_gpus, None)
        self.assertEqual(op4_cpu.memory, None)

    def test_cpu_and_gpu_actors(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False, ray_execution_mode='actor')
        op1._name = 'op1'
        op1.num_cpus = 2
        
        op2 = self.mock_op(use_cuda=True, num_proc=2)
        op2._name = 'op2'

        op3 = self.mock_op(use_cuda=True)
        op3._name = 'op3'

        self.mock_gpu.return_value = 5
        self.mock_cpu.return_value = 20

        calculate_ray_np([op1, op2, op3])

        self.assertEqual(op1.num_proc, (7,9))
        self.assertEqual(op1.num_cpus, 2)
        self.assertEqual(op1.num_gpus, 0)
        self.assertEqual(op1.memory, None) 

        self.assertEqual(op2.num_proc, 2)
        self.assertEqual(op2.num_cpus, None)
        self.assertEqual(op2.num_gpus, 1)
        self.assertEqual(op2.memory, None)

        self.assertEqual(op3.num_proc, 3)
        self.assertEqual(op3.num_cpus, None)
        self.assertEqual(op3.num_gpus, 1)
        self.assertEqual(op3.memory, None)

    def test_cpu_and_gpu_actors2(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False, ray_execution_mode='actor')
        op1._name = 'op1'
        op1.num_cpus = 2
        
        op2 = self.mock_op(use_cuda=True, num_proc=(1, 2))
        op2._name = 'op2'

        op3 = self.mock_op(use_cuda=True)
        op3._name = 'op3'

        self.mock_gpu.return_value = 5
        self.mock_cpu.return_value = 20

        calculate_ray_np([op1, op2, op3])

        self.assertEqual(op1.num_proc, (7,9))
        self.assertEqual(op1.num_cpus, 2)
        self.assertEqual(op1.num_gpus, 0)
        self.assertEqual(op1.memory, None) 

        self.assertEqual(op2.num_proc, (1,2))
        self.assertEqual(op2.num_cpus, None)
        self.assertEqual(op2.num_gpus, 1)
        self.assertEqual(op2.memory, None)

        self.assertEqual(op3.num_proc, (3, 4))
        self.assertEqual(op3.num_cpus, None)
        self.assertEqual(op3.num_gpus, 1)
        self.assertEqual(op3.memory, None)

    def test_cpu_and_gpu_actors3(self):
        """Test resource overallocation exception"""
        op1 = self.mock_op(use_cuda=False, ray_execution_mode='actor')
        op1._name = 'op1'
        op1.num_cpus = 2
        
        op2 = self.mock_op(use_cuda=True)
        op2._name = 'op2'

        op3 = self.mock_op(use_cuda=True)
        op3._name = 'op3'

        self.mock_gpu.return_value = 5
        self.mock_cpu.return_value = 20

        calculate_ray_np([op1, op2, op3])

        self.assertEqual(op1.num_proc, (7, 10))
        self.assertEqual(op1.num_cpus, 2)
        self.assertEqual(op1.num_gpus, 0)
        self.assertEqual(op1.memory, None) 

        self.assertEqual(op2.num_proc, (2, 5))
        self.assertEqual(op2.num_cpus, None)
        self.assertEqual(op2.num_gpus, 1)
        self.assertEqual(op2.memory, None)

        self.assertEqual(op3.num_proc, (3, 5))
        self.assertEqual(op3.num_cpus, None)
        self.assertEqual(op3.num_gpus, 1)
        self.assertEqual(op3.memory, None)


class SetupWorkerThreadsTest(DataJuicerTestCaseBase):

    def setUp(self):
        import data_juicer.utils.process_utils as mod
        self._orig = mod._WORKER_THREADS_CONFIGURED
        mod._WORKER_THREADS_CONFIGURED = False

    def tearDown(self):
        import data_juicer.utils.process_utils as mod
        mod._WORKER_THREADS_CONFIGURED = self._orig

    @TEST_TAG("standalone")
    def test_sets_torch_threads(self):
        from data_juicer.utils.process_utils import setup_worker_threads
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            setup_worker_threads(num_threads=2)
        mock_torch.set_num_threads.assert_called_once_with(2)
        mock_torch.set_num_interop_threads.assert_called_once_with(2)

    @TEST_TAG("standalone")
    def test_runtime_error_torch(self):
        import data_juicer.utils.process_utils as mod
        from data_juicer.utils.process_utils import setup_worker_threads
        mod._WORKER_THREADS_CONFIGURED = False
        mock_torch = MagicMock()
        mock_torch.set_num_interop_threads.side_effect = RuntimeError("already set")
        with patch.dict('sys.modules', {'torch': mock_torch}):
            setup_worker_threads(num_threads=1)
        mock_torch.set_num_threads.assert_called_once_with(1)

    @TEST_TAG("standalone")
    def test_only_configures_once(self):
        import data_juicer.utils.process_utils as mod
        from data_juicer.utils.process_utils import setup_worker_threads
        mod._WORKER_THREADS_CONFIGURED = True
        mock_torch = MagicMock()
        with patch.dict('sys.modules', {'torch': mock_torch}):
            setup_worker_threads(num_threads=4)
        mock_torch.set_num_threads.assert_not_called()


class SetupMpMockTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.mp')
    def test_non_main_process_returns_early(self, mock_mp):
        from data_juicer.utils.process_utils import setup_mp
        mock_mp.current_process.return_value.name = "Worker-1"
        setup_mp()
        mock_mp.set_start_method.assert_not_called()

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.mp')
    def test_env_method_override(self, mock_mp):
        from data_juicer.utils.process_utils import setup_mp
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['fork', 'spawn', 'forkserver']
        with patch.dict('os.environ', {'MP_START_METHOD': 'spawn'}):
            setup_mp(method=['fork', 'spawn'])
        mock_mp.set_start_method.assert_called_once_with('spawn', force=True)

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.mp')
    def test_method_not_available(self, mock_mp):
        from data_juicer.utils.process_utils import setup_mp
        mock_mp.current_process.return_value.name = "MainProcess"
        mock_mp.get_all_start_methods.return_value = ['spawn']
        setup_mp(method='forkserver')
        mock_mp.set_start_method.assert_not_called()


class FindOptimalConcurrencyTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_empty_input(self):
        from data_juicer.utils.process_utils import _find_optimal_concurrency
        result = _find_optimal_concurrency([], 1.0)
        self.assertEqual(result, (None, 0, 0))

    @TEST_TAG("standalone")
    def test_all_zero_ratios(self):
        from data_juicer.utils.process_utils import _find_optimal_concurrency
        result = _find_optimal_concurrency([0, 0, 0], 1.0)
        self.assertEqual(result, (None, 0, 0))

    @TEST_TAG("standalone")
    def test_single_operator(self):
        from data_juicer.utils.process_utils import _find_optimal_concurrency
        combo, usage, std = _find_optimal_concurrency([0.25], 1.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 1)
        self.assertGreater(usage, 0)

    @TEST_TAG("standalone")
    def test_two_operators_equal_resource(self):
        from data_juicer.utils.process_utils import _find_optimal_concurrency
        combo, usage, std = _find_optimal_concurrency([0.2, 0.2], 1.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 2)
        self.assertLessEqual(usage, 1.0 + 1e-10)

    @TEST_TAG("standalone")
    def test_resource_constraint_respected(self):
        from data_juicer.utils.process_utils import _find_optimal_concurrency
        combo, usage, std = _find_optimal_concurrency([0.5, 0.5], 0.8)
        if combo is not None:
            total_used = sum(c * r for c, r in zip(combo, [0.5, 0.5]))
            self.assertLessEqual(total_used, 0.8 + 1e-10)

    @TEST_TAG("standalone")
    def test_unequal_ratios(self):
        from data_juicer.utils.process_utils import _find_optimal_concurrency
        combo, usage, std = _find_optimal_concurrency([0.1, 0.3, 0.2], 1.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 3)


class CalculateNpMockTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.cpu_count', return_value=16)
    @patch('data_juicer.utils.process_utils.available_memories', return_value=[16384, 16384])
    def test_cpu_with_memory(self, mock_mem, mock_cpu):
        result = calculate_np("test_op", memory=4, num_cpus=2, use_cuda=False, num_gpus=0)
        self.assertGreater(result, 0)

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.cpu_count', return_value=8)
    @patch('data_juicer.utils.process_utils.available_memories', return_value=[8192])
    def test_cpu_only_no_memory(self, mock_mem, mock_cpu):
        result = calculate_np("test_op", memory=0, num_cpus=2, use_cuda=False, num_gpus=0)
        self.assertEqual(result, 4)

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.cpu_count', return_value=4)
    @patch('data_juicer.utils.process_utils.available_memories', return_value=[1024])
    def test_cpu_insufficient_resources(self, mock_mem, mock_cpu):
        result = calculate_np("test_op", memory=8, num_cpus=8, use_cuda=False, num_gpus=0)
        self.assertEqual(result, 1)

    @TEST_TAG("standalone")
    def test_use_cuda_false_but_num_gpus_raises(self):
        with self.assertRaises(ValueError) as ctx:
            calculate_np("test_op", memory=0, num_cpus=0, use_cuda=False, num_gpus=2)
        self.assertIn("GPU resources", str(ctx.exception))

    @TEST_TAG("standalone")
    @patch('data_juicer.utils.process_utils.cpu_count', return_value=16)
    @patch('data_juicer.utils.process_utils.cuda_device_count', return_value=4)
    @patch('data_juicer.utils.process_utils.available_gpu_memories', return_value=[8192, 8192, 8192, 8192])
    def test_cuda_no_memory_no_gpus(self, mock_gpu_mem, mock_cuda_count, mock_cpu):
        result = calculate_np("test_op", memory=0, num_cpus=0, use_cuda=True, num_gpus=0)
        self.assertEqual(result, 4)


if __name__ == '__main__':
    unittest.main()
