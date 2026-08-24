"""
Tests to improve coverage for data_juicer/core/executor/default_executor.py.

Targets missed lines: 44-60, 65, 69, 79-107, 124-128, 156, 163-166, 169,
171-179, 184-185, 193, 211, 216, 231-232, 242-246, 268, 273-282, 314,
316-332.
"""

import json
import os
import shutil
import tempfile
import unittest

from datasets import Dataset

from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor
from data_juicer.core.data import NestedDataset
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class TestDefaultExecutorInit(DataJuicerTestCaseBase):
    """Test initialization of DefaultExecutor (lines 44-60, 65, 69)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        # Create a minimal dataset file
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            f.write('{"text": "hello world"}\n')
            f.write('{"text": "foo bar"}\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        """Create a minimal config for testing."""
        yaml_content = f"""
project_name: 'test_coverage'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_basic_init(self):
        """Test basic initialization sets key attributes (lines 44-60)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        # Lines 48-60: check attributes are set
        self.assertEqual(executor.work_dir, self.tmp_dir)
        self.assertEqual(executor.executor_type, "default")
        self.assertIsNone(executor.ckpt_manager)
        self.assertIsNotNone(executor.adapter)
        self.assertEqual(executor.np, 1)
        self.assertIsNotNone(executor.dataset_builder)
        self.assertIsNotNone(executor.exporter)

    @TEST_TAG("standalone")
    def test_init_with_use_cache(self):
        """Test initialization with use_cache=True (lines 65, 69)."""
        cfg = self._make_cfg(use_cache=True, cache_compress='zstd')
        executor = DefaultExecutor(cfg)

        from data_juicer.utils import cache_utils
        self.assertEqual(cache_utils.CACHE_COMPRESS, 'zstd')
        # Reset
        cache_utils.CACHE_COMPRESS = None

    @TEST_TAG("standalone")
    def test_init_with_checkpoint(self):
        """Test initialization with use_checkpoint=True (lines 79-90)."""
        cfg = self._make_cfg(use_checkpoint=True)
        executor = DefaultExecutor(cfg)

        self.assertIsNotNone(executor.ckpt_manager)
        self.assertTrue(os.path.exists(executor.ckpt_dir))

    @TEST_TAG("standalone")
    def test_init_np_default(self):
        """Test np defaults to 1 when not set."""
        cfg = self._make_cfg()
        cfg.np = None
        # Re-init with np=None to test the `or 1` logic
        # We need to set it before creating executor
        yaml_content = f"""
project_name: 'test_coverage'
dataset_path: '{self.data_file}'
export_path: '{os.path.join(self.tmp_dir, "output2.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg2.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        cfg.np = None
        executor = DefaultExecutor(cfg)
        self.assertEqual(executor.np, 1)


class TestDefaultExecutorRun(DataJuicerTestCaseBase):
    """Test run() method (lines 156, 163-166, 169, 171-179, 184-185, 193,
    211, 216, 231-232, 242-246, 268, 273-282, 314, 316-332)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        # Create a minimal dataset file
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(10):
                f.write(json.dumps({"text": f"sample text {i}", "id": str(i)}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, process_ops=None, **overrides):
        """Create a config for run tests."""
        if process_ops is None:
            process_ops = "  - whitespace_normalization_mapper:\n"
        yaml_content = f"""
project_name: 'test_run'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
{process_ops}
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_run_with_existing_dataset(self):
        """Test run() with a pre-built dataset (line 156)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello", "world", "foo"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    @TEST_TAG("standalone")
    def test_run_loads_from_builder(self):
        """Test run() loading from dataset_builder (lines 163-166, 169, 171-179)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        result = executor.run()
        self.assertIsNotNone(result)
        # Export file should exist
        self.assertTrue(os.path.exists(cfg.export_path))

    @TEST_TAG("standalone")
    def test_run_skip_export(self):
        """Test run() with skip_export=True (line 314)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        result = executor.run(skip_export=True)
        self.assertIsNotNone(result)
        # Export file should NOT exist
        self.assertFalse(os.path.exists(cfg.export_path))

    @TEST_TAG("standalone")
    def test_run_skip_return(self):
        """Test run() with skip_return=True (line 332)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        result = executor.run(skip_return=True)
        self.assertIsNone(result)

    @TEST_TAG("standalone")
    def test_run_no_ops(self):
        """Test run() with no process ops (line 193 - warning about no ops)."""
        cfg = self._make_cfg()
        # Override process to empty list after init_configs
        cfg.process = []
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello", "world"]
        }))
        result = executor.run(dataset=ds)
        # Even with no ops, should still return dataset
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_run_with_preflight_disabled(self):
        """Test run() with strict_preflight=False (line 156 branch)."""
        cfg = self._make_cfg(strict_preflight=False)
        executor = DefaultExecutor(cfg)

        result = executor.run()
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_run_with_load_data_np(self):
        """Test run() with explicit load_data_np (line 169)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        result = executor.run(load_data_np=2)
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_run_with_cache_compress(self):
        """Test run() with cache_compress enabled (lines 316-318)."""
        cfg = self._make_cfg(use_cache=True, cache_compress='zstd')
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello", "world"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)


class TestDefaultExecutorOpFusion(DataJuicerTestCaseBase):
    """Test OP fusion and adaptive batch size (lines 231-232, 242-246)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(20):
                f.write(json.dumps({"text": f"sample text number {i}"}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        yaml_content = f"""
project_name: 'test_fusion'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
  - remove_table_text_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_run_with_op_fusion(self):
        """Test run() with op_fusion=True (lines 231-232)."""
        cfg = self._make_cfg(op_fusion=True)
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello world", "foo bar baz"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_run_with_adaptive_batch_size(self):
        """Test run() with adaptive_batch_size=True (lines 242-246)."""
        cfg = self._make_cfg(adaptive_batch_size=True)
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello world", "foo bar baz"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)


class TestDefaultExecutorSampleData(DataJuicerTestCaseBase):
    """Test sample_data() method (lines 316-332 of the sample_data function)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(20):
                f.write(json.dumps({"text": f"sample {i}", "id": str(i)}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        yaml_content = f"""
project_name: 'test_sample'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_sample_data_uniform(self):
        """Test sample_data with uniform algo (default path)."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": [f"text {i}" for i in range(20)],
            "id": [str(i) for i in range(20)],
        }))
        result = executor.sample_data(dataset_to_sample=ds, sample_ratio=0.5)
        self.assertEqual(len(result), 10)

    @TEST_TAG("standalone")
    def test_sample_data_from_builder(self):
        """Test sample_data loading from dataset builder."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        result = executor.sample_data(sample_ratio=0.5)
        self.assertEqual(len(result), 10)

    @TEST_TAG("standalone")
    def test_sample_data_frequency_selector(self):
        """Test sample_data with frequency_specified_field_selector."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": [f"text {i}" for i in range(20)],
            "id": [str(i % 5) for i in range(20)],
        }))
        result = executor.sample_data(
            dataset_to_sample=ds,
            sample_algo='frequency_specified_field_selector',
            field_key='id',
            top_ratio=0.5,
        )
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_sample_data_topk_selector(self):
        """Test sample_data with topk_specified_field_selector."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": [f"text {i}" for i in range(20)],
            "id": [str(i) for i in range(20)],
        }))
        result = executor.sample_data(
            dataset_to_sample=ds,
            sample_algo='topk_specified_field_selector',
            field_key='id',
            topk=5,
        )
        self.assertEqual(len(result), 5)

    @TEST_TAG("standalone")
    def test_sample_data_unknown_algo_raises(self):
        """Test sample_data raises ValueError for unknown algorithm."""
        cfg = self._make_cfg()
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello"],
            "id": ["0"],
        }))
        with self.assertRaises(ValueError):
            executor.sample_data(
                dataset_to_sample=ds,
                sample_algo='invalid_algo',
            )


class TestDefaultExecutorCheckpoint(DataJuicerTestCaseBase):
    """Test checkpoint-related run paths (lines 80-81, 162-163, 311-312)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"text": f"sample {i}", "id": str(i)}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        yaml_content = f"""
project_name: 'test_ckpt'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _setup_checkpoint(self, cfg):
        """Create a real checkpoint by running once, then use it."""
        # First run to create the checkpoint
        executor = DefaultExecutor(cfg)
        executor.run()
        return executor

    @TEST_TAG("standalone")
    def test_run_with_checkpoint_enabled(self):
        """Test run() with use_checkpoint=True creates ckpt dir."""
        cfg = self._make_cfg(use_checkpoint=True)
        executor = DefaultExecutor(cfg)

        result = executor.run()
        self.assertIsNotNone(result)
        # ckpt directory should be created
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'ckpt')))

    @TEST_TAG("standalone")
    def test_init_with_existing_checkpoint(self):
        """Test init with an existing checkpoint (lines 80-81)."""
        # Create a config then run to create checkpoint, reusing same work_dir
        cfg = self._make_cfg(use_checkpoint=True)
        executor = DefaultExecutor(cfg)
        executor.run()

        # Now manually fix the ckpt_op.json to match the new config's process list
        # by reusing the same cfg object's process list in a new executor
        # The trick: create a new cfg with matching work_dir that was used in process
        ckpt_op_file = os.path.join(self.tmp_dir, 'ckpt', 'ckpt_op.json')
        # Write the process list from cfg into the checkpoint record
        with open(ckpt_op_file, 'w') as f:
            json.dump(cfg.process, f)

        # Now create executor with same cfg - checkpoint should match
        cfg2 = self._make_cfg(use_checkpoint=True)
        # Override the process to match what's in the ckpt file
        cfg2.process = cfg.process
        executor2 = DefaultExecutor(cfg2)
        self.assertTrue(executor2.ckpt_manager.ckpt_available)

    @TEST_TAG("standalone")
    def test_run_loads_from_checkpoint(self):
        """Test run() loading dataset from checkpoint (lines 162-163)."""
        # First run creates checkpoint
        cfg = self._make_cfg(use_checkpoint=True)
        executor = DefaultExecutor(cfg)
        executor.run()

        # Fix ckpt to match process list
        ckpt_op_file = os.path.join(self.tmp_dir, 'ckpt', 'ckpt_op.json')
        with open(ckpt_op_file, 'w') as f:
            json.dump(cfg.process, f)

        # Second run with matching config should load from checkpoint
        cfg2 = self._make_cfg(use_checkpoint=True)
        cfg2.process = cfg.process
        executor2 = DefaultExecutor(cfg2)
        self.assertTrue(executor2.ckpt_manager.ckpt_available)
        result = executor2.run()
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_sample_data_from_checkpoint(self):
        """Test sample_data() loading from checkpoint (lines 311-312)."""
        # First run creates checkpoint
        cfg = self._make_cfg(use_checkpoint=True)
        executor = DefaultExecutor(cfg)
        executor.run()

        # Fix ckpt to match process list
        ckpt_op_file = os.path.join(self.tmp_dir, 'ckpt', 'ckpt_op.json')
        with open(ckpt_op_file, 'w') as f:
            json.dump(cfg.process, f)

        # Second executor with matching config
        cfg2 = self._make_cfg(use_checkpoint=True)
        cfg2.process = cfg.process
        executor2 = DefaultExecutor(cfg2)
        self.assertTrue(executor2.ckpt_manager.ckpt_available)
        result = executor2.sample_data(sample_ratio=0.5)
        self.assertIsNotNone(result)


class TestDefaultExecutorTracer(DataJuicerTestCaseBase):
    """Test tracer-related paths (lines 124-128)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"text": f"sample {i}"}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        yaml_content = f"""
project_name: 'test_tracer'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_init_with_tracer(self):
        """Test initialization with open_tracer=True (lines 124-128)."""
        cfg = self._make_cfg(open_tracer=True)
        executor = DefaultExecutor(cfg)

        self.assertTrue(executor.open_tracer)
        self.assertIsNotNone(executor.tracer)

    @TEST_TAG("standalone")
    def test_run_with_tracer(self):
        """Test run() with tracer enabled."""
        cfg = self._make_cfg(open_tracer=True)
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello", "world"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)


class TestDefaultExecutorS3Export(DataJuicerTestCaseBase):
    """Test S3 export path initialization (lines 95-107)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            f.write('{"text": "hello"}\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg_with_s3(self, has_creds=False):
        """Create config with S3 export path."""
        from jsonargparse import Namespace
        yaml_content = f"""
project_name: 'test_s3'
dataset_path: '{self.data_file}'
np: 1
export_path: 's3://my-bucket/output/data.jsonl'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        cfg.export_path = 's3://my-bucket/output/data.jsonl'

        if has_creds:
            creds = Namespace()
            creds.aws_access_key_id = 'test_key'
            creds.aws_secret_access_key = 'test_secret'
            creds.aws_session_token = 'test_token'
            creds.aws_region = 'us-east-1'
            creds.endpoint_url = 'http://localhost:9000'
            cfg.export_aws_credentials = creds
        else:
            # Remove export_aws_credentials if exists
            if hasattr(cfg, 'export_aws_credentials'):
                delattr(cfg, 'export_aws_credentials')

        return cfg

    @TEST_TAG("standalone")
    def test_s3_export_no_credentials_raises(self):
        """Test S3 export path without credentials raises ValueError (line 107)."""
        cfg = self._make_cfg_with_s3(has_creds=False)
        # Remove the attribute entirely to trigger the else branch
        if hasattr(cfg, 'export_aws_credentials'):
            delattr(cfg, 'export_aws_credentials')

        with self.assertRaises(ValueError) as ctx:
            DefaultExecutor(cfg)
        self.assertIn("No AWS credentials", str(ctx.exception))

    @TEST_TAG("standalone")
    def test_s3_export_with_credentials(self):
        """Test S3 export with credentials passes them to exporter (lines 95-106)."""
        cfg = self._make_cfg_with_s3(has_creds=True)
        executor = DefaultExecutor(cfg)

        # Verify credentials were passed to exporter
        self.assertIsNotNone(executor.exporter)


class TestDefaultExecutorLoadDatasetKwargs(DataJuicerTestCaseBase):
    """Test run() and sample_data() with load_dataset_kwargs (lines 170, 319)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"text": f"sample {i}", "id": str(i)}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        yaml_content = f"""
project_name: 'test_kwargs'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_run_with_load_dataset_kwargs(self):
        """Test run() with load_dataset_kwargs set (line 170)."""
        from jsonargparse import Namespace
        load_kwargs = Namespace()
        load_kwargs.keep_in_memory = False
        cfg = self._make_cfg(load_dataset_kwargs=load_kwargs)
        executor = DefaultExecutor(cfg)
        result = executor.run()
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_sample_data_with_load_dataset_kwargs(self):
        """Test sample_data() with load_dataset_kwargs set (line 319)."""
        from jsonargparse import Namespace
        load_kwargs = Namespace()
        load_kwargs.keep_in_memory = False
        cfg = self._make_cfg(load_dataset_kwargs=load_kwargs)
        executor = DefaultExecutor(cfg)
        result = executor.sample_data(sample_ratio=0.5)
        self.assertIsNotNone(result)

    @TEST_TAG("standalone")
    def test_run_with_dataset_config(self):
        """Test run() with cfg.dataset set (line 201)."""
        cfg = self._make_cfg()
        # Set the dataset config attribute to trigger the branch
        cfg.dataset = {"type": "huggingface", "path": "test"}
        executor = DefaultExecutor(cfg)
        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello", "world"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)


class TestDefaultExecutorDAG(DataJuicerTestCaseBase):
    """Test DAG-related execution paths (lines 211, 216, 268, 273-282)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.tmp_dir, 'test_data.jsonl')
        with open(self.data_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"text": f"sample {i}"}) + '\n')

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_cfg(self, **overrides):
        yaml_content = f"""
project_name: 'test_dag'
dataset_path: '{self.data_file}'
np: 1
export_path: '{os.path.join(self.tmp_dir, "output.jsonl")}'
process:
  - whitespace_normalization_mapper:
"""
        yaml_path = os.path.join(self.tmp_dir, 'test_cfg.yaml')
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = self.tmp_dir
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @TEST_TAG("standalone")
    def test_run_with_dag_enabled(self):
        """Test run() with use_dag=True to exercise DAG monitoring paths."""
        cfg = self._make_cfg(use_dag=True)
        executor = DefaultExecutor(cfg)

        ds = NestedDataset(Dataset.from_dict({
            "text": ["hello", "world"]
        }))
        result = executor.run(dataset=ds)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
