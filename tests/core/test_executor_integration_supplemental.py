import json
import os
import tempfile
import unittest

from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor, NestedDataset
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExecutorIntegrationSupplementalTest(DataJuicerTestCaseBase):
    """
    Lightweight integration tests that exercise the default executor
    with simple configs, hitting code in default_executor.py, exporter.py,
    adapter.py, formatter.py, base_op.py, config.py all at once.
    """

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.created_files = []

    def tearDown(self):
        super().tearDown()
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _write_jsonl(self, records, filename='input.jsonl'):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        self.created_files.append(path)
        return path

    def _write_yaml(self, content, filename='config.yaml'):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, 'w') as f:
            f.write(content)
        self.created_files.append(path)
        return path

    def _make_config(self, dataset_path, export_path, ops_yaml):
        yaml_content = f"""dataset_path: '{dataset_path}'
export_path: '{export_path}'
np: 1
open_tracer: false
use_checkpoint: false
op_fusion: false
process:
{ops_yaml}
"""
        return self._write_yaml(yaml_content)

    def test_basic_filter_pipeline(self):
        """Run executor with text_length_filter and verify output."""
        records = [
            {'text': 'Hello world, this is a test sentence.'},
            {'text': 'Short'},
            {'text': 'Another reasonably long sentence for testing purposes.'},
            {'text': 'OK'},
            {'text': 'This sentence is definitely long enough to pass the filter.'},
        ]
        dataset_path = self._write_jsonl(records)
        export_path = os.path.join(self.tmp_dir, 'output', 'result.jsonl')

        ops_yaml = """\
  - text_length_filter:
      min_len: 10
      max_len: 1000"""

        yaml_path = self._make_config(dataset_path, export_path, ops_yaml)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = os.path.join(self.tmp_dir, 'work')

        executor = DefaultExecutor(cfg)
        dataset = executor.run()

        # Output file should exist
        self.assertTrue(os.path.exists(export_path))

        # Only records with text length >= 10 should remain
        # "Short" (5 chars) and "OK" (2 chars) should be filtered out
        self.assertEqual(len(dataset), 3)

    def test_multiple_ops_pipeline(self):
        """Run executor with two filter ops in sequence."""
        records = [
            {'text': 'Hello world'},
            {'text': 'A'},
            {'text': 'This is a much longer sentence with many words in it for testing'},
            {'text': 'Word'},
            {'text': 'Another test sentence here with enough words'},
        ]
        dataset_path = self._write_jsonl(records)
        export_path = os.path.join(self.tmp_dir, 'output', 'result.jsonl')

        ops_yaml = """\
  - text_length_filter:
      min_len: 10
      max_len: 1000
  - words_num_filter:
      lang: en
      min_num: 5
      max_num: 10000"""

        yaml_path = self._make_config(dataset_path, export_path, ops_yaml)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = os.path.join(self.tmp_dir, 'work')

        executor = DefaultExecutor(cfg)
        dataset = executor.run()

        self.assertTrue(os.path.exists(export_path))
        # "A" and "Word" filtered by text_length_filter (< 10 chars)
        # "Hello world" has only 2 words, filtered by words_num_filter (< 5)
        # Remaining: 2 sentences
        self.assertEqual(len(dataset), 2)

    def test_empty_dataset_all_filtered(self):
        """All samples are filtered out, resulting in empty output."""
        records = [
            {'text': 'Hi'},
            {'text': 'No'},
            {'text': 'OK'},
        ]
        dataset_path = self._write_jsonl(records)
        export_path = os.path.join(self.tmp_dir, 'output', 'result.jsonl')

        ops_yaml = """\
  - text_length_filter:
      min_len: 100
      max_len: 1000"""

        yaml_path = self._make_config(dataset_path, export_path, ops_yaml)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = os.path.join(self.tmp_dir, 'work')

        executor = DefaultExecutor(cfg)
        dataset = executor.run()

        self.assertTrue(os.path.exists(export_path))
        self.assertEqual(len(dataset), 0)

    def test_export_json_suffix(self):
        """Export path with .json suffix works."""
        records = [
            {'text': 'This is a valid sentence for testing.'},
        ]
        dataset_path = self._write_jsonl(records)
        export_path = os.path.join(self.tmp_dir, 'output', 'result.json')

        ops_yaml = """\
  - text_length_filter:
      min_len: 5
      max_len: 1000"""

        yaml_path = self._make_config(dataset_path, export_path, ops_yaml)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = os.path.join(self.tmp_dir, 'work')

        executor = DefaultExecutor(cfg)
        dataset = executor.run()

        self.assertTrue(os.path.exists(export_path))
        self.assertEqual(len(dataset), 1)

    def test_export_parquet_suffix(self):
        """Export path with .parquet suffix works."""
        records = [
            {'text': 'This is a valid sentence for testing.'},
            {'text': 'Another valid sentence that is long enough.'},
        ]
        dataset_path = self._write_jsonl(records)
        export_path = os.path.join(self.tmp_dir, 'output', 'result.parquet')

        ops_yaml = """\
  - text_length_filter:
      min_len: 5
      max_len: 1000"""

        yaml_path = self._make_config(dataset_path, export_path, ops_yaml)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = os.path.join(self.tmp_dir, 'work')

        executor = DefaultExecutor(cfg)
        dataset = executor.run()

        self.assertTrue(os.path.exists(export_path))
        self.assertEqual(len(dataset), 2)

    def test_no_ops_passthrough(self):
        """When no process ops are specified, dataset passes through."""
        records = [
            {'text': 'Hello world'},
            {'text': 'Test data'},
        ]
        dataset_path = self._write_jsonl(records)
        export_path = os.path.join(self.tmp_dir, 'output', 'result.jsonl')

        yaml_content = f"""dataset_path: '{dataset_path}'
export_path: '{export_path}'
np: 1
open_tracer: false
use_checkpoint: false
op_fusion: false
process: []
"""
        yaml_path = self._write_yaml(yaml_content)
        cfg = init_configs(['--config', yaml_path])
        cfg.work_dir = os.path.join(self.tmp_dir, 'work')

        executor = DefaultExecutor(cfg)
        dataset = executor.run()

        self.assertTrue(os.path.exists(export_path))
        self.assertEqual(len(dataset), 2)


if __name__ == '__main__':
    unittest.main()
