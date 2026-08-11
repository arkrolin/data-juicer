"""
Supplemental tests for data_juicer/core/exporter.py covering edge cases
not present in test_exporter.py:
- Suffix detection from various paths
- Export path generation for sharded output
- The _router method
- _write_jsonl_utf8 with unicode content
- export_compute_stats behavior
"""
import json
import os
import shutil
import tempfile
import unittest

from datasets import Dataset

from data_juicer.core import Exporter
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestExporterSuffixDetection(DataJuicerTestCaseBase):
    """Test _get_suffix with various file paths."""

    def test_suffix_jsonl(self):
        e = Exporter(export_path='/tmp/out.jsonl')
        self.assertEqual(e.suffix, 'jsonl')

    def test_suffix_json(self):
        e = Exporter(export_path='/tmp/out.json')
        self.assertEqual(e.suffix, 'json')

    def test_suffix_parquet(self):
        e = Exporter(export_path='/tmp/data.parquet')
        self.assertEqual(e.suffix, 'parquet')

    def test_suffix_uppercase_normalized(self):
        e = Exporter(export_path='/tmp/data.JSONL')
        self.assertEqual(e.suffix, 'jsonl')

    def test_suffix_unsupported_raises(self):
        with self.assertRaises(NotImplementedError):
            Exporter(export_path='/tmp/out.csv')

    def test_export_type_overrides_suffix(self):
        e = Exporter(export_path='/tmp/out.csv', export_type='jsonl')
        self.assertEqual(e.suffix, 'jsonl')


class TestExporterRouter(DataJuicerTestCaseBase):
    """Test the _router static method."""

    def test_router_keys(self):
        router = Exporter._router()
        self.assertIn('jsonl', router)
        self.assertIn('json', router)
        self.assertIn('parquet', router)
        self.assertEqual(len(router), 3)

    def test_router_values_are_callable(self):
        router = Exporter._router()
        for key, func in router.items():
            self.assertTrue(callable(func))


class TestExporterExportShardedLocal(DataJuicerTestCaseBase):
    """Test that sharded export creates multiple files with correct naming."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='dj_exp_shard_')
        self.dataset = Dataset.from_list([
            {'text': f'text {i}', Fields.stats: {'s': i}, HashKeys.hash: f'h{i}'}
            for i in range(10)
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_sharded_export_creates_numbered_files(self):
        export_path = os.path.join(self.tmp_dir, 'output.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=1,  # very small -> multiple shards
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(self.dataset)

        # Should have created shard files
        files = os.listdir(self.tmp_dir)
        shard_files = [f for f in files if 'output-' in f and f.endswith('.jsonl')]
        self.assertGreater(len(shard_files), 0)
        # Verify naming pattern: output-XX-of-YY.jsonl
        for sf in shard_files:
            self.assertIn('-of-', sf)


class TestExporterUnicodeExport(DataJuicerTestCaseBase):
    """Test that _write_jsonl_utf8 preserves unicode characters."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='dj_exp_utf8_')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_utf8_preserved_in_jsonl(self):
        dataset = Dataset.from_list([
            {'text': '你好世界'},
            {'text': 'café résumé'},
            {'text': 'emoji: 🎉'},
        ])
        export_path = os.path.join(self.tmp_dir, 'utf8.jsonl')
        Exporter._write_jsonl_utf8(dataset, export_path)

        with open(export_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)
        row0 = json.loads(lines[0])
        self.assertEqual(row0['text'], '你好世界')
        row1 = json.loads(lines[1])
        self.assertEqual(row1['text'], 'café résumé')
        row2 = json.loads(lines[2])
        self.assertEqual(row2['text'], 'emoji: 🎉')
        # Ensure no ascii escape sequences
        self.assertNotIn('\\u', lines[0])


class TestExporterExportComputeStats(DataJuicerTestCaseBase):
    """Test export_compute_stats keeps stats in exported data."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='dj_exp_cs_')
        self.dataset = Dataset.from_list([
            {'text': 'a', Fields.stats: {'score': 1}, HashKeys.hash: 'h1'},
            {'text': 'b', Fields.stats: {'score': 2}, HashKeys.hash: 'h2'},
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_export_compute_stats_keeps_stats_columns(self):
        export_path = os.path.join(self.tmp_dir, 'cs_out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        # export_compute_stats temporarily enables keep_stats_in_res_ds
        exporter.export_compute_stats(self.dataset, export_path)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        row = json.loads(lines[0])
        # Stats should be present since keep_stats was forced on
        self.assertIn(Fields.stats, row)

    def test_export_compute_stats_restores_flag(self):
        export_path = os.path.join(self.tmp_dir, 'cs_restore.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        self.assertFalse(exporter.keep_stats_in_res_ds)
        exporter.export_compute_stats(self.dataset, export_path)
        # After call, the flag should be restored
        self.assertFalse(exporter.keep_stats_in_res_ds)


class TestExporterKeepFields(DataJuicerTestCaseBase):
    """Test keep_stats_in_res_ds and keep_hashes_in_res_ds behavior."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix='dj_exp_keep_')
        self.dataset = Dataset.from_list([
            {
                'text': 'hello',
                Fields.stats: {'s': 1},
                Fields.meta: {'m': 'v'},
                HashKeys.hash: 'abc',
            },
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_keep_stats_true_includes_stats(self):
        export_path = os.path.join(self.tmp_dir, 'keep_stats.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            num_proc=1,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(self.dataset)
        with open(export_path, 'r', encoding='utf-8') as f:
            row = json.loads(f.readline())
        self.assertIn(Fields.stats, row)
        self.assertNotIn(HashKeys.hash, row)

    def test_keep_hashes_true_includes_hashes(self):
        export_path = os.path.join(self.tmp_dir, 'keep_hashes.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            num_proc=1,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=True,
            export_stats=False,
        )
        exporter.export(self.dataset)
        with open(export_path, 'r', encoding='utf-8') as f:
            row = json.loads(f.readline())
        self.assertIn(HashKeys.hash, row)
        self.assertNotIn(Fields.stats, row)


class TestExporterMaxShardSizeStr(DataJuicerTestCaseBase):
    """Test max_shard_size_str formatting."""

    def test_zero_bytes(self):
        e = Exporter(export_path='/tmp/x.jsonl', export_shard_size=0)
        self.assertIn('Bytes', e.max_shard_size_str)

    def test_kib_range(self):
        e = Exporter(export_path='/tmp/x.jsonl', export_shard_size=5 * 1024)
        self.assertIn('KiB', e.max_shard_size_str)

    def test_mib_range(self):
        e = Exporter(export_path='/tmp/x.jsonl', export_shard_size=10 * 1024 * 1024)
        self.assertIn('MiB', e.max_shard_size_str)

    def test_gib_range(self):
        e = Exporter(export_path='/tmp/x.jsonl', export_shard_size=2 * 1024**3)
        self.assertIn('GiB', e.max_shard_size_str)


if __name__ == '__main__':
    unittest.main()
