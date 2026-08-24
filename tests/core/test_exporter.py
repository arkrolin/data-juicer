import json
import os
import shutil
import tempfile
import unittest
import jsonlines as jl
from unittest.mock import patch

import numpy as np
from datasets import Dataset
from cryptography.fernet import Fernet

from data_juicer.core import Exporter
from data_juicer.core import exporter as exporter_module
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.file_utils import add_suffix_to_filename

class ExporterTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.work_dir = 'tmp/test_exporter/'
        os.makedirs(self.work_dir, exist_ok=True)

        self.test_data = Dataset.from_list([
            {
                'text': 'text 1',
                Fields.stats: {
                    'a': 1,
                    'b': 2
                },
                Fields.meta: {
                    'c': 'tag1'
                },
                HashKeys.hash: 'hash1'
            },
            {
                'text': 'text 2',
                Fields.stats: {
                    'a': 3,
                    'b': 4
                },
                Fields.meta: {
                    'c': 'tag2'
                },
                HashKeys.hash: 'hash2'
            },
            {
                'text': 'text 3',
                Fields.stats: {
                    'a': 5,
                    'b': 6
                },
                Fields.meta: {
                    'c': 'tag3'
                },
                HashKeys.hash: 'hash3'
            },
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def test_normal_function(self):
        export_path = os.path.join(self.work_dir, 'normal', 'test.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=True,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.test_data)

        # check exported files
        self.assertTrue(os.path.exists(export_path))
        self.assertTrue(os.path.exists(add_suffix_to_filename(export_path, '_stats')))

    def test_different_shard_size(self):
        export_path = os.path.join(self.work_dir, 'shard_size', 'test.json')
        # bytes
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
        )
        self.assertIn('Bytes', exporter.max_shard_size_str)

        # KiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 10,
        )
        self.assertIn('KiB', exporter.max_shard_size_str)

        # MiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 20,
        )
        self.assertIn('MiB', exporter.max_shard_size_str)

        # GiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 30,
        )
        self.assertIn('GiB', exporter.max_shard_size_str)

        # TiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 40,
        )
        self.assertIn('TiB', exporter.max_shard_size_str)

        # more --> TiB
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=2 * 2 ** 50,
        )
        self.assertIn('TiB', exporter.max_shard_size_str)

    def test_supported_suffix(self):
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'json', 'test.json'),
        )
        self.assertEqual('json', exporter.suffix)
        exporter.export(self.test_data)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'json', 'test.json')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'json', 'test_stats.jsonl')))

        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'jsonl', 'test.jsonl'),
        )
        self.assertEqual('jsonl', exporter.suffix)
        exporter.export(self.test_data)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'jsonl', 'test.jsonl')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'jsonl', 'test_stats.jsonl')))

        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'parquet', 'test.parquet'),
        )
        self.assertEqual('parquet', exporter.suffix)
        exporter.export(self.test_data)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'parquet', 'test.parquet')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'parquet', 'test_stats.jsonl')))

        with self.assertRaises(NotImplementedError):
            Exporter(
                export_path=os.path.join(self.work_dir, 'txt', 'test.txt'),
            )

    def test_export_multiple_shards(self):
        export_path = os.path.join(self.work_dir, 'shards', 'test.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=1024,
            export_in_parallel=True,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.test_data)

        # check exported files
        self.assertTrue(os.path.exists(add_suffix_to_filename(export_path, '-00-of-01')))
        self.assertTrue(os.path.exists(add_suffix_to_filename(export_path, '_stats')))

    def test_export_compute_stats(self):
        export_path = os.path.join(self.work_dir, 'stats', 'res.jsonl')
        exporter = Exporter(
            export_path=export_path,
        )
        exporter.export_compute_stats(self.test_data, export_path)

        self.assertTrue(os.path.exists(export_path))
        self.assertFalse(os.path.exists(add_suffix_to_filename(export_path, '_stats')))


class ExporterEncryptTest(DataJuicerTestCaseBase):
    """Tests for the encrypt_before_export feature of Exporter."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_encrypt/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.key = Fernet.generate_key()
        self.fernet = Fernet(self.key)
        self.test_data = Dataset.from_list([
            {'text': 'hello', Fields.stats: {'score': 1}},
            {'text': 'world', Fields.stats: {'score': 2}},
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')

    def _write_key_file(self):
        key_file = os.path.join(self.work_dir, 'test.key')
        with open(key_file, 'wb') as f:
            f.write(self.key)
        return key_file

    # ------------------------------------------------------------------
    # __init__ parameter handling
    # ------------------------------------------------------------------

    def test_encrypt_flag_disabled_by_default(self):
        os.makedirs(os.path.join(self.work_dir, 'default'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'default', 'out.jsonl'),
        )
        self.assertFalse(exporter.encrypt_before_export)
        self.assertIsNone(exporter._fernet)

    def test_encrypt_flag_enabled_with_key_file(self):
        key_file = self._write_key_file()
        os.makedirs(os.path.join(self.work_dir, 'enabled'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'enabled', 'out.jsonl'),
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        self.assertTrue(exporter.encrypt_before_export)
        self.assertIsNotNone(exporter._fernet)

    def test_s3_path_disables_encryption_with_warning(self):
        """S3 export_path should disable local-file encryption with a warning."""
        from loguru import logger

        key_file = self._write_key_file()
        warning_messages = []
        handler_id = logger.add(
            lambda msg: warning_messages.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = Exporter(
                export_path='s3://bucket/prefix/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=key_file,
            )
        finally:
            logger.remove(handler_id)

        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            len(warning_messages) > 0,
            'Expected a loguru WARNING about S3 path skipping encryption',
        )

    @patch("fsspec.implementations.arrow.ArrowFSWrapper")
    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_hdfs_path_disables_encryption_with_warning(self, mock_create_hdfs_fs, mock_arrow_wrapper):
        """HDFS export_path should disable local-file encryption with a warning."""
        from loguru import logger

        key_file = self._write_key_file()
        warning_messages = []
        handler_id = logger.add(
            lambda msg: warning_messages.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = Exporter(
                export_path='hdfs://namenode:8020/user/data/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=key_file,
            )
        finally:
            logger.remove(handler_id)

        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            len(warning_messages) > 0,
            'Expected a loguru WARNING about HDFS path skipping encryption',
        )


    # ------------------------------------------------------------------
    # _encrypt_local_file helper
    # ------------------------------------------------------------------

    def test_encrypt_local_file_encrypts_in_place(self):
        key_file = self._write_key_file()
        os.makedirs(os.path.join(self.work_dir, 'inplace'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'inplace', 'out.jsonl'),
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        plain_path = os.path.join(self.work_dir, 'plain.txt')
        plaintext = b'plaintext content'
        with open(plain_path, 'wb') as f:
            f.write(plaintext)

        exporter._encrypt_local_file(plain_path)

        with open(plain_path, 'rb') as f:
            content = f.read()
        # File must have been overwritten with ciphertext
        self.assertNotEqual(content, plaintext)
        self.assertEqual(self.fernet.decrypt(content), plaintext)

    def test_encrypt_local_file_noop_when_disabled(self):
        os.makedirs(os.path.join(self.work_dir, 'noop'), exist_ok=True)
        exporter = Exporter(
            export_path=os.path.join(self.work_dir, 'noop', 'out.jsonl'),
            encrypt_before_export=False,
        )
        plain_path = os.path.join(self.work_dir, 'plain.txt')
        plaintext = b'untouched'
        with open(plain_path, 'wb') as f:
            f.write(plaintext)

        exporter._encrypt_local_file(plain_path)

        with open(plain_path, 'rb') as f:
            self.assertEqual(f.read(), plaintext)

    # ------------------------------------------------------------------
    # Full export round-trip
    # ------------------------------------------------------------------

    def test_export_single_file_is_encrypted(self):
        key_file = self._write_key_file()
        export_path = os.path.join(self.work_dir, 'enc', 'out.jsonl')
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        exporter.export(self.test_data)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, 'rb') as f:
            raw = f.read()
        # Must not be plaintext JSON
        self.assertFalse(raw.lstrip().startswith(b'{'))
        # Must be decryptable
        decrypted = self.fernet.decrypt(raw)
        self.assertIn(b'hello', decrypted)

    def test_export_stats_file_is_encrypted(self):
        key_file = self._write_key_file()
        export_path = os.path.join(self.work_dir, 'enc_stats', 'out.jsonl')
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
            encrypt_before_export=True,
            encryption_key_path=key_file,
        )
        exporter.export(self.test_data)

        # stats file naming rule: replace ".jsonl" with "_stats.jsonl"
        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertTrue(os.path.exists(stats_path))
        with open(stats_path, 'rb') as f:
            raw = f.read()
        # Stats file must be encrypted
        self.assertFalse(raw.lstrip().startswith(b'{'))
        self.fernet.decrypt(raw)  # must not raise


class CoreExporterFileTest(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="dj_exporter_file_")
        self.Exporter = exporter_module.Exporter
        self.Fields = exporter_module.Fields
        self.HashKeys = exporter_module.HashKeys
        self.dataset = Dataset.from_list([
            {
                "text": "text 1",
                self.Fields.stats: {"score": 1},
                self.Fields.meta: {"source": "a"},
                self.HashKeys.hash: "h1",
            },
            {
                "text": "text 2",
                self.Fields.stats: {"score": 2},
                self.Fields.meta: {"source": "b"},
                self.HashKeys.hash: "h2",
            },
            {
                "text": "text 3",
                self.Fields.stats: {"score": 3},
                self.Fields.meta: {"source": "c"},
                self.HashKeys.hash: "h3",
            },
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_meta_stats_json_strings_are_restored_before_export(self):
        ds = Dataset.from_dict({
            self.Fields.meta: ['{"source": "zh"}', '{"source": "en"}'],
            self.Fields.stats: ['{"score": 1}', '{"score": 2}'],
        })

        fixed = self.Exporter._ensure_meta_stats_dicts_for_export(ds)

        self.assertEqual(fixed[0][self.Fields.meta], {"source": "zh"})
        self.assertEqual(fixed[0][self.Fields.stats], {"score": 1})
        self.assertEqual(fixed[1][self.Fields.meta], {"source": "en"})
        self.assertEqual(fixed[1][self.Fields.stats], {"score": 2})

    def test_invalid_meta_stats_json_strings_are_left_unchanged(self):
        ds = Dataset.from_dict({
            self.Fields.meta: ["not-json"],
            self.Fields.stats: ["also-not-json"],
        })

        fixed = self.Exporter._ensure_meta_stats_dicts_for_export(ds)

        self.assertEqual(fixed[0][self.Fields.meta], "not-json")
        self.assertEqual(fixed[0][self.Fields.stats], "also-not-json")

    def test_meta_stats_restore_is_noop_without_columns(self):
        ds = Dataset.from_list([{"text": "plain"}])

        self.assertIs(self.Exporter._ensure_meta_stats_dicts_for_export(ds), ds)

    def test_row_to_json_serializable_handles_scalars_lists_and_arrow_values(self):
        class ArrowLike:
            def as_py(self):
                return {"nested": np.int64(3)}

        class ListLike:
            def tolist(self):
                return [1, 2]

        row = {
            "scalar": np.int64(7),
            "array": ListLike(),
            "nested": [ArrowLike()],
        }

        self.assertEqual(
            self.Exporter._row_to_json_serializable(row),
            {"scalar": 7, "array": [1, 2], "nested": [{"nested": 3}]},
        )

    def test_json_jsonl_parquet_exports_and_filtered_shards(self):
        jsonl_path = os.path.join(self.tmp_dir, "out.jsonl")
        json_path = os.path.join(self.tmp_dir, "out.json")
        parquet_path = os.path.join(self.tmp_dir, "out.parquet")
        shard_path = os.path.join(self.tmp_dir, "shards", "out.jsonl")

        self.Exporter.to_jsonl(self.dataset, jsonl_path)
        self.Exporter.to_json(self.dataset, json_path, num_proc=1)
        self.Exporter.to_parquet(self.dataset, parquet_path)

        with open(jsonl_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(rows[0]["text"], "text 1")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(parquet_path))

        filtered = self.dataset.filter(lambda row: row["text"] != "text 2")
        exporter = self.Exporter(
            export_path=shard_path,
            export_shard_size=1,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(filtered)

        shard_files = os.listdir(os.path.dirname(shard_path))
        self.assertTrue(any(name.endswith(".jsonl") for name in shard_files))


class ExporterRouterAndSuffixTest(DataJuicerTestCaseBase):
    """Tests for _router and _get_suffix methods."""

    def test_router_returns_correct_methods(self):
        router = Exporter._router()
        self.assertIn('jsonl', router)
        self.assertIn('json', router)
        self.assertIn('parquet', router)
        self.assertEqual(len(router), 3)
        self.assertEqual(router['jsonl'], Exporter.to_jsonl)
        self.assertEqual(router['json'], Exporter.to_json)
        self.assertEqual(router['parquet'], Exporter.to_parquet)

    def test_get_suffix_extracts_extension(self):
        exp = Exporter(export_path='/tmp/test_get_suffix/out.jsonl')
        self.assertEqual(exp._get_suffix('/tmp/foo.jsonl'), 'jsonl')
        self.assertEqual(exp._get_suffix('/tmp/foo.json'), 'json')
        self.assertEqual(exp._get_suffix('/tmp/foo.parquet'), 'parquet')
        self.assertEqual(exp._get_suffix('/tmp/dir.name/foo.JSONL'), 'jsonl')

    def test_explicit_export_type_overrides_suffix(self):
        """When export_type is provided, it should be used instead of the
        file extension."""
        exp = Exporter(
            export_path='/tmp/test_export_type/out.txt',
            export_type='jsonl',
        )
        self.assertEqual(exp.suffix, 'jsonl')

    def test_unsupported_export_type_raises(self):
        with self.assertRaises(NotImplementedError):
            Exporter(
                export_path='/tmp/test_unsupported/out.jsonl',
                export_type='csv',
            )


class ExporterKeepFieldsTest(DataJuicerTestCaseBase):
    """Tests for keep_stats_in_res_ds and keep_hashes_in_res_ds options."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_keep_fields/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {
                'text': 'hello',
                Fields.stats: {'score': 1},
                Fields.meta: {'src': 'a'},
                HashKeys.hash: 'h1',
                HashKeys.minhash: 'mh1',
            },
            {
                'text': 'world',
                Fields.stats: {'score': 2},
                Fields.meta: {'src': 'b'},
                HashKeys.hash: 'h2',
                HashKeys.minhash: 'mh2',
            },
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_keep_stats_in_res_ds_true(self):
        """When keep_stats_in_res_ds=True, the exported dataset should
        contain stats and meta columns."""
        export_path = os.path.join(self.work_dir, 'keep_stats', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(self.dataset)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        # stats and meta should be present
        self.assertIn(Fields.stats, rows[0])
        self.assertIn(Fields.meta, rows[0])
        # hashes should NOT be present
        self.assertNotIn(HashKeys.hash, rows[0])
        self.assertNotIn(HashKeys.minhash, rows[0])

    def test_keep_hashes_in_res_ds_true(self):
        """When keep_hashes_in_res_ds=True, the exported dataset should
        contain hash columns."""
        export_path = os.path.join(self.work_dir, 'keep_hashes', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=True,
            export_stats=False,
        )
        exporter.export(self.dataset)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        # hashes should be present
        self.assertIn(HashKeys.hash, rows[0])
        self.assertIn(HashKeys.minhash, rows[0])
        # stats/meta should NOT be present
        self.assertNotIn(Fields.stats, rows[0])
        self.assertNotIn(Fields.meta, rows[0])

    def test_keep_both_stats_and_hashes(self):
        """Both keep flags set to True should retain all fields."""
        export_path = os.path.join(self.work_dir, 'keep_both', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=True,
            export_stats=False,
        )
        exporter.export(self.dataset)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertIn(Fields.stats, rows[0])
        self.assertIn(Fields.meta, rows[0])
        self.assertIn(HashKeys.hash, rows[0])
        self.assertIn(HashKeys.minhash, rows[0])
        self.assertIn('text', rows[0])


class ExporterExportDsFalseTest(DataJuicerTestCaseBase):
    """Tests for export_ds=False (only stats are exported)."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_no_ds/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {'text': 'a', Fields.stats: {'x': 1}},
            {'text': 'b', Fields.stats: {'x': 2}},
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_export_ds_false_only_exports_stats(self):
        """When export_ds=False, only the stats file should be created."""
        export_path = os.path.join(self.work_dir, 'no_ds', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=False,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.dataset)

        # The main export file should NOT exist
        self.assertFalse(os.path.exists(export_path))
        # But the stats file should exist
        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertTrue(os.path.exists(stats_path))

    def test_export_ds_false_stats_false_produces_nothing(self):
        """When both export_ds and export_stats are False, nothing is exported."""
        export_path = os.path.join(self.work_dir, 'nothing', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=False,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(self.dataset)

        self.assertFalse(os.path.exists(export_path))
        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertFalse(os.path.exists(stats_path))


class ExporterParallelAndProcTest(DataJuicerTestCaseBase):
    """Tests for export_in_parallel and num_proc interactions."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_parallel/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {'text': f'row {i}', Fields.stats: {'n': i}}
            for i in range(10)
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_export_in_parallel_false_single_file(self):
        """export_in_parallel=False should still produce correct output."""
        export_path = os.path.join(self.work_dir, 'no_parallel', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=2,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.dataset)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 10)

    def test_export_in_parallel_true_with_multiproc(self):
        """export_in_parallel=True with num_proc=2 should work."""
        export_path = os.path.join(self.work_dir, 'parallel', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=True,
            num_proc=2,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=True,
        )
        exporter.export(self.dataset)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 10)

    def test_sharded_export_with_multiproc(self):
        """Shard export with num_proc > 1 uses multiprocessing pool."""
        export_path = os.path.join(self.work_dir, 'shards_mp', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=1,  # very small -> multiple shards
            export_in_parallel=True,
            num_proc=2,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(self.dataset)

        shard_dir = os.path.dirname(os.path.abspath(export_path))
        shard_files = [f for f in os.listdir(shard_dir) if f.endswith('.jsonl')]
        self.assertGreater(len(shard_files), 1)


class ExporterShardFormatsTest(DataJuicerTestCaseBase):
    """Tests for shard export in JSON and Parquet formats."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_shard_formats/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {'text': f'row {i}', Fields.stats: {'n': i}}
            for i in range(5)
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_json_sharded_export(self):
        """Shard export in JSON format should produce multiple .json files."""
        export_path = os.path.join(self.work_dir, 'json_shards', 'out.json')
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

        shard_dir = os.path.dirname(os.path.abspath(export_path))
        shard_files = [f for f in os.listdir(shard_dir) if f.endswith('.json')]
        self.assertGreater(len(shard_files), 1)

    def test_parquet_sharded_export(self):
        """Shard export in Parquet format should produce multiple .parquet files."""
        export_path = os.path.join(self.work_dir, 'parquet_shards', 'out.parquet')
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

        shard_dir = os.path.dirname(os.path.abspath(export_path))
        shard_files = [f for f in os.listdir(shard_dir) if f.endswith('.parquet')]
        self.assertGreater(len(shard_files), 1)


class ExporterUnicodeAndWriteTest(DataJuicerTestCaseBase):
    """Tests for _write_jsonl_utf8 Unicode handling."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_unicode/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_write_jsonl_utf8_preserves_unicode(self):
        """_write_jsonl_utf8 should write proper UTF-8, not escaped sequences."""
        dataset = Dataset.from_list([
            {'text': '你好世界'},
            {'text': 'こんにちは'},
            {'text': 'emoji: 🎉'},
        ])
        export_path = os.path.join(self.work_dir, 'unicode.jsonl')
        Exporter._write_jsonl_utf8(dataset, export_path)

        with open(export_path, encoding='utf-8') as f:
            content = f.read()
        # Should NOT contain escaped Unicode
        self.assertNotIn('\\u', content)
        self.assertIn('你好世界', content)
        self.assertIn('こんにちは', content)
        self.assertIn('🎉', content)

    def test_write_jsonl_utf8_large_batch(self):
        """_write_jsonl_utf8 should correctly handle datasets larger than
        the internal batch size (1000)."""
        dataset = Dataset.from_list([
            {'text': f'item_{i}', 'value': i}
            for i in range(1500)
        ])
        export_path = os.path.join(self.work_dir, 'large.jsonl')
        Exporter._write_jsonl_utf8(dataset, export_path)

        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1500)
        self.assertEqual(rows[0]['text'], 'item_0')
        self.assertEqual(rows[1499]['text'], 'item_1499')

    def test_to_jsonl_uses_utf8_writer_for_local(self):
        """to_jsonl without storage_options should use _write_jsonl_utf8."""
        dataset = Dataset.from_list([
            {'text': '中文测试'},
        ])
        export_path = os.path.join(self.work_dir, 'local_jsonl.jsonl')
        Exporter.to_jsonl(dataset, export_path, num_proc=1)

        with open(export_path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('中文测试', content)
        self.assertNotIn('\\u', content)


class ExporterNoStatsColumnsTest(DataJuicerTestCaseBase):
    """Tests for exporting datasets that lack stats/meta/hash columns."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_no_stats/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {'text': 'plain text 1', 'label': 'A'},
            {'text': 'plain text 2', 'label': 'B'},
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_export_dataset_without_stats_or_hashes(self):
        """Exporting a dataset with no stats/meta/hash columns should work."""
        export_path = os.path.join(self.work_dir, 'plain', 'out.jsonl')
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
        exporter.export(self.dataset)

        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['text'], 'plain text 1')
        # No stats file should be created since there are no stats columns
        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertFalse(os.path.exists(stats_path))

    def test_export_dataset_with_only_meta(self):
        """Exporting a dataset with only meta (no stats or hashes) should work."""
        dataset = Dataset.from_list([
            {'text': 'x', Fields.meta: {'tag': 'v1'}},
            {'text': 'y', Fields.meta: {'tag': 'v2'}},
        ])
        export_path = os.path.join(self.work_dir, 'meta_only', 'out.jsonl')
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
        exporter.export(dataset)

        self.assertTrue(os.path.exists(export_path))
        # Stats file should be created (contains meta)
        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertTrue(os.path.exists(stats_path))
        with open(stats_path, encoding='utf-8') as f:
            stats_rows = [json.loads(line) for line in f]
        self.assertIn(Fields.meta, stats_rows[0])


class ExporterComputeStatsTest(DataJuicerTestCaseBase):
    """Tests for export_compute_stats method behavior."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_compute_stats/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {'text': 'a', Fields.stats: {'score': 10}},
            {'text': 'b', Fields.stats: {'score': 20}},
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_export_compute_stats_keeps_stats_in_output(self):
        """export_compute_stats should temporarily set keep_stats_in_res_ds=True
        and include stats in the exported file."""
        export_path = os.path.join(self.work_dir, 'compute', 'res.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )

        # Before: keep_stats_in_res_ds is False
        self.assertFalse(exporter.keep_stats_in_res_ds)

        exporter.export_compute_stats(self.dataset, export_path)

        # After: keep_stats_in_res_ds should be restored to False
        self.assertFalse(exporter.keep_stats_in_res_ds)

        # The exported file should contain stats
        self.assertTrue(os.path.exists(export_path))
        with open(export_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertIn(Fields.stats, rows[0])
        self.assertEqual(rows[0][Fields.stats], {'score': 10})

    def test_export_compute_stats_no_stats_file(self):
        """export_compute_stats should not produce a separate stats file."""
        export_path = os.path.join(self.work_dir, 'no_stats_file', 'res.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=0,
            export_in_parallel=False,
            num_proc=1,
        )
        exporter.export_compute_stats(self.dataset, export_path)

        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertFalse(os.path.exists(stats_path))


class ExporterToJsonDirectTest(DataJuicerTestCaseBase):
    """Tests for the to_json and to_parquet static methods directly."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_direct/'
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset = Dataset.from_list([
            {'text': 'hello', 'num': 1},
            {'text': 'world', 'num': 2},
        ])

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_to_json_produces_valid_json_array(self):
        """to_json should produce a valid JSON file (not lines)."""
        export_path = os.path.join(self.work_dir, 'direct.json')
        Exporter.to_json(self.dataset, export_path, num_proc=1)

        with open(export_path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['text'], 'hello')

    def test_to_json_with_unicode(self):
        """to_json should preserve Unicode characters."""
        dataset = Dataset.from_list([{'text': '你好'}])
        export_path = os.path.join(self.work_dir, 'unicode.json')
        Exporter.to_json(dataset, export_path, num_proc=1)

        with open(export_path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('你好', content)

    def test_to_parquet_round_trip(self):
        """to_parquet should produce a readable parquet file."""
        import pyarrow.parquet as pq
        export_path = os.path.join(self.work_dir, 'direct.parquet')
        Exporter.to_parquet(self.dataset, export_path)

        table = pq.read_table(export_path)
        self.assertEqual(table.num_rows, 2)
        self.assertIn('text', table.column_names)


class ExporterStatsExportWithJsonStringsTest(DataJuicerTestCaseBase):
    """Test that stats export handles JSON-string meta/stats during full export."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_stats_json_str/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_export_with_json_string_stats_produces_valid_stats_file(self):
        """If stats/meta are stored as JSON strings, they should be parsed
        back to dicts for the stats export file."""
        dataset = Dataset.from_dict({
            'text': ['hello', 'world'],
            Fields.stats: ['{"score": 1}', '{"score": 2}'],
            Fields.meta: ['{"src": "a"}', '{"src": "b"}'],
        })
        export_path = os.path.join(self.work_dir, 'str_stats', 'out.jsonl')
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
        exporter.export(dataset)

        stats_path = export_path.replace('.jsonl', '_stats.jsonl')
        self.assertTrue(os.path.exists(stats_path))
        with open(stats_path, encoding='utf-8') as f:
            stats_rows = [json.loads(line) for line in f]
        # The parsed stats should be dicts, not strings
        self.assertEqual(stats_rows[0][Fields.stats], {'score': 1})
        self.assertEqual(stats_rows[0][Fields.meta], {'src': 'a'})


class ExporterShardSizeWarningTest(DataJuicerTestCaseBase):
    """Test shard size warning messages."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_warnings/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_small_shard_size_warning(self):
        """Shard size < 1 MiB should trigger a warning."""
        from loguru import logger
        warnings_captured = []
        handler_id = logger.add(
            lambda msg: warnings_captured.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            Exporter(
                export_path=os.path.join(self.work_dir, 'small.jsonl'),
                export_shard_size=500,  # 500 bytes, less than 1 MiB
            )
        finally:
            logger.remove(handler_id)

        self.assertTrue(
            any('less than 1MiB' in w for w in warnings_captured),
            f'Expected warning about small shard size, got: {warnings_captured}'
        )

    def test_large_shard_size_warning(self):
        """Shard size >= 1 TiB should trigger a warning."""
        from loguru import logger
        from data_juicer.utils.file_utils import Sizes
        warnings_captured = []
        handler_id = logger.add(
            lambda msg: warnings_captured.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            Exporter(
                export_path=os.path.join(self.work_dir, 'large.jsonl'),
                export_shard_size=Sizes.TiB,
            )
        finally:
            logger.remove(handler_id)

        self.assertTrue(
            any('larger than 1TiB' in w for w in warnings_captured),
            f'Expected warning about large shard size, got: {warnings_captured}'
        )


class ExporterFilteredDatasetShardTest(DataJuicerTestCaseBase):
    """Test sharding with filtered datasets (exercises _indices path)."""

    def setUp(self):
        super().setUp()
        self.work_dir = 'tmp/test_exporter_filtered_shard/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_filtered_dataset_shard_computes_correct_nbytes(self):
        """When a dataset has _indices (from filter), nbytes is adjusted
        proportionally."""
        dataset = Dataset.from_list([
            {'text': f'row {i}', 'value': i}
            for i in range(20)
        ])
        # Filter creates _indices
        filtered = dataset.filter(lambda x: x['value'] % 2 == 0)
        self.assertEqual(len(filtered), 10)

        export_path = os.path.join(self.work_dir, 'filtered', 'out.jsonl')
        exporter = Exporter(
            export_path=export_path,
            export_shard_size=1,  # tiny -> multiple shards
            export_in_parallel=False,
            num_proc=1,
            export_ds=True,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False,
            export_stats=False,
        )
        exporter.export(filtered)

        shard_dir = os.path.dirname(os.path.abspath(export_path))
        shard_files = [f for f in os.listdir(shard_dir) if f.endswith('.jsonl')]
        self.assertGreater(len(shard_files), 1)

        # Verify all rows are present across shards
        all_rows = []
        for sf in sorted(shard_files):
            with open(os.path.join(shard_dir, sf), encoding='utf-8') as f:
                all_rows.extend([json.loads(line) for line in f])
        self.assertEqual(len(all_rows), 10)
        values = sorted([r['value'] for r in all_rows])
        self.assertEqual(values, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])


if __name__ == '__main__':
    unittest.main()
