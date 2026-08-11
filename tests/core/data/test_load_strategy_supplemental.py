"""
Supplemental tests for data_juicer/core/data/load_strategy.py

Focuses on:
- Format detection logic (file_extension_map usage)
- StrategyKey matching edge cases
- ConfigValidator integration (type errors, custom validators)
- DataLoadStrategy base class defaults (weight)
- Registry lookup with None/empty fields
- RayLocalJsonDataLoadStrategy directory scanning and source-based detection
- DefaultHdfsDataLoadStrategy / RayS3DataLoadStrategy format resolution
"""

import json
import os
import tempfile
import unittest

from jsonargparse import Namespace

from data_juicer.core.data.config_validator import ConfigValidationError
from data_juicer.core.data.load_strategy import (
    DataLoadStrategy,
    DataLoadStrategyRegistry,
    DefaultHdfsDataLoadStrategy,
    DefaultLocalDataLoadStrategy,
    DefaultS3DataLoadStrategy,
    RayHdfsDataLoadStrategy,
    RayLocalJsonDataLoadStrategy,
    RayS3DataLoadStrategy,
    StrategyKey,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestStrategyKeyMatchingEdgeCases(DataJuicerTestCaseBase):
    """Edge cases for StrategyKey.matches"""

    def test_exact_match_returns_true(self):
        key = StrategyKey("default", "local", "json")
        self.assertTrue(key.matches(StrategyKey("default", "local", "json")))

    def test_no_match_different_fields(self):
        key = StrategyKey("default", "local", "json")
        self.assertFalse(key.matches(StrategyKey("ray", "local", "json")))
        self.assertFalse(key.matches(StrategyKey("default", "remote", "json")))
        self.assertFalse(key.matches(StrategyKey("default", "local", "csv")))

    def test_wildcard_executor_only(self):
        key = StrategyKey("*", "local", "json")
        self.assertTrue(key.matches(StrategyKey("ray", "local", "json")))
        self.assertTrue(key.matches(StrategyKey("default", "local", "json")))
        self.assertFalse(key.matches(StrategyKey("ray", "remote", "json")))

    def test_wildcard_all_fields(self):
        key = StrategyKey("*", "*", "*")
        self.assertTrue(key.matches(StrategyKey("ray", "remote", "s3")))
        self.assertTrue(key.matches(StrategyKey("default", "local", "csv")))

    def test_question_mark_pattern(self):
        key = StrategyKey("default", "local", "js?n")
        self.assertTrue(key.matches(StrategyKey("default", "local", "json")))
        self.assertTrue(key.matches(StrategyKey("default", "local", "jsan")))
        self.assertFalse(key.matches(StrategyKey("default", "local", "jsonl")))

    def test_bracket_pattern(self):
        key = StrategyKey("default", "local", "[cj]sv")
        self.assertTrue(key.matches(StrategyKey("default", "local", "csv")))
        self.assertTrue(key.matches(StrategyKey("default", "local", "jsv")))
        self.assertFalse(key.matches(StrategyKey("default", "local", "tsv")))

    def test_negated_bracket_pattern(self):
        key = StrategyKey("default", "local", "[!c]sv")
        self.assertFalse(key.matches(StrategyKey("default", "local", "csv")))
        self.assertTrue(key.matches(StrategyKey("default", "local", "tsv")))

    def test_symmetric_non_match_wildcard(self):
        """A specific key should not match a wildcard key (direction matters)."""
        specific = StrategyKey("default", "local", "json")
        wildcard = StrategyKey("*", "*", "*")
        # wildcard.matches(specific) should be True
        self.assertTrue(wildcard.matches(specific))
        # specific.matches(wildcard) should be False: "default" != "*"
        self.assertFalse(specific.matches(wildcard))


class TestRegistryLookupWithNoneFields(DataJuicerTestCaseBase):
    """Registry defaults None to '*' for lookup."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._orig = DataLoadStrategyRegistry._strategies.copy()

    @classmethod
    def tearDownClass(cls):
        DataLoadStrategyRegistry._strategies = cls._orig
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        DataLoadStrategyRegistry._strategies = {}

    def tearDown(self):
        DataLoadStrategyRegistry._strategies = {}
        super().tearDown()

    def test_none_executor_type_defaults_to_wildcard(self):
        class _Stub(DataLoadStrategy):
            CONFIG_VALIDATION_RULES = {
                "required_fields": [],
                "field_types": {},
                "custom_validators": {},
            }

            def load_data(self, **kwargs):
                pass

        @DataLoadStrategyRegistry.register("*", "local", "json")
        class WildStub(_Stub):
            pass

        result = DataLoadStrategyRegistry.get_strategy_class(None, "local", "json")
        self.assertEqual(result, WildStub)

    def test_none_data_type_defaults_to_wildcard(self):
        class _Stub(DataLoadStrategy):
            CONFIG_VALIDATION_RULES = {
                "required_fields": [],
                "field_types": {},
                "custom_validators": {},
            }

            def load_data(self, **kwargs):
                pass

        @DataLoadStrategyRegistry.register("default", "*", "json")
        class WildStub2(_Stub):
            pass

        result = DataLoadStrategyRegistry.get_strategy_class("default", None, "json")
        self.assertEqual(result, WildStub2)

    def test_all_none_matches_triple_wildcard(self):
        class _Stub(DataLoadStrategy):
            CONFIG_VALIDATION_RULES = {
                "required_fields": [],
                "field_types": {},
                "custom_validators": {},
            }

            def load_data(self, **kwargs):
                pass

        @DataLoadStrategyRegistry.register("*", "*", "*")
        class GlobalStub(_Stub):
            pass

        result = DataLoadStrategyRegistry.get_strategy_class(None, None, None)
        self.assertEqual(result, GlobalStub)


class TestDataLoadStrategyBaseDefaults(DataJuicerTestCaseBase):
    """Test base class weight and config storage."""

    def _make_concrete(self):
        class ConcreteStrategy(DataLoadStrategy):
            CONFIG_VALIDATION_RULES = {
                "required_fields": [],
                "field_types": {},
                "custom_validators": {},
            }

            def load_data(self, **kwargs):
                return None

        return ConcreteStrategy

    def test_default_weight_is_one(self):
        cls = self._make_concrete()
        s = cls({"path": "/tmp/x"}, Namespace())
        self.assertEqual(s.weight, 1.0)

    def test_custom_weight(self):
        cls = self._make_concrete()
        s = cls({"path": "/tmp/x", "weight": 0.5}, Namespace())
        self.assertEqual(s.weight, 0.5)

    def test_ds_config_stored(self):
        cls = self._make_concrete()
        cfg_dict = {"path": "/tmp/x", "extra": "val"}
        s = cls(cfg_dict, Namespace())
        self.assertIs(s.ds_config, cfg_dict)

    def test_cfg_stored(self):
        cls = self._make_concrete()
        ns = Namespace(text_keys=["content"])
        s = cls({"path": "/tmp/x"}, ns)
        self.assertIs(s.cfg, ns)


class TestConfigValidatorIntegration(DataJuicerTestCaseBase):
    """Test ConfigValidator via DataLoadStrategy subclasses."""

    def test_missing_required_field_raises(self):
        """Strategy requiring 'path' should raise on missing path."""
        with self.assertRaises(ConfigValidationError) as ctx:
            DefaultLocalDataLoadStrategy({}, Namespace())
        self.assertIn("path", str(ctx.exception))

    def test_wrong_type_raises(self):
        """path field must be str."""
        with self.assertRaises(ConfigValidationError) as ctx:
            DefaultLocalDataLoadStrategy({"path": 123}, Namespace())
        self.assertIn("path", str(ctx.exception))

    def test_custom_validator_s3_path(self):
        """S3 validate_s3_path raises ValueError for non-s3:// paths."""
        from data_juicer.utils.s3_utils import validate_s3_path

        with self.assertRaises(ValueError):
            validate_s3_path("/local/file.jsonl")

    def test_custom_validator_hdfs_path(self):
        """HDFS validate_hdfs_path rejects non-hdfs:// paths during load."""
        from data_juicer.utils.hdfs_utils import validate_hdfs_path

        with self.assertRaises(ValueError):
            validate_hdfs_path("s3://bucket/file.jsonl")

    def test_valid_config_no_error(self):
        """Valid config should not raise."""
        s = DefaultS3DataLoadStrategy(
            {"path": "s3://bucket/data.jsonl"}, Namespace(text_keys=["text"])
        )
        self.assertEqual(s.ds_config["path"], "s3://bucket/data.jsonl")


class TestRayLocalFormatDetection(DataJuicerTestCaseBase):
    """Test file extension detection in RayLocalJsonDataLoadStrategy."""

    def setUp(self):
        super().setUp()
        from data_juicer.config import get_default_cfg

        self.cfg = get_default_cfg()
        self.cfg.ray_address = "local"
        self.cfg.executor_type = "ray"

    def _write_jsonl(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_detect_json_from_extension(self):
        """Files with .json/.jsonl should be read as json format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(fpath, [{"text": "a"}, {"text": "b"}])
            strategy = RayLocalJsonDataLoadStrategy({"path": fpath}, self.cfg)
            ds = strategy.load_data()
            rows = list(ds.get(2))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "a")

    def test_detect_csv_from_extension(self):
        """Files with .csv should be read as csv format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.csv")
            with open(fpath, "w") as f:
                f.write("text\n")
                f.write("hello\n")
                f.write("world\n")
            strategy = RayLocalJsonDataLoadStrategy({"path": fpath}, self.cfg)
            ds = strategy.load_data()
            rows = list(ds.get(2))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "hello")

    def test_detect_parquet_from_extension(self):
        """Files with .parquet should be read as parquet format."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.parquet")
            table = pa.table({"text": ["row1", "row2"]})
            pq.write_table(table, fpath)
            strategy = RayLocalJsonDataLoadStrategy({"path": fpath}, self.cfg)
            ds = strategy.load_data()
            rows = list(ds.get(2))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "row1")

    def test_detect_format_from_directory(self):
        """When path is a directory, the first file found determines format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "chunk.jsonl")
            self._write_jsonl(fpath, [{"text": "dir_test"}])
            strategy = RayLocalJsonDataLoadStrategy({"path": tmpdir}, self.cfg)
            ds = strategy.load_data()
            rows = list(ds.get(1))
            self.assertEqual(rows[0]["text"], "dir_test")

    def test_source_field_overrides_auto_detection(self):
        """When ds_config['source'] specifies an extension, use it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File is actually jsonl but named without extension
            fpath = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(fpath, [{"text": "src_test"}])
            strategy = RayLocalJsonDataLoadStrategy(
                {"path": fpath, "source": "jsonl"}, self.cfg
            )
            ds = strategy.load_data()
            rows = list(ds.get(1))
            self.assertEqual(rows[0]["text"], "src_test")

    def test_source_field_with_dot_extension(self):
        """source can be specified as '.jsonl' (with leading dot)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.jsonl")
            self._write_jsonl(fpath, [{"text": "dot_test"}])
            strategy = RayLocalJsonDataLoadStrategy(
                {"path": fpath, "source": "file.jsonl"}, self.cfg
            )
            ds = strategy.load_data()
            rows = list(ds.get(1))
            self.assertEqual(rows[0]["text"], "dot_test")

    def test_nonexistent_path_raises_error(self):
        """Missing absolute path raises RuntimeError during read."""
        strategy = RayLocalJsonDataLoadStrategy(
            {"path": "/nonexistent/path/data.jsonl"}, self.cfg
        )
        with self.assertRaises(RuntimeError):
            strategy.load_data()

    def test_nonexistent_relative_path_raises_file_not_found(self):
        """Missing relative path raises FileNotFoundError."""
        strategy = RayLocalJsonDataLoadStrategy(
            {"path": "nonexistent_relative/data.jsonl"}, self.cfg
        )
        with self.assertRaises(FileNotFoundError):
            strategy.load_data()

    def test_relative_path_cwd_resolution(self):
        """Relative path resolved from cwd."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "rel.jsonl")
            self._write_jsonl(fpath, [{"text": "cwd"}])
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                strategy = RayLocalJsonDataLoadStrategy(
                    {"path": "rel.jsonl"}, self.cfg
                )
                ds = strategy.load_data()
                rows = list(ds.get(1))
                self.assertEqual(rows[0]["text"], "cwd")
            finally:
                os.chdir(old_cwd)


class TestRayS3FormatDetection(DataJuicerTestCaseBase):
    """Test format detection in RayS3DataLoadStrategy without actual S3 access."""

    def setUp(self):
        super().setUp()
        from data_juicer.config import get_default_cfg

        self.cfg = get_default_cfg()
        self.cfg.text_keys = ["text"]

    def test_auto_detect_jsonl(self):
        """Path ending in .jsonl should auto-detect as json."""
        ds_config = {"path": "s3://bucket/data.jsonl"}
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        # We only test config initialization, not actual loading
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/data.jsonl")

    def test_auto_detect_parquet(self):
        """Path ending in .parquet should auto-detect as parquet."""
        ds_config = {"path": "s3://bucket/data.parquet"}
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/data.parquet")

    def test_explicit_format_field(self):
        """Explicit 'format' field in config should be used."""
        ds_config = {"path": "s3://bucket/data", "format": "parquet"}
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["format"], "parquet")

    def test_format_field_as_extension_name(self):
        """format='jsonl' should be interpreted correctly."""
        ds_config = {"path": "s3://bucket/data", "format": "jsonl"}
        strategy = RayS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["format"], "jsonl")


class TestDefaultHdfsFormatDetection(DataJuicerTestCaseBase):
    """Test format detection logic in DefaultHdfsDataLoadStrategy."""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_format_from_extension_jsonl(self):
        ds_config = {"path": "hdfs://namenode:8020/data/file.jsonl"}
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        # format not specified, will auto-detect from extension during load_data
        self.assertNotIn("format", strategy.ds_config)

    def test_format_from_config_explicit(self):
        ds_config = {
            "path": "hdfs://namenode:8020/data/file",
            "format": "parquet",
        }
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["format"], "parquet")

    def test_format_from_config_as_extension(self):
        """format='jsonl' should be interpreted as json during load."""
        ds_config = {
            "path": "hdfs://namenode:8020/data/file",
            "format": "jsonl",
        }
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["format"], "jsonl")

    def test_csv_extension_detected(self):
        ds_config = {"path": "hdfs://namenode:8020/data/file.csv"}
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "hdfs://namenode:8020/data/file.csv")

    def test_parquet_extension_detected(self):
        ds_config = {"path": "hdfs://namenode:8020/data/file.parquet"}
        strategy = DefaultHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "hdfs://namenode:8020/data/file.parquet")


class TestRayHdfsFormatDetection(DataJuicerTestCaseBase):
    """Test format detection in RayHdfsDataLoadStrategy."""

    def setUp(self):
        super().setUp()
        from data_juicer.config import get_default_cfg

        self.cfg = get_default_cfg()
        self.cfg.text_keys = ["text"]

    def test_valid_config_with_format(self):
        ds_config = {
            "path": "hdfs://namenode:8020/data/file.jsonl",
            "format": "json",
        }
        strategy = RayHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["format"], "json")

    def test_valid_config_without_format(self):
        ds_config = {"path": "hdfs://namenode:8020/data/file.parquet"}
        strategy = RayHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertNotIn("format", strategy.ds_config)

    def test_config_with_optional_hdfs_fields(self):
        ds_config = {
            "path": "hdfs://namenode:8020/data/file.jsonl",
            "format": "json",
            "hdfs_host": "namenode",
            "hdfs_port": 8020,
            "hdfs_user": "testuser",
            "hdfs_kerb_ticket": "/tmp/krb",
            "hdfs_extra_conf": {"dfs.replication": "2"},
        }
        strategy = RayHdfsDataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["hdfs_host"], "namenode")
        self.assertEqual(strategy.ds_config["hdfs_port"], 8020)
        self.assertEqual(strategy.ds_config["hdfs_user"], "testuser")


class TestDefaultS3FormatDetection(DataJuicerTestCaseBase):
    """Test format detection in DefaultS3DataLoadStrategy."""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace(text_keys=["text"])

    def test_jsonl_extension(self):
        ds_config = {"path": "s3://bucket/file.jsonl"}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/file.jsonl")

    def test_json_extension(self):
        ds_config = {"path": "s3://bucket/file.json"}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/file.json")

    def test_parquet_extension(self):
        ds_config = {"path": "s3://bucket/file.parquet"}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/file.parquet")

    def test_csv_extension(self):
        ds_config = {"path": "s3://bucket/file.csv"}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/file.csv")

    def test_tsv_extension(self):
        ds_config = {"path": "s3://bucket/file.tsv"}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/file.tsv")

    def test_no_extension_defaults_json(self):
        """No extension defaults to json format in the load logic."""
        ds_config = {"path": "s3://bucket/data"}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertEqual(strategy.ds_config["path"], "s3://bucket/data")

    def test_anonymous_access_config(self):
        ds_config = {"path": "s3://public-bucket/file.jsonl", "anon": True}
        strategy = DefaultS3DataLoadStrategy(ds_config, self.cfg)
        self.assertTrue(strategy.ds_config["anon"])


class TestDefaultLocalDataLoadStrategy(DataJuicerTestCaseBase):
    """Test DefaultLocalDataLoadStrategy loading actual files."""

    def test_load_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.jsonl")
            with open(fpath, "w") as f:
                f.write(json.dumps({"text": "line1"}) + "\n")
                f.write(json.dumps({"text": "line2"}) + "\n")

            cfg = Namespace(text_keys=["text"], suffixes=None, process=[])
            strategy = DefaultLocalDataLoadStrategy({"path": fpath}, cfg)
            ds = strategy.load_data(num_proc=1)
            rows = ds.to_list()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "line1")
            self.assertEqual(rows[1]["text"], "line2")

    def test_load_csv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.csv")
            with open(fpath, "w") as f:
                f.write("text\n")
                f.write("csv_line1\n")
                f.write("csv_line2\n")

            cfg = Namespace(text_keys=["text"], suffixes=None, process=[])
            strategy = DefaultLocalDataLoadStrategy({"path": fpath}, cfg)
            ds = strategy.load_data(num_proc=1)
            rows = ds.to_list()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["text"], "csv_line1")

    def test_load_parquet_file(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.parquet")
            table = pa.table({"text": ["pq1", "pq2", "pq3"]})
            pq.write_table(table, fpath)

            cfg = Namespace(text_keys=["text"], suffixes=None, process=[])
            strategy = DefaultLocalDataLoadStrategy({"path": fpath}, cfg)
            ds = strategy.load_data(num_proc=1)
            rows = ds.to_list()
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[2]["text"], "pq3")

    def test_suffix_filter_enables_add_suffix(self):
        """When process contains suffix_filter, the suffix field is added."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.jsonl")
            with open(fpath, "w") as f:
                f.write(json.dumps({"text": "sf_test"}) + "\n")

            cfg = Namespace(
                text_keys=["text"],
                suffixes=None,
                process=[{"suffix_filter": {}}],
            )
            strategy = DefaultLocalDataLoadStrategy({"path": fpath}, cfg)
            ds = strategy.load_data(num_proc=1)
            from data_juicer.utils.constant import Fields

            self.assertIn(Fields.suffix, ds.features)

    def test_missing_path_raises(self):
        """Missing required 'path' field raises ConfigValidationError."""
        with self.assertRaises(ConfigValidationError):
            DefaultLocalDataLoadStrategy({}, Namespace())


class TestRegistrySpecificityOrdering(DataJuicerTestCaseBase):
    """Test that specificity scoring correctly orders matches."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._orig = DataLoadStrategyRegistry._strategies.copy()

    @classmethod
    def tearDownClass(cls):
        DataLoadStrategyRegistry._strategies = cls._orig
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        DataLoadStrategyRegistry._strategies = {}

    def tearDown(self):
        DataLoadStrategyRegistry._strategies = {}
        super().tearDown()

    def test_two_wildcards_less_specific_than_one(self):
        class _Base(DataLoadStrategy):
            CONFIG_VALIDATION_RULES = {
                "required_fields": [],
                "field_types": {},
                "custom_validators": {},
            }

            def load_data(self, **kwargs):
                pass

        @DataLoadStrategyRegistry.register("*", "*", "json")
        class TwoWild(_Base):
            pass

        @DataLoadStrategyRegistry.register("default", "*", "json")
        class OneWild(_Base):
            pass

        result = DataLoadStrategyRegistry.get_strategy_class(
            "default", "local", "json"
        )
        self.assertEqual(result, OneWild)

    def test_no_match_returns_none(self):
        result = DataLoadStrategyRegistry.get_strategy_class(
            "spark", "streaming", "kafka"
        )
        self.assertIsNone(result)

    def test_exact_beats_wildcard(self):
        class _Base(DataLoadStrategy):
            CONFIG_VALIDATION_RULES = {
                "required_fields": [],
                "field_types": {},
                "custom_validators": {},
            }

            def load_data(self, **kwargs):
                pass

        @DataLoadStrategyRegistry.register("*", "*", "*")
        class CatchAll(_Base):
            pass

        @DataLoadStrategyRegistry.register("ray", "local", "csv")
        class Exact(_Base):
            pass

        result = DataLoadStrategyRegistry.get_strategy_class(
            "ray", "local", "csv"
        )
        self.assertEqual(result, Exact)


class TestRayLocalDirectoryScanFormats(DataJuicerTestCaseBase):
    """Test directory scanning for format detection in RayLocalJsonDataLoadStrategy."""

    def setUp(self):
        super().setUp()
        from data_juicer.config import get_default_cfg

        self.cfg = get_default_cfg()
        self.cfg.ray_address = "local"
        self.cfg.executor_type = "ray"

    def test_nested_directory_finds_first_file(self):
        """Nested directories are scanned to find the first file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub1", "sub2")
            os.makedirs(subdir)
            fpath = os.path.join(subdir, "nested.jsonl")
            with open(fpath, "w") as f:
                f.write(json.dumps({"text": "nested"}) + "\n")

            strategy = RayLocalJsonDataLoadStrategy({"path": tmpdir}, self.cfg)
            ds = strategy.load_data()
            rows = list(ds.get(1))
            self.assertEqual(rows[0]["text"], "nested")

    def test_txt_extension_detected_as_text(self):
        """Files with .txt should be detected as text format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.txt")
            with open(fpath, "w") as f:
                f.write("line one\n")
                f.write("line two\n")
            strategy = RayLocalJsonDataLoadStrategy({"path": fpath}, self.cfg)
            ds = strategy.load_data()
            rows = list(ds.get(2))
            self.assertEqual(len(rows), 2)

    def test_unknown_extension_defaults_to_json(self):
        """Unknown file extensions default to json format, raising RuntimeError
        if the file doesn't match json expectations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "data.xyz")
            with open(fpath, "w") as f:
                f.write(json.dumps({"text": "unknown_ext"}) + "\n")
            strategy = RayLocalJsonDataLoadStrategy({"path": fpath}, self.cfg)
            # The format defaults to json but RayDataset.read expects
            # files with json extensions, so this raises RuntimeError
            with self.assertRaises(RuntimeError) as ctx:
                strategy.load_data()
            self.assertIn("Failed to load data from", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
