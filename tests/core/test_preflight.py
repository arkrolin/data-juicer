"""
Tests for the preflight check system.

These tests verify that configuration errors are caught early (at preflight)
rather than silently passing through to runtime.
"""

import os
import sys
import tempfile
import unittest
from typing import List, Optional, Union
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG

sys.path.insert(0, ".")




class TestPreflightPreInstantiation(unittest.TestCase):
    """
    Tests for pre-instantiation checks (before load_ops).
    These test the DESIRED behavior after preflight is implemented.
    """

    def test_unknown_op_name_with_suggestion(self):
        """Should report the error with fuzzy-match suggestions."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"remove_long_wordss_mapper": {"min_len": 1}}])

        error = ctx.exception
        # Should contain the bad op name and suggestions
        self.assertIn("remove_long_wordss_mapper", str(error))
        self.assertIn("remove_long_words_mapper", str(error))  # suggestion

    def test_completely_unknown_op_name(self):
        """Should report error even without close matches."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"totally_fake_xyz_op": None}])
        self.assertIn("totally_fake_xyz_op", str(ctx.exception))

    def test_unknown_param_name_detected(self):
        """Should catch typos in parameter names."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"remove_long_words_mapper": {"min_length": 1}}])

        error = ctx.exception
        self.assertIn("min_length", str(error))
        self.assertIn("min_len", str(error))  # suggestion

    def test_wrong_param_type_detected(self):
        """Should catch type mismatches based on annotations."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"remove_long_words_mapper": {"min_len": "hello"}}])

        error = ctx.exception
        self.assertIn("min_len", str(error))
        self.assertIn("int", str(error))

    def test_numeric_type_coercion_accepted(self):
        """int annotation should accept int-compatible float (e.g., 10.0)."""
        from data_juicer.core.preflight import pre_instantiation_check

        # Should NOT raise - 10.0 is int-compatible (YAML may parse 10 as 10.0)
        pre_instantiation_check([{"remove_long_words_mapper": {"min_len": 10}}])
        pre_instantiation_check([{"remove_long_words_mapper": {"min_len": 10.0}}])

    def test_none_args_accepted(self):
        """OP with null config (all defaults) should pass."""
        from data_juicer.core.preflight import pre_instantiation_check

        # Should NOT raise
        pre_instantiation_check([{"remove_long_words_mapper": None}])

    def test_valid_config_passes(self):
        """A fully valid config should pass without errors."""
        from data_juicer.core.preflight import pre_instantiation_check

        pre_instantiation_check([
            {"remove_long_words_mapper": {"min_len": 1, "max_len": 100}},
            {"language_id_score_filter": {"lang": "en", "min_score": 0.8}},
        ])

    def test_multiple_errors_collected(self):
        """All errors in the pipeline should be reported at once."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([
                {"remove_long_words_mapper": {"min_length": 1}},  # typo param
                {"fake_op_xyz": None},  # bad op name
            ])

        error = ctx.exception
        # Both errors should be reported
        self.assertTrue(len(error.errors) >= 2)

    def test_base_op_params_accepted(self):
        """Parameters defined in OP base class should be valid for any op."""
        from data_juicer.core.preflight import pre_instantiation_check

        # These are base class params, should be valid for any OP
        pre_instantiation_check([
            {"remove_long_words_mapper": {
                "min_len": 1,
                "text_key": "content",
                "batch_size": 500,
                "num_proc": 4,
            }},
        ])


class TestPreflightPostInstantiation(unittest.TestCase):
    """
    Tests for post-instantiation checks (after load_ops, before process).
    """

    def test_missing_custom_text_key_in_schema(self):
        """Should detect when a user-specified (non-default) text_key doesn't exist."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import (
            PipelineRuntimeError,
            post_instantiation_check,
        )
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"text_key": "content"}}])
        schema = Schema(column_types={"text": str, "meta": dict}, columns=["text", "meta"])

        with self.assertRaises(PipelineRuntimeError) as ctx:
            post_instantiation_check(ops, schema)

        self.assertIn("content", str(ctx.exception))

    def test_default_text_key_missing_not_blocked(self):
        """Default text_key missing should NOT block — multimedia ops may not need it."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"min_len": 1}}])
        # Dataset without 'text' column — but since text_key is default, don't block
        schema = Schema(column_types={"images": str, "meta": dict}, columns=["images", "meta"])

        # Should NOT raise
        post_instantiation_check(ops, schema)

    def test_valid_schema_passes(self):
        """When text_key exists in schema, should pass."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"min_len": 1}}])
        schema = Schema(column_types={"text": str}, columns=["text"])

        # Should not raise
        post_instantiation_check(ops, schema)

    def test_unsupported_op_type_in_ray_mode(self):
        """Selector/Grouper/Aggregator should be rejected in Ray mode."""
        from types import SimpleNamespace

        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import (
            PipelineRuntimeError,
            post_instantiation_check,
        )
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"topk_specified_field_selector": {
            "field_key": "stats.lang_score",
            "top_ratio": 0.5,
        }}])
        schema = Schema(column_types={"text": str}, columns=["text"])
        cfg = SimpleNamespace(executor_type="ray", export_path="/tmp/out")

        with self.assertRaises(PipelineRuntimeError) as ctx:
            post_instantiation_check(ops, schema, cfg)

        self.assertIn("does not support executor mode", str(ctx.exception))

    def test_mapper_in_ray_mode_passes(self):
        """Mapper should be fine in Ray mode."""
        from types import SimpleNamespace

        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check
        from data_juicer.ops.load import load_ops

        ops = load_ops([{"remove_long_words_mapper": {"min_len": 1}}])
        schema = Schema(column_types={"text": str}, columns=["text"])
        cfg = SimpleNamespace(executor_type="ray", export_path="/tmp/out")

        # Should not raise
        post_instantiation_check(ops, schema, cfg)





class TestPreInstantiationEdgeCases(DataJuicerTestCaseBase):
    """Tests for pre_instantiation_check edge cases."""

    @TEST_TAG("standalone")
    def test_empty_process_list_returns_immediately(self):
        """Empty list should return without error."""
        from data_juicer.core.preflight import pre_instantiation_check

        # Should NOT raise
        pre_instantiation_check([])

    @TEST_TAG("standalone")
    def test_malformed_entry_not_a_dict(self):
        """A non-dict entry should produce PreflightError with op_name '__config__'."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check(["not_a_dict"])

        error = ctx.exception
        self.assertEqual(len(error.errors), 1)
        self.assertEqual(error.errors[0].op_name, "__config__")
        self.assertIn("Malformed process entry", error.errors[0].message)

    @TEST_TAG("standalone")
    def test_malformed_entry_dict_wrong_length(self):
        """A dict with more than one key should produce PreflightError with op_name '__config__'."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{"op_a": None, "op_b": None}])

        error = ctx.exception
        self.assertEqual(len(error.errors), 1)
        self.assertEqual(error.errors[0].op_name, "__config__")
        self.assertIn("Malformed process entry", error.errors[0].message)

    @TEST_TAG("standalone")
    def test_malformed_entry_empty_dict(self):
        """An empty dict should produce PreflightError with op_name '__config__'."""
        from data_juicer.core.preflight import (
            PipelineConfigError,
            pre_instantiation_check,
        )

        with self.assertRaises(PipelineConfigError) as ctx:
            pre_instantiation_check([{}])

        error = ctx.exception
        self.assertEqual(len(error.errors), 1)
        self.assertEqual(error.errors[0].op_name, "__config__")
        self.assertIn("Malformed process entry", error.errors[0].message)


class TestTypeCompatibleAdvanced(DataJuicerTestCaseBase):
    """Tests for _type_compatible with complex typing constructs."""

    @TEST_TAG("standalone")
    def test_union_type_returns_true(self):
        """Union type (has __origin__) should return True."""
        from data_juicer.core.preflight import _type_compatible

        # Union has __origin__ = typing.Union
        self.assertTrue(_type_compatible("hello", Union[str, int]))
        self.assertTrue(_type_compatible(42, Union[str, int]))

    @TEST_TAG("standalone")
    def test_optional_type_returns_true(self):
        """Optional type (has __origin__) should return True."""
        from data_juicer.core.preflight import _type_compatible

        # Optional[X] is Union[X, None] which has __origin__
        self.assertTrue(_type_compatible("hello", Optional[str]))
        self.assertTrue(_type_compatible(None, Optional[str]))

    @TEST_TAG("standalone")
    def test_list_type_returns_true(self):
        """List[X] (has __origin__) should return True."""
        from data_juicer.core.preflight import _type_compatible

        self.assertTrue(_type_compatible([1, 2, 3], List[int]))
        self.assertTrue(_type_compatible("not_a_list", List[int]))

    @TEST_TAG("standalone")
    def test_isinstance_raises_typeerror_returns_true(self):
        """When isinstance raises TypeError, should return True."""
        from data_juicer.core.preflight import _type_compatible

        # Create a type that causes isinstance to raise TypeError
        class BadMeta(type):
            def __instancecheck__(cls, instance):
                raise TypeError("broken instancecheck")

        class BadType(metaclass=BadMeta):
            pass

        # Should return True (graceful fallback)
        self.assertTrue(_type_compatible("anything", BadType))


class TestCheckExportPath(DataJuicerTestCaseBase):
    """Tests for _check_export_path."""

    @TEST_TAG("standalone")
    def test_remote_s3_path_no_errors(self):
        """s3:// scheme should return no errors."""
        from data_juicer.core.preflight import _check_export_path

        cfg = SimpleNamespace(export_path="s3://bucket/path/to/output.jsonl")
        errors = _check_export_path(cfg)
        self.assertEqual(errors, [])

    @TEST_TAG("standalone")
    def test_remote_hdfs_path_no_errors(self):
        """hdfs:// scheme should return no errors."""
        from data_juicer.core.preflight import _check_export_path

        cfg = SimpleNamespace(export_path="hdfs://cluster/data/output.jsonl")
        errors = _check_export_path(cfg)
        self.assertEqual(errors, [])

    @TEST_TAG("standalone")
    def test_remote_oss_path_no_errors(self):
        """oss:// scheme should return no errors."""
        from data_juicer.core.preflight import _check_export_path

        cfg = SimpleNamespace(export_path="oss://bucket/data/output.jsonl")
        errors = _check_export_path(cfg)
        self.assertEqual(errors, [])

    @TEST_TAG("standalone")
    def test_no_export_path_returns_empty(self):
        """No export_path attribute should return empty list."""
        from data_juicer.core.preflight import _check_export_path

        cfg = SimpleNamespace()
        errors = _check_export_path(cfg)
        self.assertEqual(errors, [])

    @TEST_TAG("standalone")
    def test_none_export_path_returns_empty(self):
        """export_path=None should return empty list."""
        from data_juicer.core.preflight import _check_export_path

        cfg = SimpleNamespace(export_path=None)
        errors = _check_export_path(cfg)
        self.assertEqual(errors, [])

    @TEST_TAG("standalone")
    def test_non_writable_local_directory(self):
        """Non-writable local export directory should produce an error."""
        from data_juicer.core.preflight import _check_export_path

        # Create a temporary directory and make it non-writable
        with tempfile.TemporaryDirectory() as tmpdir:
            non_writable_dir = os.path.join(tmpdir, "readonly")
            os.makedirs(non_writable_dir)
            os.chmod(non_writable_dir, 0o555)

            try:
                cfg = SimpleNamespace(
                    export_path=os.path.join(non_writable_dir, "output.jsonl")
                )
                errors = _check_export_path(cfg)
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0].op_name, "__config__")
                self.assertIn("not writable", errors[0].message)
            finally:
                # Restore permissions for cleanup
                os.chmod(non_writable_dir, 0o755)


class TestPostInstantiationEdgeCases(DataJuicerTestCaseBase):
    """Tests for post_instantiation_check edge cases."""

    @TEST_TAG("standalone")
    def test_empty_ops_list_succeeds(self):
        """Empty ops list should succeed with no error."""
        from data_juicer.core.data.schema import Schema
        from data_juicer.core.preflight import post_instantiation_check

        schema = Schema(column_types={"text": str}, columns=["text"])
        # Should NOT raise
        post_instantiation_check([], schema)


class TestPreflightErrorStr(DataJuicerTestCaseBase):
    """Tests for PreflightError.__str__() formatting."""

    @TEST_TAG("standalone")
    def test_str_without_suggestions(self):
        """__str__ without suggestions should not have 'did you mean' part."""
        from data_juicer.core.preflight import PreflightError

        err = PreflightError("my_op", "Something went wrong")
        result = str(err)
        self.assertEqual(result, "[my_op] Something went wrong")
        self.assertNotIn("did you mean", result)

    @TEST_TAG("standalone")
    def test_str_with_suggestions(self):
        """__str__ with suggestions should include 'did you mean' part."""
        from data_juicer.core.preflight import PreflightError

        err = PreflightError("my_op", "Unknown parameter 'foo'", ["foo_bar", "foo_baz"])
        result = str(err)
        self.assertIn("[my_op]", result)
        self.assertIn("Unknown parameter 'foo'", result)
        self.assertIn("did you mean: foo_bar, foo_baz?", result)

    @TEST_TAG("standalone")
    def test_str_with_empty_suggestions_list(self):
        """__str__ with explicit empty suggestions list should not have 'did you mean'."""
        from data_juicer.core.preflight import PreflightError

        err = PreflightError("op_x", "Bad param", [])
        result = str(err)
        self.assertEqual(result, "[op_x] Bad param")
        self.assertNotIn("did you mean", result)


class TestPipelineConfigErrorFormatting(DataJuicerTestCaseBase):
    """Tests for PipelineConfigError message formatting."""

    @TEST_TAG("standalone")
    def test_single_error_message(self):
        """PipelineConfigError with one error should format correctly."""
        from data_juicer.core.preflight import PipelineConfigError, PreflightError

        errors = [PreflightError("op_a", "Something is wrong")]
        exc = PipelineConfigError(errors)
        msg = str(exc)
        self.assertIn("Pipeline configuration errors detected:", msg)
        self.assertIn("[op_a] Something is wrong", msg)
        self.assertIn("1 error(s) found", msg)
        self.assertIn("strict_preflight: false", msg)

    @TEST_TAG("standalone")
    def test_multiple_errors_message(self):
        """PipelineConfigError with multiple errors should list all."""
        from data_juicer.core.preflight import PipelineConfigError, PreflightError

        errors = [
            PreflightError("op_a", "Error one"),
            PreflightError("op_b", "Error two", ["suggestion"]),
        ]
        exc = PipelineConfigError(errors)
        msg = str(exc)
        self.assertIn("[op_a] Error one", msg)
        self.assertIn("[op_b] Error two", msg)
        self.assertIn("did you mean: suggestion?", msg)
        self.assertIn("2 error(s) found", msg)


class TestPipelineRuntimeErrorFormatting(DataJuicerTestCaseBase):
    """Tests for PipelineRuntimeError message formatting."""

    @TEST_TAG("standalone")
    def test_runtime_error_message(self):
        """PipelineRuntimeError should format with 'runtime' prefix."""
        from data_juicer.core.preflight import PipelineRuntimeError, PreflightError

        errors = [PreflightError("mapper_op", "text_key 'content' not found")]
        exc = PipelineRuntimeError(errors)
        msg = str(exc)
        self.assertIn("Pipeline runtime preflight errors detected:", msg)
        self.assertIn("[mapper_op] text_key 'content' not found", msg)
        self.assertIn("1 error(s) found", msg)
        self.assertIn("strict_preflight: false", msg)

    @TEST_TAG("standalone")
    def test_runtime_error_stores_errors(self):
        """PipelineRuntimeError should store the errors list."""
        from data_juicer.core.preflight import PipelineRuntimeError, PreflightError

        errors = [
            PreflightError("op1", "err1"),
            PreflightError("op2", "err2"),
        ]
        exc = PipelineRuntimeError(errors)
        self.assertEqual(len(exc.errors), 2)
        self.assertEqual(exc.errors[0].op_name, "op1")
        self.assertEqual(exc.errors[1].op_name, "op2")


if __name__ == '__main__':
    unittest.main()
