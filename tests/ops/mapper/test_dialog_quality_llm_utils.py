# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.dialog_quality_llm_utils import (
    build_agent_tool_fit_user_content,
    build_agent_trace_eval_user_content,
    build_dialog_turn_eval_user_content,
    extract_json_object,
    normalize_score_1_5,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class ExtractJsonObjectTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_valid_json_string(self):
        text = '{"score": 3, "reason": "good"}'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 3, "reason": "good"})

    @TEST_TAG("standalone")
    def test_markdown_code_block(self):
        text = '```json\n{"score": 4, "reason": "great"}\n```'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 4, "reason": "great"})

    @TEST_TAG("standalone")
    def test_markdown_code_block_no_lang(self):
        text = '```\n{"score": 2, "reason": "ok"}\n```'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 2, "reason": "ok"})

    @TEST_TAG("standalone")
    def test_nested_json_with_text_around(self):
        text = 'Here is the result: {"score": 5, "reason": "excellent", "details": {"sub": 1}} end.'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 5, "reason": "excellent", "details": {"sub": 1}})

    @TEST_TAG("standalone")
    def test_none_input(self):
        self.assertIsNone(extract_json_object(None))

    @TEST_TAG("standalone")
    def test_empty_string(self):
        self.assertIsNone(extract_json_object(""))

    @TEST_TAG("standalone")
    def test_no_braces(self):
        self.assertIsNone(extract_json_object("no json here"))

    @TEST_TAG("standalone")
    def test_invalid_json(self):
        text = '{"score": bad_value}'
        self.assertIsNone(extract_json_object(text))

    @TEST_TAG("standalone")
    def test_only_opening_brace(self):
        text = '{"score": 3'
        self.assertIsNone(extract_json_object(text))

    @TEST_TAG("standalone")
    def test_non_string_input(self):
        self.assertIsNone(extract_json_object(123))


class NormalizeScore15Test(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_valid_dict_with_score(self):
        result = normalize_score_1_5({"score": 3, "reason": "good"})
        self.assertEqual(result["score"], 3.0)
        self.assertEqual(result["reason"], "good")
        self.assertNotIn("error", result)

    @TEST_TAG("standalone")
    def test_score_below_1_clamped(self):
        result = normalize_score_1_5({"score": -2, "reason": "low"})
        self.assertEqual(result["score"], 1.0)

    @TEST_TAG("standalone")
    def test_score_above_5_clamped(self):
        result = normalize_score_1_5({"score": 10, "reason": "high"})
        self.assertEqual(result["score"], 5.0)

    @TEST_TAG("standalone")
    def test_score_float_string(self):
        result = normalize_score_1_5({"score": "4.5", "reason": "ok"})
        self.assertEqual(result["score"], 4.5)

    @TEST_TAG("standalone")
    def test_non_numeric_score(self):
        result = normalize_score_1_5({"score": "abc", "reason": "bad"})
        self.assertEqual(result["error"], "bad_score")
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "bad")

    @TEST_TAG("standalone")
    def test_none_input(self):
        result = normalize_score_1_5(None)
        self.assertEqual(result["error"], "invalid_json")
        self.assertIsNone(result["score"])

    @TEST_TAG("standalone")
    def test_non_dict_input(self):
        result = normalize_score_1_5("not a dict")
        self.assertEqual(result["error"], "invalid_json")
        self.assertIsNone(result["score"])

    @TEST_TAG("standalone")
    def test_missing_score_key(self):
        result = normalize_score_1_5({"reason": "no score"})
        self.assertEqual(result["error"], "bad_score")
        self.assertIsNone(result["score"])

    @TEST_TAG("standalone")
    def test_very_long_reason_truncated(self):
        long_reason = "x" * 5000
        result = normalize_score_1_5({"score": 3, "reason": long_reason})
        self.assertEqual(result["score"], 3.0)
        self.assertEqual(len(result["reason"]), 2000)

    @TEST_TAG("standalone")
    def test_reason_none_becomes_empty_string(self):
        result = normalize_score_1_5({"score": 2, "reason": None})
        self.assertEqual(result["reason"], "")


class BuildDialogTurnEvalUserContentTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_with_history(self):
        sample = {
            "history": [("hi", "hello"), ("how are you", "fine")],
            "query": "what is 1+1",
            "response": "2",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=10,
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("Earlier turns", result)
        self.assertIn("[User]", result)
        self.assertIn("[Assistant]", result)
        self.assertIn("what is 1+1", result)
        self.assertIn("2", result)
        self.assertIn("Current user message", result)
        self.assertIn("Assistant reply to score", result)

    @TEST_TAG("standalone")
    def test_empty_turns(self):
        sample = {
            "history": [],
            "query": "",
            "response": "",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=10,
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertEqual(result, "")

    @TEST_TAG("standalone")
    def test_single_turn(self):
        sample = {
            "history": [],
            "query": "hello",
            "response": "world",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=10,
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("Current user message", result)
        self.assertIn("hello", result)
        self.assertIn("world", result)
        # No earlier turns section content for a single turn
        self.assertIn("Earlier turns", result)

    @TEST_TAG("standalone")
    def test_max_round_truncates(self):
        sample = {
            "history": [("a", "b"), ("c", "d"), ("e", "f")],
            "query": "e",
            "response": "f",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=2,
            max_query_chars=0,
            max_response_chars=0,
        )
        # With max_round=2, only last 2 turns are kept
        self.assertNotIn("[User]\na\n", result)
        self.assertIn("e", result)


class BuildAgentTraceEvalUserContentTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_normal_text(self):
        sample = {"text": "This is a session trace."}
        result = build_agent_trace_eval_user_content(
            sample, text_key="text", max_chars=0
        )
        self.assertIn("Session trace excerpt", result)
        self.assertIn("This is a session trace.", result)

    @TEST_TAG("standalone")
    def test_empty_text(self):
        sample = {"text": ""}
        result = build_agent_trace_eval_user_content(
            sample, text_key="text", max_chars=0
        )
        self.assertEqual(result, "")

    @TEST_TAG("standalone")
    def test_non_string(self):
        sample = {"text": 123}
        result = build_agent_trace_eval_user_content(
            sample, text_key="text", max_chars=0
        )
        self.assertEqual(result, "")

    @TEST_TAG("standalone")
    def test_whitespace_only(self):
        sample = {"text": "   \n\t  "}
        result = build_agent_trace_eval_user_content(
            sample, text_key="text", max_chars=0
        )
        self.assertEqual(result, "")

    @TEST_TAG("standalone")
    def test_missing_key(self):
        sample = {"other": "value"}
        result = build_agent_trace_eval_user_content(
            sample, text_key="text", max_chars=0
        )
        self.assertEqual(result, "")


class BuildAgentToolFitUserContentTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_with_tools_list(self):
        sample = {
            "query": "search for files",
            "response": "found 3 files",
            Fields.meta: {
                "tool_types": ["search", "read_file", "write_file"],
                "primary_tool": "search",
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("User request", result)
        self.assertIn("search for files", result)
        self.assertIn("Assistant reply", result)
        self.assertIn("found 3 files", result)
        self.assertIn("search, read_file, write_file", result)
        self.assertIn("Primary tool", result)
        self.assertIn("search", result)

    @TEST_TAG("standalone")
    def test_tools_as_string(self):
        sample = {
            "query": "q",
            "response": "r",
            Fields.meta: {
                "tool_types": "single_tool",
                "primary_tool": None,
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("single_tool", result)
        self.assertIn("(none)", result)  # primary is None

    @TEST_TAG("standalone")
    def test_no_tools(self):
        sample = {
            "query": "q",
            "response": "r",
            Fields.meta: {},
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("(none)", result)

    @TEST_TAG("standalone")
    def test_meta_not_a_dict(self):
        sample = {
            "query": "q",
            "response": "r",
            Fields.meta: "not a dict",
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("(none)", result)
        self.assertIn("User request", result)

    @TEST_TAG("standalone")
    def test_no_meta_key(self):
        sample = {
            "query": "hello",
            "response": "world",
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=0,
            max_response_chars=0,
        )
        self.assertIn("(none)", result)
        self.assertIn("hello", result)


if __name__ == "__main__":
    unittest.main()
