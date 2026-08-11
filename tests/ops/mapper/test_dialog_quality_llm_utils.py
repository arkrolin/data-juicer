# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.dialog_quality_llm_utils import (
    _normalize_dialog_tail,
    build_agent_tool_fit_user_content,
    build_agent_trace_eval_user_content,
    build_dialog_turn_eval_user_content,
    extract_json_object,
    normalize_score_1_5,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExtractJsonObjectTest(DataJuicerTestCaseBase):

    def test_plain_json_string(self):
        result = extract_json_object('{"score": 3, "reason": "ok"}')
        self.assertEqual(result, {"score": 3, "reason": "ok"})

    def test_with_markdown_fence(self):
        text = '```json\n{"score": 4}\n```'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 4})

    def test_with_markdown_fence_no_json_tag(self):
        text = '```\n{"score": 2, "reason": "low"}\n```'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 2, "reason": "low"})

    def test_text_before_and_after(self):
        text = 'The result is {"score": 5} end'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 5})

    def test_none_input(self):
        result = extract_json_object(None)
        self.assertIsNone(result)

    def test_empty_string(self):
        result = extract_json_object("")
        self.assertIsNone(result)

    def test_no_json_content(self):
        result = extract_json_object("no json here at all")
        self.assertIsNone(result)

    def test_integer_input(self):
        result = extract_json_object(123)
        self.assertIsNone(result)

    def test_nested_braces(self):
        text = '{"outer": {"inner": 1}, "val": 2}'
        result = extract_json_object(text)
        self.assertEqual(result, {"outer": {"inner": 1}, "val": 2})

    def test_deeply_nested(self):
        text = '{"a": {"b": {"c": 3}}}'
        result = extract_json_object(text)
        self.assertEqual(result, {"a": {"b": {"c": 3}}})

    def test_with_leading_whitespace(self):
        text = '   \n  {"score": 1}\n  '
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 1})

    def test_invalid_json_with_braces(self):
        text = '{not valid json}'
        result = extract_json_object(text)
        self.assertIsNone(result)

    def test_only_opening_brace(self):
        text = '{ incomplete'
        result = extract_json_object(text)
        self.assertIsNone(result)

    def test_brace_order_reversed(self):
        # } before { - no valid range
        text = '} before {'
        result = extract_json_object(text)
        self.assertIsNone(result)

    def test_json_array_not_extracted(self):
        # extract_json_object only finds objects, not arrays
        text = '[1, 2, 3]'
        result = extract_json_object(text)
        self.assertIsNone(result)

    def test_json_with_string_values_containing_braces(self):
        text = '{"reason": "the code {x} is bad", "score": 2}'
        result = extract_json_object(text)
        self.assertEqual(result, {"reason": "the code {x} is bad", "score": 2})

    def test_markdown_fence_case_insensitive(self):
        text = '```JSON\n{"score": 3}\n```'
        result = extract_json_object(text)
        self.assertEqual(result, {"score": 3})


class NormalizeScore15Test(DataJuicerTestCaseBase):

    def test_valid_score(self):
        result = normalize_score_1_5({"score": 3, "reason": "good"})
        self.assertEqual(result, {"score": 3.0, "reason": "good"})

    def test_score_at_lower_bound(self):
        result = normalize_score_1_5({"score": 1, "reason": "poor"})
        self.assertEqual(result, {"score": 1.0, "reason": "poor"})

    def test_score_at_upper_bound(self):
        result = normalize_score_1_5({"score": 5, "reason": "excellent"})
        self.assertEqual(result, {"score": 5.0, "reason": "excellent"})

    def test_score_below_range_clamps_to_1(self):
        result = normalize_score_1_5({"score": 0})
        self.assertEqual(result["score"], 1.0)

    def test_score_negative_clamps_to_1(self):
        result = normalize_score_1_5({"score": -5})
        self.assertEqual(result["score"], 1.0)

    def test_score_above_range_clamps_to_5(self):
        result = normalize_score_1_5({"score": 10})
        self.assertEqual(result["score"], 5.0)

    def test_score_above_range_large_clamps_to_5(self):
        result = normalize_score_1_5({"score": 100})
        self.assertEqual(result["score"], 5.0)

    def test_float_score_in_range(self):
        result = normalize_score_1_5({"score": 3.7, "reason": "decent"})
        self.assertAlmostEqual(result["score"], 3.7)
        self.assertEqual(result["reason"], "decent")

    def test_string_score_convertible(self):
        result = normalize_score_1_5({"score": "3", "reason": "ok"})
        self.assertEqual(result["score"], 3.0)
        self.assertEqual(result["reason"], "ok")

    def test_string_score_float_convertible(self):
        result = normalize_score_1_5({"score": "4.5"})
        self.assertEqual(result["score"], 4.5)

    def test_non_dict_input_none(self):
        result = normalize_score_1_5(None)
        self.assertEqual(result["error"], "invalid_json")
        self.assertIsNone(result["score"])

    def test_non_dict_input_list(self):
        result = normalize_score_1_5([1, 2, 3])
        self.assertEqual(result["error"], "invalid_json")
        self.assertIsNone(result["score"])

    def test_non_dict_input_string(self):
        result = normalize_score_1_5("not a dict")
        self.assertEqual(result["error"], "invalid_json")

    def test_missing_score_key(self):
        result = normalize_score_1_5({"reason": "no score"})
        self.assertEqual(result["error"], "bad_score")
        self.assertIsNone(result["score"])

    def test_non_numeric_score_value(self):
        result = normalize_score_1_5({"score": "abc", "reason": "bad"})
        self.assertEqual(result["error"], "bad_score")
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "bad")

    def test_score_none_value(self):
        result = normalize_score_1_5({"score": None, "reason": "missing"})
        self.assertEqual(result["error"], "bad_score")
        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "missing")

    def test_reason_truncated_to_2000(self):
        long_reason = "x" * 3000
        result = normalize_score_1_5({"score": 3, "reason": long_reason})
        self.assertEqual(result["score"], 3.0)
        self.assertEqual(len(result["reason"]), 2000)

    def test_missing_reason_defaults_to_empty(self):
        result = normalize_score_1_5({"score": 4})
        self.assertEqual(result["score"], 4.0)
        self.assertEqual(result["reason"], "")

    def test_reason_none_defaults_to_empty(self):
        result = normalize_score_1_5({"score": 2, "reason": None})
        self.assertEqual(result["score"], 2.0)
        self.assertEqual(result["reason"], "")


class NormalizeDialogTailTest(DataJuicerTestCaseBase):

    def test_basic_tail(self):
        sample = {
            "history": [("u1", "a1"), ("u2", "a2"), ("u3", "a3")],
            "query": "u3",
            "response": "a3",
        }
        result = _normalize_dialog_tail(sample, "history", "query", "response", 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1], ("u3", "a3"))

    def test_max_round_zero_returns_all(self):
        sample = {
            "history": [("u1", "a1"), ("u2", "a2"), ("u3", "a3")],
            "query": "u3",
            "response": "a3",
        }
        result = _normalize_dialog_tail(sample, "history", "query", "response", 0)
        self.assertEqual(len(result), 3)

    def test_max_round_larger_than_dialog(self):
        sample = {
            "history": [("u1", "a1")],
            "query": "u1",
            "response": "a1",
        }
        result = _normalize_dialog_tail(sample, "history", "query", "response", 10)
        self.assertEqual(len(result), 1)

    def test_empty_dialog(self):
        sample = {"history": [], "query": "", "response": ""}
        result = _normalize_dialog_tail(sample, "history", "query", "response", 5)
        self.assertEqual(result, [])

    def test_max_round_one(self):
        sample = {
            "history": [("u1", "a1"), ("u2", "a2")],
            "query": "u2",
            "response": "a2",
        }
        result = _normalize_dialog_tail(sample, "history", "query", "response", 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("u2", "a2"))


class BuildDialogTurnEvalUserContentTest(DataJuicerTestCaseBase):

    def test_single_turn(self):
        sample = {
            "history": [],
            "query": "Hello",
            "response": "Hi there",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=5,
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("Hello", result)
        self.assertIn("Hi there", result)
        self.assertIn("Current user message", result)
        self.assertIn("Assistant reply to score", result)

    def test_multi_turn(self):
        sample = {
            "history": [("u1", "a1"), ("u2", "a2")],
            "query": "u2",
            "response": "a2",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=5,
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("Earlier turns", result)
        self.assertIn("[User]", result)
        self.assertIn("[Assistant]", result)
        self.assertIn("u1", result)
        self.assertIn("a1", result)

    def test_empty_dialog_returns_empty(self):
        sample = {"history": [], "query": "", "response": ""}
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=5,
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertEqual(result, "")

    def test_truncation_of_query(self):
        sample = {
            "history": [],
            "query": "x" * 500,
            "response": "short",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=5,
            max_query_chars=50,
            max_response_chars=1000,
        )
        # The full 500 chars should not appear
        self.assertNotIn("x" * 500, result)
        self.assertIn("truncated", result)

    def test_max_round_limits_earlier_turns(self):
        sample = {
            "history": [("u1", "a1"), ("u2", "a2"), ("u3", "a3")],
            "query": "u3",
            "response": "a3",
        }
        result = build_dialog_turn_eval_user_content(
            sample,
            history_key="history",
            query_key="query",
            response_key="response",
            max_round=2,
            max_query_chars=1000,
            max_response_chars=1000,
        )
        # max_round=2: only 2 turns total, so 1 earlier + 1 last
        self.assertIn("u2", result)
        self.assertIn("u3", result)
        # u1 should not appear since only last 2 turns are kept
        self.assertNotIn("u1", result)


class BuildAgentTraceEvalUserContentTest(DataJuicerTestCaseBase):

    def test_basic_trace(self):
        sample = {"text": "User asked X. Agent did Y."}
        result = build_agent_trace_eval_user_content(
            sample,
            text_key="text",
            max_chars=1000,
        )
        self.assertIn("Session trace excerpt", result)
        self.assertIn("User asked X", result)

    def test_empty_text_returns_empty(self):
        sample = {"text": ""}
        result = build_agent_trace_eval_user_content(
            sample,
            text_key="text",
            max_chars=1000,
        )
        self.assertEqual(result, "")

    def test_whitespace_only_returns_empty(self):
        sample = {"text": "   \n  "}
        result = build_agent_trace_eval_user_content(
            sample,
            text_key="text",
            max_chars=1000,
        )
        self.assertEqual(result, "")

    def test_missing_key_returns_empty(self):
        sample = {"other_key": "value"}
        result = build_agent_trace_eval_user_content(
            sample,
            text_key="text",
            max_chars=1000,
        )
        self.assertEqual(result, "")

    def test_non_string_value_returns_empty(self):
        sample = {"text": 12345}
        result = build_agent_trace_eval_user_content(
            sample,
            text_key="text",
            max_chars=1000,
        )
        self.assertEqual(result, "")

    def test_truncation(self):
        sample = {"text": "a" * 500}
        result = build_agent_trace_eval_user_content(
            sample,
            text_key="text",
            max_chars=50,
        )
        self.assertNotIn("a" * 500, result)
        self.assertIn("truncated", result)


class BuildAgentToolFitUserContentTest(DataJuicerTestCaseBase):

    def test_basic_tool_fit(self):
        sample = {
            "query": "search for flights",
            "response": "I found 3 flights.",
            Fields.meta: {
                "tool_types": ["search", "booking"],
                "primary_tool": "search",
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("User request", result)
        self.assertIn("search for flights", result)
        self.assertIn("I found 3 flights", result)
        self.assertIn("search, booking", result)
        self.assertIn("search", result)
        self.assertIn("Primary tool", result)

    def test_missing_meta(self):
        sample = {
            "query": "hello",
            "response": "hi",
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("(none)", result)
        self.assertIn("User request", result)

    def test_meta_not_dict(self):
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
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("(none)", result)

    def test_tools_as_string(self):
        sample = {
            "query": "q",
            "response": "r",
            Fields.meta: {
                "tool_types": "single_tool",
                "primary_tool": "single_tool",
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("single_tool", result)

    def test_empty_query_and_response(self):
        sample = {
            "query": "",
            "response": "",
            Fields.meta: {
                "tool_types": [],
                "primary_tool": None,
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=1000,
            max_response_chars=1000,
        )
        self.assertIn("(none)", result)

    def test_truncation_of_query_and_response(self):
        sample = {
            "query": "q" * 500,
            "response": "r" * 500,
            Fields.meta: {
                "tool_types": ["t1"],
                "primary_tool": "t1",
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=50,
            max_response_chars=50,
        )
        self.assertNotIn("q" * 500, result)
        self.assertNotIn("r" * 500, result)
        self.assertIn("truncated", result)

    def test_tools_list_limited_to_40(self):
        tools = [f"tool_{i}" for i in range(60)]
        sample = {
            "query": "q",
            "response": "r",
            Fields.meta: {
                "tool_types": tools,
                "primary_tool": "tool_0",
            },
        }
        result = build_agent_tool_fit_user_content(
            sample,
            query_key="query",
            response_key="response",
            tool_types_key="tool_types",
            primary_tool_key="primary_tool",
            max_query_chars=1000,
            max_response_chars=1000,
        )
        # Only first 40 tools should appear
        self.assertIn("tool_39", result)
        self.assertNotIn("tool_40", result)


if __name__ == "__main__":
    unittest.main()
