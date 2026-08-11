# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.dialog_llm_input_utils import (
    build_dialog_turns_for_prompt,
    clip_query_response_pair,
    clip_text_for_dialog_prompt,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestBuildDialogTurnsForPrompt(DataJuicerTestCaseBase):

    def test_empty_sample_no_history_no_query(self):
        sample = {}
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [])

    def test_history_only_no_query(self):
        sample = {
            "dialog_history": [["u1", "a1"], ["u2", "a2"]],
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1"), ("u2", "a2")])

    def test_query_response_only_no_history(self):
        sample = {
            "query": "hello",
            "response": "world",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("hello", "world")])

    def test_dedupes_last_turn_when_same_as_query_response(self):
        sample = {
            "dialog_history": [("u1", "a1"), ("u2", "a2")],
            "query": "u2",
            "response": "a2",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1"), ("u2", "a2")])

    def test_appends_when_query_differs_from_last_user(self):
        sample = {
            "dialog_history": [("u1", "a1")],
            "query": "u2",
            "response": "a2",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1"), ("u2", "a2")])

    def test_updates_response_when_last_user_matches_query_but_response_differs(self):
        sample = {
            "dialog_history": [("u1", "a1"), ("u2", "old_response")],
            "query": "u2",
            "response": "new_response",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1"), ("u2", "new_response")])

    def test_non_list_history_is_ignored(self):
        sample = {
            "dialog_history": "not a list",
            "query": "q",
            "response": "r",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("q", "r")])

    def test_none_values_in_history_tuples_converted_to_empty_string(self):
        sample = {
            "dialog_history": [[None, "a1"], ["u2", None]],
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("", "a1"), ("u2", "")])

    def test_short_tuples_in_history_are_skipped(self):
        sample = {
            "dialog_history": [["only_one"], ["u1", "a1"]],
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("u1", "a1")])

    def test_does_not_mutate_sample_dialog_history(self):
        hist = [("u1", "a1")]
        sample = {
            "dialog_history": hist,
            "query": "q",
            "response": "r",
        }
        build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(hist, [("u1", "a1")])

    def test_query_with_no_response_key_in_sample(self):
        sample = {
            "query": "hello",
        }
        turns = build_dialog_turns_for_prompt(
            sample,
            history_key="dialog_history",
            query_key="query",
            response_key="response",
        )
        self.assertEqual(turns, [("hello", "")])


class TestClipTextForDialogPrompt(DataJuicerTestCaseBase):

    def test_text_within_limit_unchanged(self):
        text = "short text"
        result = clip_text_for_dialog_prompt(text, 100)
        self.assertEqual(result, "short text")

    def test_text_exceeding_limit_truncated_with_suffix(self):
        text = "a" * 100
        result = clip_text_for_dialog_prompt(text, 30)
        self.assertIn("truncated", result)
        self.assertTrue(len(result) <= 30 or result.endswith("…"))
        self.assertTrue(result.startswith("a"))

    def test_max_chars_zero_returns_unchanged(self):
        text = "hello world"
        result = clip_text_for_dialog_prompt(text, 0)
        self.assertEqual(result, "hello world")

    def test_max_chars_none_returns_unchanged(self):
        text = "hello world"
        result = clip_text_for_dialog_prompt(text, None)
        self.assertEqual(result, "hello world")

    def test_empty_text_returns_unchanged(self):
        result = clip_text_for_dialog_prompt("", 10)
        self.assertEqual(result, "")

    def test_exact_length_text_returns_unchanged(self):
        text = "abcde"
        result = clip_text_for_dialog_prompt(text, 5)
        self.assertEqual(result, "abcde")

    def test_custom_note_appears_in_suffix(self):
        text = "a" * 100
        result = clip_text_for_dialog_prompt(text, 30, note="cut")
        self.assertIn("cut", result)

    def test_very_small_max_chars_returns_suffix_only(self):
        text = "a" * 100
        result = clip_text_for_dialog_prompt(text, 3)
        # When take <= 0, returns suffix.strip()
        self.assertIn("truncated", result)


class TestClipQueryResponsePair(DataJuicerTestCaseBase):

    def test_both_within_limits(self):
        q, r = clip_query_response_pair("short", "text", 100, 100)
        self.assertEqual(q, "short")
        self.assertEqual(r, "text")

    def test_both_exceeding_limits(self):
        q, r = clip_query_response_pair("a" * 100, "b" * 100, 20, 20)
        self.assertIn("query truncated", q)
        self.assertIn("response truncated", r)

    def test_none_inputs_converted_to_empty_string(self):
        q, r = clip_query_response_pair(None, None, 100, 100)
        self.assertEqual(q, "")
        self.assertEqual(r, "")

    def test_non_string_inputs_converted(self):
        q, r = clip_query_response_pair(123, 456, 100, 100)
        self.assertEqual(q, "123")
        self.assertEqual(r, "456")


if __name__ == "__main__":
    unittest.main()
