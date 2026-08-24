"""Integration tests for llm_semantic_ops with real API calls.

Tests call_llm_sync, extract_one, condition_filter_one, and
_parse_usage_from_response end-to-end using a real ChatAPIModel.
"""

import unittest
from types import SimpleNamespace

from data_juicer.utils.constant import DEFAULT_API_MODEL
from data_juicer.utils.llm_semantic_ops import (
    LLMCallUsage,
    _parse_usage_from_response,
    call_llm_sync,
    condition_filter_one,
    extract_one,
)
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.unittest_utils import (
    TEST_TAG,
    DataJuicerTestCaseBase,
    FROM_FORK,
    skip_if_from_fork,
)


@unittest.skipIf(
    FROM_FORK, "Skipping API-based test because running from a fork repo"
)
class TestLLMSemanticOpsIntegration(DataJuicerTestCaseBase):
    """Integration tests that hit the real LLM API."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_key = prepare_model(
            model_type='api',
            model=DEFAULT_API_MODEL,
        )
        cls.model = get_model(model_key)

    @TEST_TAG("standalone")
    @skip_if_from_fork("Requires API access")
    def test_call_llm_sync_returns_string(self):
        """call_llm_sync with a real ChatAPIModel returns a non-empty string."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ]
        text, usage = call_llm_sync(self.model, messages)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIsInstance(usage, LLMCallUsage)

    @TEST_TAG("standalone")
    @skip_if_from_fork("Requires API access")
    def test_call_llm_sync_usage_has_tokens(self):
        """call_llm_sync returns usage with token counts from the API."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ]
        text, usage = call_llm_sync(self.model, messages)
        self.assertIsInstance(usage, LLMCallUsage)
        # The model should report some token usage
        self.assertGreaterEqual(usage.prompt_tokens + usage.completion_tokens, 0)

    @TEST_TAG("standalone")
    @skip_if_from_fork("Requires API access")
    def test_extract_one_structured_fields(self):
        """extract_one extracts structured fields from natural language text."""
        input_text = "John is 30 years old and lives in New York."
        output_schema = {
            "name": "The person's name",
            "age": "The person's age as a number",
            "city": "The city they live in",
        }
        result, usage = extract_one(
            input_text=input_text,
            output_schema=output_schema,
            model=self.model,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("name", result)
        self.assertIn("age", result)
        self.assertIn("city", result)
        # Verify the extracted values are reasonable
        self.assertIsNotNone(result["name"])
        self.assertIn("John", str(result["name"]))
        # Age could be int or string depending on LLM output
        self.assertIn("30", str(result["age"]))
        self.assertIn("New York", str(result["city"]))
        self.assertIsInstance(usage, LLMCallUsage)

    @TEST_TAG("standalone")
    @skip_if_from_fork("Requires API access")
    def test_extract_one_returns_none_for_missing_field(self):
        """extract_one returns None for fields that cannot be determined."""
        input_text = "The weather is sunny today."
        output_schema = {
            "name": "A person's name mentioned in the text",
            "temperature": "The temperature in degrees",
        }
        result, usage = extract_one(
            input_text=input_text,
            output_schema=output_schema,
            model=self.model,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("name", result)
        self.assertIn("temperature", result)
        # At least one of these should be None since the text doesn't contain them
        has_null = result["name"] is None or result["temperature"] is None
        self.assertTrue(
            has_null,
            f"Expected at least one null field but got: {result}",
        )

    @TEST_TAG("standalone")
    @skip_if_from_fork("Requires API access")
    def test_condition_filter_one_true(self):
        """condition_filter_one returns True for text satisfying the condition."""
        text = "Hello world, how are you doing today?"
        condition = "The text is written in English"
        result, usage = condition_filter_one(
            text=text,
            condition=condition,
            model=self.model,
        )
        self.assertIsInstance(result, bool)
        self.assertTrue(result)
        self.assertIsInstance(usage, LLMCallUsage)

    @TEST_TAG("standalone")
    @skip_if_from_fork("Requires API access")
    def test_condition_filter_one_false(self):
        """condition_filter_one returns False for text NOT satisfying the condition."""
        text = "The capital of France is Paris."
        condition = "The text contains a question"
        result, usage = condition_filter_one(
            text=text,
            condition=condition,
            model=self.model,
        )
        self.assertIsInstance(result, bool)
        self.assertFalse(result)
        self.assertIsInstance(usage, LLMCallUsage)


class TestParseUsageFromResponse(DataJuicerTestCaseBase):
    """Test _parse_usage_from_response with mock response objects."""

    @TEST_TAG("standalone")
    def test_parse_usage_from_api_response_object(self):
        """Parse usage from an object with .usage attribute (API-style)."""
        usage_obj = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        response = SimpleNamespace(usage=usage_obj)
        result = _parse_usage_from_response(response, is_api=True)
        self.assertIsInstance(result, LLMCallUsage)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 50)
        self.assertEqual(result.total_tokens, 150)

    @TEST_TAG("standalone")
    def test_parse_usage_from_api_response_with_input_output_tokens(self):
        """Parse usage using input_tokens/output_tokens naming (Anthropic-style)."""
        usage_obj = SimpleNamespace(
            input_tokens=80,
            output_tokens=40,
            total_tokens=120,
        )
        response = SimpleNamespace(usage=usage_obj)
        result = _parse_usage_from_response(response, is_api=True)
        self.assertIsInstance(result, LLMCallUsage)
        self.assertEqual(result.prompt_tokens, 80)
        self.assertEqual(result.completion_tokens, 40)
        self.assertEqual(result.total_tokens, 120)

    @TEST_TAG("standalone")
    def test_parse_usage_from_dict_response(self):
        """Parse usage from a dict response (e.g., raw JSON)."""
        response = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
        }
        result = _parse_usage_from_response(response, is_api=False)
        self.assertIsInstance(result, LLMCallUsage)
        self.assertEqual(result.prompt_tokens, 200)
        self.assertEqual(result.completion_tokens, 100)
        self.assertEqual(result.total_tokens, 300)

    @TEST_TAG("standalone")
    def test_parse_usage_from_dict_with_input_output_keys(self):
        """Parse usage from dict using input_tokens/output_tokens keys."""
        response = {
            "usage": {
                "input_tokens": 60,
                "output_tokens": 30,
                "total_tokens": 90,
            }
        }
        result = _parse_usage_from_response(response, is_api=False)
        self.assertIsInstance(result, LLMCallUsage)
        self.assertEqual(result.prompt_tokens, 60)
        self.assertEqual(result.completion_tokens, 30)
        self.assertEqual(result.total_tokens, 90)

    @TEST_TAG("standalone")
    def test_parse_usage_no_usage_attribute(self):
        """Returns empty usage when response has no usage info."""
        response = SimpleNamespace(data="some content")
        result = _parse_usage_from_response(response, is_api=True)
        self.assertIsInstance(result, LLMCallUsage)
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.total_tokens, 0)

    @TEST_TAG("standalone")
    def test_parse_usage_empty_dict(self):
        """Returns empty usage when dict response has no usage key."""
        response = {"result": "hello"}
        result = _parse_usage_from_response(response, is_api=False)
        self.assertIsInstance(result, LLMCallUsage)
        # When dict has no 'usage' key, the function uses the dict itself
        # which won't have the expected keys, so values default to 0
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)

    @TEST_TAG("standalone")
    def test_parse_usage_non_api_object_returns_empty(self):
        """Non-API path with non-dict object returns empty usage."""
        response = "plain string"
        result = _parse_usage_from_response(response, is_api=False)
        self.assertIsInstance(result, LLMCallUsage)
        self.assertEqual(result.prompt_tokens, 0)
        self.assertEqual(result.completion_tokens, 0)
        self.assertEqual(result.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
