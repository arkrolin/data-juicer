import inspect
import os
import unittest
from unittest.mock import patch

from data_juicer.utils.model_utils import (
    filter_arguments,
    get_backup_model_link,
    _is_dashscope_openai_compatible_base,
    _maybe_remap_model_for_dashscope,
    _merge_openai_compatible_env_into_model_params,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FilterArgumentsTest(DataJuicerTestCaseBase):
    """Tests for filter_arguments function."""

    def test_filters_to_matching_params(self):
        def func(a, b, c):
            pass

        result = filter_arguments(func, {'a': 1, 'b': 2, 'c': 3, 'd': 4})
        self.assertEqual(result, {'a': 1, 'b': 2, 'c': 3})

    def test_returns_subset_when_not_all_params_provided(self):
        def func(a, b, c):
            pass

        result = filter_arguments(func, {'a': 1, 'c': 3})
        self.assertEqual(result, {'a': 1, 'c': 3})

    def test_returns_empty_when_no_match(self):
        def func(x, y):
            pass

        result = filter_arguments(func, {'a': 1, 'b': 2})
        self.assertEqual(result, {})

    def test_returns_all_args_when_var_keyword_present(self):
        def func(a, **kwargs):
            pass

        args = {'a': 1, 'b': 2, 'c': 3, 'extra': 'value'}
        result = filter_arguments(func, args)
        self.assertEqual(result, args)

    def test_skips_self_parameter(self):
        class MyClass:
            def method(self, a, b):
                pass

        result = filter_arguments(
            MyClass.method, {'self': 'ignored', 'a': 1, 'b': 2, 'c': 3}
        )
        self.assertEqual(result, {'a': 1, 'b': 2})

    def test_skips_cls_parameter(self):
        class MyClass:
            @classmethod
            def method(cls, a, b):
                pass

        result = filter_arguments(
            MyClass.method, {'cls': 'ignored', 'a': 1, 'b': 2, 'c': 3}
        )
        self.assertEqual(result, {'a': 1, 'b': 2})

    def test_empty_args_dict(self):
        def func(a, b, c):
            pass

        result = filter_arguments(func, {})
        self.assertEqual(result, {})

    def test_function_with_defaults(self):
        def func(a, b=10, c=20):
            pass

        result = filter_arguments(func, {'a': 1, 'b': 2, 'extra': 99})
        self.assertEqual(result, {'a': 1, 'b': 2})

    def test_function_with_var_positional(self):
        def func(a, *args, b):
            pass

        result = filter_arguments(func, {'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(result, {'a': 1, 'b': 2})


class MergeOpenAICompatibleEnvTest(DataJuicerTestCaseBase):
    """Tests for _merge_openai_compatible_env_into_model_params."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_no_params(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result, {})

    @patch.dict(os.environ, {}, clear=True)
    def test_none_input(self):
        result = _merge_openai_compatible_env_into_model_params(None)
        self.assertEqual(result, {})

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'}, clear=True)
    def test_api_key_from_openai_env(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['api_key'], 'sk-test123')

    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'ds-key'}, clear=True)
    def test_api_key_from_dashscope_env(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['api_key'], 'ds-key')

    @patch.dict(os.environ, {'SK': 'sk-fallback'}, clear=True)
    def test_api_key_from_sk_env(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['api_key'], 'sk-fallback')

    @patch.dict(
        os.environ,
        {'OPENAI_API_KEY': 'openai-key', 'DASHSCOPE_API_KEY': 'ds-key'},
        clear=True,
    )
    def test_openai_key_takes_priority_over_dashscope(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['api_key'], 'openai-key')

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}, clear=True)
    def test_explicit_api_key_takes_precedence(self):
        result = _merge_openai_compatible_env_into_model_params(
            {'api_key': 'explicit-key'}
        )
        self.assertEqual(result['api_key'], 'explicit-key')

    @patch.dict(
        os.environ, {'OPENAI_BASE_URL': 'http://example.com/v1/'}, clear=True
    )
    def test_base_url_from_openai_env(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['base_url'], 'http://example.com/v1')

    @patch.dict(
        os.environ, {'OPENAI_API_URL': 'http://api.example.com/v1/'}, clear=True
    )
    def test_base_url_from_openai_api_url_env(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['base_url'], 'http://api.example.com/v1')

    @patch.dict(
        os.environ,
        {'DASHSCOPE_BASE_URL': 'https://dashscope.aliyuncs.com/'},
        clear=True,
    )
    def test_base_url_from_dashscope_env(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(
            result['base_url'], 'https://dashscope.aliyuncs.com'
        )

    @patch.dict(
        os.environ,
        {
            'OPENAI_BASE_URL': 'http://first.com/v1/',
            'DASHSCOPE_BASE_URL': 'http://second.com/',
        },
        clear=True,
    )
    def test_openai_base_url_priority_over_dashscope(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['base_url'], 'http://first.com/v1')

    @patch.dict(
        os.environ, {'OPENAI_BASE_URL': 'http://env.com/v1'}, clear=True
    )
    def test_explicit_base_url_takes_precedence(self):
        result = _merge_openai_compatible_env_into_model_params(
            {'base_url': 'http://explicit.com/v1'}
        )
        self.assertEqual(result['base_url'], 'http://explicit.com/v1')

    @patch.dict(
        os.environ, {'OPENAI_BASE_URL': 'http://example.com/v1'}, clear=True
    )
    def test_no_trailing_slash_left_as_is(self):
        result = _merge_openai_compatible_env_into_model_params({})
        self.assertEqual(result['base_url'], 'http://example.com/v1')

    @patch.dict(os.environ, {}, clear=True)
    def test_preserves_other_params(self):
        result = _merge_openai_compatible_env_into_model_params(
            {'model': 'gpt-4', 'temperature': 0.7}
        )
        self.assertEqual(result['model'], 'gpt-4')
        self.assertEqual(result['temperature'], 0.7)


class IsDashscopeOpenAICompatibleBaseTest(DataJuicerTestCaseBase):
    """Tests for _is_dashscope_openai_compatible_base."""

    def test_none_returns_false(self):
        self.assertFalse(_is_dashscope_openai_compatible_base(None))

    def test_empty_string_returns_false(self):
        self.assertFalse(_is_dashscope_openai_compatible_base(''))

    def test_dashscope_in_url(self):
        self.assertTrue(
            _is_dashscope_openai_compatible_base(
                'https://dashscope.aliyuncs.com/compatible-mode/v1'
            )
        )

    def test_dashscope_case_insensitive(self):
        self.assertTrue(
            _is_dashscope_openai_compatible_base(
                'https://DASHSCOPE.aliyuncs.com/v1'
            )
        )

    def test_compatible_mode_and_aliyun(self):
        self.assertTrue(
            _is_dashscope_openai_compatible_base(
                'https://api.aliyun.com/compatible-mode/v1'
            )
        )

    def test_compatible_mode_without_aliyun_returns_false(self):
        self.assertFalse(
            _is_dashscope_openai_compatible_base(
                'https://api.example.com/compatible-mode/v1'
            )
        )

    def test_aliyun_without_compatible_mode_returns_false(self):
        self.assertFalse(
            _is_dashscope_openai_compatible_base(
                'https://api.aliyun.com/v1'
            )
        )

    def test_openai_url_returns_false(self):
        self.assertFalse(
            _is_dashscope_openai_compatible_base(
                'https://api.openai.com/v1'
            )
        )

    def test_random_url_returns_false(self):
        self.assertFalse(
            _is_dashscope_openai_compatible_base(
                'http://localhost:8080/v1'
            )
        )


class MaybeRemapModelForDashscopeTest(DataJuicerTestCaseBase):
    """Tests for _maybe_remap_model_for_dashscope."""

    def test_embedding_endpoint_skips_remap(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/embeddings',
        )
        self.assertEqual(result, 'gpt-4o')

    def test_embedding_endpoint_case_insensitive(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/Embeddings',
        )
        self.assertEqual(result, 'gpt-4o')

    def test_none_model_returns_none(self):
        result = _maybe_remap_model_for_dashscope(
            None, 'https://dashscope.aliyuncs.com/v1', '/chat/completions'
        )
        self.assertIsNone(result)

    def test_non_dashscope_url_returns_model_unchanged(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o', 'https://api.openai.com/v1', '/chat/completions'
        )
        self.assertEqual(result, 'gpt-4o')

    def test_qwen_model_not_remapped(self):
        result = _maybe_remap_model_for_dashscope(
            'qwen-plus',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'qwen-plus')

    def test_qwen_model_case_insensitive(self):
        result = _maybe_remap_model_for_dashscope(
            'Qwen-Max',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'Qwen-Max')

    def test_deepseek_model_not_remapped(self):
        result = _maybe_remap_model_for_dashscope(
            'deepseek-chat',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'deepseek-chat')

    @patch.dict(os.environ, {'DASHSCOPE_DEFAULT_MODEL': 'qwen-max'}, clear=True)
    def test_remaps_using_dashscope_default_model_env(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'qwen-max')

    @patch.dict(os.environ, {'OPENAI_DEFAULT_MODEL': 'qwen-turbo'}, clear=True)
    def test_remaps_using_openai_default_model_env(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'qwen-turbo')

    @patch.dict(
        os.environ,
        {'DASHSCOPE_DEFAULT_MODEL': 'qwen-max', 'OPENAI_DEFAULT_MODEL': 'qwen-turbo'},
        clear=True,
    )
    def test_dashscope_default_model_takes_priority(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'qwen-max')

    @patch.dict(os.environ, {}, clear=True)
    def test_falls_back_to_qwen_plus(self):
        result = _maybe_remap_model_for_dashscope(
            'gpt-4o',
            'https://dashscope.aliyuncs.com/compatible-mode/v1',
            '/chat/completions',
        )
        self.assertEqual(result, 'qwen-plus')

    def test_none_endpoint_treated_as_chat(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _maybe_remap_model_for_dashscope(
                'gpt-4o',
                'https://dashscope.aliyuncs.com/compatible-mode/v1',
                None,
            )
            self.assertEqual(result, 'qwen-plus')


class GetBackupModelLinkTest(DataJuicerTestCaseBase):
    """Tests for get_backup_model_link."""

    def test_exact_match_lid(self):
        result = get_backup_model_link('lid.176.bin')
        self.assertIsNotNone(result)
        self.assertIn('fasttext', result)

    def test_glob_match_sp_model(self):
        result = get_backup_model_link('en.sp.model')
        self.assertIsNotNone(result)
        self.assertIn('kenlm', result)

    def test_glob_match_arpa_bin(self):
        result = get_backup_model_link('en.arpa.bin')
        self.assertIsNotNone(result)
        self.assertIn('kenlm', result)

    def test_glob_match_punkt_pickle(self):
        result = get_backup_model_link('punkt.english.pickle')
        self.assertIsNotNone(result)

    def test_exact_match_fastsam(self):
        result = get_backup_model_link('FastSAM-x.pt')
        self.assertIsNotNone(result)
        self.assertIn('ultralytics', result)

    def test_no_match_returns_none(self):
        result = get_backup_model_link('nonexistent_model.bin')
        self.assertIsNone(result)

    def test_partial_name_no_match(self):
        result = get_backup_model_link('lid.176')
        self.assertIsNone(result)

    def test_yolo_model(self):
        result = get_backup_model_link('yolo11n.pt')
        self.assertIsNotNone(result)
        self.assertIn('yolo11n.pt', result)

    def test_spacy_glob_match(self):
        result = get_backup_model_link('en_core_web_md-3.7.0')
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
