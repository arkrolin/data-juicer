import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestDialogQualityLLMMapperBase(DataJuicerTestCaseBase):

    def _make_mapper(self, **kwargs):
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.prepare_model',
            return_value='mock_key',
        ):
            from data_juicer.ops.mapper.dialog_quality_llm_base import (
                _DialogTurnQualityMapper,
            )

            class _TestMapper(_DialogTurnQualityMapper):
                OP_NAME = 'test_mapper'
                META_KEY = 'test_quality'

                def _system_prompt(self):
                    return 'You are a test evaluator.'

            return _TestMapper(api_model='fake-model', **kwargs)

    def _sample(self, meta=None):
        s = {
            'dialog_history': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi there!'},
            ],
            'query': 'Hello',
            'response': 'Hi there!',
        }
        if meta is not None:
            s[Fields.meta] = meta
        return s

    def test_skip_if_meta_exists_and_no_overwrite(self):
        mapper = self._make_mapper(overwrite=False)
        sample = self._sample(meta={'test_quality': {'score': 4}})
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model'
        ) as mock_get:
            result = mapper.process_single(sample)
        mock_get.assert_not_called()
        self.assertEqual(result[Fields.meta]['test_quality']['score'], 4)

    def test_overwrite_calls_llm(self):
        mapper = self._make_mapper(overwrite=True)
        sample = self._sample(meta={'test_quality': {'score': 3}})
        mock_client = MagicMock(
            return_value='{"score": 5, "reason": "excellent"}'
        )
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        self.assertEqual(result[Fields.meta]['test_quality']['score'], 5)

    def test_empty_user_content_skips(self):
        mapper = self._make_mapper()
        sample = self._sample()
        sample['dialog_history'] = []
        sample['query'] = ''
        sample['response'] = ''
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=MagicMock(),
        ):
            result = mapper.process_single(sample)
        meta = result[Fields.meta]['test_quality']
        self.assertTrue(meta.get('skipped'))

    def test_empty_llm_response_gives_error(self):
        mapper = self._make_mapper()
        sample = self._sample()
        mock_client = MagicMock(return_value='')
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        meta = result[Fields.meta]['test_quality']
        self.assertEqual(meta['error'], 'empty_llm_response')

    def test_json_parse_failure(self):
        mapper = self._make_mapper()
        sample = self._sample()
        mock_client = MagicMock(return_value='not json at all {{{')
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        meta = result[Fields.meta]['test_quality']
        self.assertEqual(meta['error'], 'json_parse_failed')
        self.assertIn('raw', meta)

    def test_successful_score_extraction(self):
        mapper = self._make_mapper()
        sample = self._sample()
        mock_client = MagicMock(
            return_value='{"score": 4, "reason": "good dialog"}'
        )
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        meta = result[Fields.meta]['test_quality']
        self.assertEqual(meta['score'], 4)
        self.assertEqual(meta['eval_kind'], 'dialog_turn')

    def test_retry_on_exception(self):
        mapper = self._make_mapper(try_num=3)
        sample = self._sample()
        call_count = [0]

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError('timeout')
            return '{"score": 3, "reason": "ok"}'

        mock_client = MagicMock(side_effect=side_effect)
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        meta = result[Fields.meta]['test_quality']
        self.assertEqual(meta['score'], 3)
        self.assertEqual(call_count[0], 3)

    def test_all_retries_fail_gives_empty_response_error(self):
        mapper = self._make_mapper(try_num=2)
        sample = self._sample()
        mock_client = MagicMock(side_effect=RuntimeError('fail'))
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        meta = result[Fields.meta]['test_quality']
        self.assertEqual(meta['error'], 'empty_llm_response')

    def test_json_instruction_en(self):
        mapper = self._make_mapper(preferred_output_lang='en')
        inst = mapper._json_instruction()
        self.assertIn('JSON', inst)

    def test_json_instruction_zh(self):
        mapper = self._make_mapper(preferred_output_lang='zh')
        inst = mapper._json_instruction()
        self.assertIn('JSON', inst)

    def test_meta_initialized_if_missing(self):
        mapper = self._make_mapper()
        sample = {'dialog_history': [{'role': 'user', 'content': 'hi'},
                                     {'role': 'assistant', 'content': 'hey'}],
                  'query': 'hi', 'response': 'hey'}
        self.assertNotIn(Fields.meta, sample)
        mock_client = MagicMock(return_value='{"score":2,"reason":"low"}')
        with patch(
            'data_juicer.ops.mapper.dialog_quality_llm_base.get_model',
            return_value=mock_client,
        ):
            result = mapper.process_single(sample)
        self.assertIn(Fields.meta, result)
        self.assertIn('test_quality', result[Fields.meta])


if __name__ == '__main__':
    unittest.main()
