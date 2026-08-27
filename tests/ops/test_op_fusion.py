import unittest

from data_juicer.core import NestedDataset
from data_juicer.ops.base_op import Mapper, OP
from data_juicer.ops.load import load_ops
from data_juicer.ops.op_fusion import (
    _are_ops_independent,
    _estimated_vram_fraction,
    _is_fusible_gpu_mapper,
    _is_gpu_mapper,
    _mapper_group_blocker,
    _runtime_envs_compatible,
    fuse_consecutive_mappers,
    fuse_mapper_group,
    fuse_operators,
    GeneralFusedOP,
    MAPPER_FUSION_SAFE_ATTR,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from unittest.mock import MagicMock, patch


class OpFusionTest(DataJuicerTestCaseBase):

    def _run_equal_config(self, original_process_list):
        dataset = NestedDataset.from_list([
            {'text': 'This is a test.'},
            {'text': 'This is a test. This is a test. This is a test.'},
            {'text': 'aaaaaaaaaaaaaaabbbbbbbbbbbbcccccccccccccc'},
            {'text': 'punc test。'}
        ])
        unfused_op = load_ops(original_process_list)
        fused_ops = fuse_operators(unfused_op)
        res1 = dataset.process(fused_ops)
        res2 = dataset.process(unfused_op)
        self.assertDatasetEqual(res1, res2)

    def _run_op_fusion(self, original_process_list, target_process_list, probe_res=None):
        ops = load_ops(original_process_list)
        ops = fuse_operators(ops, probe_res)
        new_process_list = [op._op_cfg for op in ops]
        self.assertEqual(new_process_list, target_process_list)

    def test_regular_config(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process)
        self._run_equal_config(original_process)

    def test_only_mapper(self):
        original_process = [{
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_only_deduplicator(self):

        original_process = [{
            'document_deduplicator': {
                'ignore_non_character': True,
                'lowercase': True,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }, {
            'document_minhash_deduplicator': {
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6,
                'num_permutations': 256,
                'jaccard_threshold': 0.7
            }
        }]
        target_process = [{
            'document_deduplicator': {
                'ignore_non_character': True,
                'lowercase': True,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }, {
            'document_minhash_deduplicator': {
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6,
                'num_permutations': 256,
                'jaccard_threshold': 0.7
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_non_fusible_filters(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'alphanumeric_filter': {
                'min_ratio': 0.25,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'alphanumeric_filter': {
                'min_ratio': 0.25,
                'text_key': 'text'
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_not_enough_fusible_ops_to_fuse(self):
        # still apply reordering:
        # - ordinary ops
        # - ops with InterVars.lines
        # - ops with InterVars.words
        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_multiple_groups(self):

        original_process = [{
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(stopwords_filter,flagged_words_filter)': [{
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }, {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }]
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process)

    def test_only_fusible_ops(self):

        original_process = [{
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
            [
                {
                    'words_num_filter': {
                        'lang': 'en',
                        'max_num': 100000,
                        'min_num': 20,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'word_repetition_filter': {
                        'lang': 'en',
                        'max_ratio': 0.19,
                        'min_ratio': 0.0,
                        'rep_len': 5,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'perplexity_filter': {
                        'lang': 'en',
                        'max_ppl': 1500,
                        'text_key': 'text'
                    }
                }
            ]
        }]
        self._run_op_fusion(original_process, target_process)

    def test_different_intermediate_vars(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'maximum_line_length_filter': {
                'min_len': 20,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(average_line_length_filter,maximum_line_length_filter)':  # noqa: E501
                [
                    {
                        'average_line_length_filter': {
                            'min_len': 10,
                            'text_key': 'text',
                        }
                    },
                    {
                        'maximum_line_length_filter': {
                            'min_len': 20,
                            'text_key': 'text',
                        }
                    }
                ]
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process)

    def test_regular_config_with_probe_res(self):
        probed_speeds = [
            # single filter
            {'speed': 100},

            # mappers
            {'speed': 2},
            {'speed': 1},
            {'speed': 4},
            {'speed': 5},
            {'speed': 3},

            # filter groups
            # fused OPs: ~2.56
            # single OP 1: 1 (slowest)
            # single OP 2: 3 (fastest)
            {'speed': 15},  # fusible
            {'speed': 1},
            {'speed': 14},  # fusible
            {'speed': 3},
            {'speed': 13},  # fusible
            {'speed': 12},  # fusible
            {'speed': 11},  # fusible

            # deduplicator
            {'speed': 0.1},
        ]

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process, probed_speeds)

    def test_not_enough_fusible_ops_to_fuse_with_probe_res(self):
        # still apply reordering:
        # - ordinary ops
        # - ops with InterVars.lines
        # - ops with InterVars.words
        probe_res_list = [
            {'speed': 3},
            {'speed': 1},
            {'speed': 4},
            {'speed': 2},
        ]

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }]
        self._run_op_fusion(original_process, target_process, probe_res_list)

    def test_multiple_groups_with_probe_res(self):
        probe_res_list = [
            # group 1
            # fused filter will be put before the single filter
            {'speed': 10},
            {'speed': 10},
            {'speed': 1},

            # mappers
            {'speed': 4},
            {'speed': 2},
            {'speed': 5},
            {'speed': 3},
            {'speed': 1},

            # group 2
            # fused filter will be put after those two single filters
            {'speed': 1},  # fusible
            {'speed': 8},
            {'speed': 1},  # fusible
            {'speed': 10},
            {'speed': 1},  # fusible

            # deduplicator
            {'speed': 1},
        ]

        original_process = [{
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'OpFusion:(stopwords_filter,flagged_words_filter)': [{
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }, {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                }]
            },
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process, probe_res_list)

    def test_only_fusible_ops_with_probe_res(self):
        probe_res_list = [
            {'speed': 1},
            {'speed': 1},
            {'speed': 1},
            {'speed': 1},
            {'speed': 1},
        ]

        original_process = [{
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
            [
                {
                    'words_num_filter': {
                        'lang': 'en',
                        'max_num': 100000,
                        'min_num': 20,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'word_repetition_filter': {
                        'lang': 'en',
                        'max_ratio': 0.19,
                        'min_ratio': 0.0,
                        'rep_len': 5,
                        'text_key': 'text',
                        'tokenization': False
                    }
                },
                {
                    'stopwords_filter': {
                        'lang': 'en',
                        'min_ratio': 0.3,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'flagged_words_filter': {
                        'lang': 'en',
                        'max_ratio': 0.01,
                        'text_key': 'text',
                        'tokenization': False,
                        'use_words_aug': False,
                        'words_aug_group_sizes': [2],
                        'words_aug_join_char': ''
                    }
                },
                {
                    'perplexity_filter': {
                        'lang': 'en',
                        'max_ppl': 1500,
                        'text_key': 'text'
                    }
                }
            ]
        }]
        self._run_op_fusion(original_process, target_process, probe_res_list)

    def test_different_intermediate_vars_with_probe_res(self):
        probe_res_list = [
            # single filter
            {'speed': 1},

            # mappers
            {'speed': 5},
            {'speed': 3},
            {'speed': 1},
            {'speed': 2},
            {'speed': 4},

            # filter group
            # single 1: 1 (2)
            # single 2: 0.5 (3)
            # group 1: 0.04 (4)
            # group 2: 1.5 (1)
            {'speed': 0.1},  # group 1
            {'speed': 1},
            {'speed': 3},  # group 2
            {'speed': 0.2},  # group 1
            {'speed': 0.5},
            {'speed': 0.3},  # group 1
            {'speed': 0.4},  # group 1
            {'speed': 3},  # group 2
            {'speed': 0.5},  # group 1

            # deduplicator
            {'speed': 1},
        ]

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'maximum_line_length_filter': {
                'min_len': 20,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [
            {
                'language_id_score_filter': {
                    'lang': 'en',
                    'min_score': 0.8,
                    'text_key': 'text'
                }
            },
            {
                'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'punctuation_normalization_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'fix_unicode_mapper': {
                    'text_key': 'text'
                }
            },
            {
                'remove_words_with_incorrect_substrings_mapper': {
                    'lang': 'en',
                    'substrings': None,
                    'text_key': 'text',
                    'tokenization': False
                }
            },
            {
                'remove_long_words_mapper': {
                    'max_len': 25,
                    'min_len': 1,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(average_line_length_filter,maximum_line_length_filter)':  # noqa: E501
                [
                    {
                        'average_line_length_filter': {
                            'min_len': 10,
                            'text_key': 'text',
                        }
                    },
                    {
                        'maximum_line_length_filter': {
                            'min_len': 20,
                            'text_key': 'text',
                        }
                    }
                ]
            },
            {
                'character_repetition_filter': {
                    'max_ratio': 0.106,
                    'min_ratio': 0.0,
                    'rep_len': 10,
                    'text_key': 'text'
                }
            },
            {
                'special_characters_filter': {
                    'max_ratio': 0.4,
                    'min_ratio': 0.0,
                    'text_key': 'text'
                }
            },
            {
                'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)':  # noqa: E501
                [
                    {
                        'words_num_filter': {
                            'lang': 'en',
                            'max_num': 100000,
                            'min_num': 20,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'word_repetition_filter': {
                            'lang': 'en',
                            'max_ratio': 0.19,
                            'min_ratio': 0.0,
                            'rep_len': 5,
                            'text_key': 'text',
                            'tokenization': False
                        }
                    },
                    {
                        'stopwords_filter': {
                            'lang': 'en',
                            'min_ratio': 0.3,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'flagged_words_filter': {
                            'lang': 'en',
                            'max_ratio': 0.01,
                            'text_key': 'text',
                            'tokenization': False,
                            'use_words_aug': False,
                            'words_aug_group_sizes': [2],
                            'words_aug_join_char': ''
                        }
                    },
                    {
                        'perplexity_filter': {
                            'lang': 'en',
                            'max_ppl': 1500,
                            'text_key': 'text'
                        }
                    }
                ]
            },
            {
                'document_simhash_deduplicator': {
                    'hamming_distance': 4,
                    'ignore_pattern': '\\p{P}',
                    'lowercase': True,
                    'num_blocks': 6,
                    'text_key': 'text',
                    'tokenization': 'space',
                    'window_size': 6
                }
            }
        ]
        self._run_op_fusion(original_process, target_process, probe_res_list)


class GeneralFusedOPTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.raw_data = [
            {'text': 'This is a test.'},
            {'text': 'This is a test. This is a test. This is a test.'},
            {'text': 'aaaaaaaaaaaaaaabbbbbbbbbbbbcccccccccccccc'},
            {'text': 'punc test。'}
        ]

    def _get_fresh_dataset(self):
        """Get a fresh dataset instance to avoid state pollution between tests."""
        return NestedDataset.from_list(self.raw_data)

    def _run_equal_config(self, fused_process, unfused_process):
        fused_op = load_ops(fused_process)
        self.assertEqual(len(fused_op), 1)
        fused_op = fused_op[0]
        unfused_op = load_ops(unfused_process)
        self.assertIsInstance(fused_op, GeneralFusedOP)
        self.assertEqual(len(fused_op.fused_ops), len(unfused_process))
        
        # Use fresh datasets for each operation to avoid state pollution
        dataset1 = self._get_fresh_dataset()
        dataset2 = self._get_fresh_dataset()
        res1 = dataset1.process(fused_op)
        res2 = dataset2.process(unfused_op)
        self.assertDatasetEqual(res1, res2)

    def test_regular_config(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }]
        fused_process = [{
            'general_fused_op': {
                'batch_size': 2,
                'fused_op_list': original_process,
            }
        }]
        self._run_equal_config(fused_process, original_process)

    def test_border_cases(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }]
        empty_fused_process = [{
            'general_fused_op': {
                'batch_size': 2,
                'fused_op_list': None,
            }
        }]
        fused_process = [{
            'general_fused_op': {
                'batch_size': 2,
                'fused_op_list': original_process,
            }
        }]
        # empty fused process
        fused_op = load_ops(empty_fused_process)[0]
        self.assertEqual(len(fused_op.fused_ops), 0)
        dataset = self._get_fresh_dataset()
        res = fused_op.run(dataset)
        self.assertDatasetEqual(res, dataset)
        # unsupported fused op
        dataset2 = self._get_fresh_dataset()
        with self.assertRaises(NotImplementedError):
            fused_op = load_ops([{
                'general_fused_op': {
                    'batch_size': 2,
                    'fused_op_list': [{
                        'document_deduplicator': {}
                    }],
                }
            }])[0]
            fused_op.process_batched(dataset2.to_dict())


class FusedFilterFingerprintTest(DataJuicerTestCaseBase):
    """Tests that FusedFilter fingerprints exclude child OP work_dirs."""

    def test_fused_filter_stable_across_work_dirs(self):
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.filter.words_num_filter import WordsNumFilter
        from data_juicer.ops.op_fusion import FusedFilter
        from data_juicer.utils.fingerprint_utils import Hasher

        f1a = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/a')
        f2a = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/a')
        fused_a = FusedFilter('fused', [f1a, f2a])

        f1b = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/b')
        f2b = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/b')
        fused_b = FusedFilter('fused', [f1b, f2b])

        self.assertEqual(Hasher.hash(fused_a), Hasher.hash(fused_b))

    def test_fused_filter_differs_when_child_params_change(self):
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter
        from data_juicer.ops.filter.words_num_filter import WordsNumFilter
        from data_juicer.ops.op_fusion import FusedFilter
        from data_juicer.utils.fingerprint_utils import Hasher

        f1a = TextLengthFilter(min_len=5, max_len=10000, work_dir='/tmp/a')
        f2a = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/a')
        fused_a = FusedFilter('fused', [f1a, f2a])

        f1b = TextLengthFilter(min_len=50, max_len=10000, work_dir='/tmp/a')
        f2b = WordsNumFilter(min_num=2, max_num=1000, work_dir='/tmp/a')
        fused_b = FusedFilter('fused', [f1b, f2b])

        self.assertNotEqual(Hasher.hash(fused_a), Hasher.hash(fused_b))


class _MockUpperCaseMapper(Mapper):
    """Mapper that uppercases text and returns a NEW dict."""
    _batched_op = True

    def __init__(self, text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self._name = 'mock_upper_case_mapper'

    def process_batched(self, samples, **kwargs):
        new_samples = samples.copy()
        new_samples[self.text_key] = [t.upper() for t in samples[self.text_key]]
        return new_samples


class _MockSuffixMapper(Mapper):
    """Mapper that appends a suffix and returns a NEW dict."""
    _batched_op = True

    def __init__(self, suffix='_DONE', text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self.suffix = suffix
        self._name = 'mock_suffix_mapper'

    def process_batched(self, samples, **kwargs):
        new_samples = samples.copy()
        new_samples[self.text_key] = [t + self.suffix for t in samples[self.text_key]]
        return new_samples


class _MockInPlaceMapper(Mapper):
    """Mapper that mutates in-place (masks the new-dict bug)."""
    _batched_op = True

    def __init__(self, text_key='text', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_key = text_key
        self._name = 'mock_inplace_mapper'

    def process_batched(self, samples, **kwargs):
        samples[self.text_key] = [t.lower() for t in samples[self.text_key]]
        return samples


class GeneralFusedOPMapperBugTest(DataJuicerTestCaseBase):
    """Regression: mapper results must chain correctly in GeneralFusedOP."""

    def _make_fused_op(self, ops):
        fused = GeneralFusedOP.__new__(GeneralFusedOP)
        fused._name = 'GeneralFusedOP:test'
        fused.fused_ops = ops
        fused.accelerator = 'cpu'
        fused.batch_size = 10
        fused.num_proc = 1
        return fused

    def test_two_new_dict_mappers_results_chained(self):
        fused_op = self._make_fused_op([
            _MockUpperCaseMapper(), _MockSuffixMapper(suffix='_DONE')])
        samples = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }
        result = fused_op.process_batched(samples)
        self.assertEqual(result['text'], ['HELLO_DONE', 'WORLD_DONE'])

    def test_single_mapper_result_returned(self):
        fused_op = self._make_fused_op([_MockUpperCaseMapper()])
        samples = {
            'text': ['hello', 'world'],
            Fields.stats: [{}, {}],
        }
        result = fused_op.process_batched(samples)
        self.assertEqual(result['text'], ['HELLO', 'WORLD'])

    def test_inplace_then_newdict_mapper(self):
        fused_op = self._make_fused_op([
            _MockInPlaceMapper(), _MockSuffixMapper(suffix='_END')])
        samples = {
            'text': ['HELLO', 'WORLD'],
            Fields.stats: [{}, {}],
        }
        result = fused_op.process_batched(samples)
        self.assertEqual(result['text'], ['hello_END', 'world_END'])





def _make_filter(name='filter1', speed=None):
    from data_juicer.ops.base_op import Filter
    op = MagicMock(spec=Filter)
    op.__class__ = Filter
    op._name = name
    op._op_cfg = {name: {}}
    op.accelerator = 'cpu'
    op.runtime_np.return_value = 4
    return op


def _make_mapper(name='mapper1', num_gpus=0, fusible=False, vram=None,
                 input_cols=None, output_cols=None, runtime_env=None,
                 num_cpus=None, batch_size=1, num_proc=4):
    from data_juicer.ops.base_op import Mapper
    op = MagicMock(spec=Mapper)
    op.__class__ = Mapper
    op._name = name
    op._op_cfg = {name: {}}
    op.num_gpus = num_gpus
    op.accelerator = 'cuda' if num_gpus else 'cpu'
    op.runtime_np.return_value = num_proc
    op.batch_size = batch_size
    op.num_cpus = num_cpus
    op.runtime_env = runtime_env
    op._input_columns = input_cols or []
    op._output_columns = output_cols or []
    if fusible:
        setattr(op, MAPPER_FUSION_SAFE_ATTR, True)
    else:
        setattr(op, MAPPER_FUSION_SAFE_ATTR, False)
    if vram is not None:
        op.estimated_vram_fraction = vram
    else:
        op.estimated_vram_fraction = None
    return op


class TestIsGpuMapper(unittest.TestCase):

    def test_gpu_mapper(self):
        op = _make_mapper(num_gpus=1)
        self.assertTrue(_is_gpu_mapper(op))

    def test_cpu_mapper(self):
        op = _make_mapper(num_gpus=0)
        self.assertFalse(_is_gpu_mapper(op))

    def test_filter_not_mapper(self):
        op = _make_filter()
        self.assertFalse(_is_gpu_mapper(op))


class TestIsFusibleGpuMapper(unittest.TestCase):

    def test_fusible(self):
        op = _make_mapper(num_gpus=1, fusible=True)
        self.assertTrue(_is_fusible_gpu_mapper(op))

    def test_not_fusible(self):
        op = _make_mapper(num_gpus=1, fusible=False)
        self.assertFalse(_is_fusible_gpu_mapper(op))

    def test_cpu_not_fusible(self):
        op = _make_mapper(num_gpus=0, fusible=True)
        self.assertFalse(_is_fusible_gpu_mapper(op))


class TestEstimatedVramFraction(unittest.TestCase):

    def test_valid_fraction(self):
        op = _make_mapper(vram=0.5)
        self.assertEqual(_estimated_vram_fraction(op), 0.5)

    def test_none_fraction(self):
        op = _make_mapper(vram=None)
        self.assertIsNone(_estimated_vram_fraction(op))

    def test_invalid_zero(self):
        op = _make_mapper(vram=0.0)
        op._name = 'bad_op'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_invalid_greater_than_one(self):
        op = _make_mapper(vram=1.5)
        op._name = 'bad_op'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)

    def test_invalid_type(self):
        op = _make_mapper()
        op.estimated_vram_fraction = "not_a_number"
        op._name = 'bad_op'
        with self.assertRaises(ValueError):
            _estimated_vram_fraction(op)


class TestRuntimeEnvsCompatible(unittest.TestCase):

    def test_empty_list(self):
        self.assertTrue(_runtime_envs_compatible([]))

    def test_same_env(self):
        ops = [_make_mapper(runtime_env={'pip': ['torch']}),
               _make_mapper(runtime_env={'pip': ['torch']})]
        self.assertTrue(_runtime_envs_compatible(ops))

    def test_different_envs(self):
        ops = [_make_mapper(runtime_env={'pip': ['torch']}),
               _make_mapper(runtime_env={'pip': ['jax']})]
        self.assertFalse(_runtime_envs_compatible(ops))

    def test_none_envs(self):
        ops = [_make_mapper(runtime_env=None), _make_mapper(runtime_env=None)]
        self.assertTrue(_runtime_envs_compatible(ops))


class TestAreOpsIndependent(unittest.TestCase):

    def test_disjoint_outputs(self):
        ops = [_make_mapper(output_cols=['col_a']),
               _make_mapper(output_cols=['col_b'])]
        self.assertTrue(_are_ops_independent(ops))

    def test_overlapping_outputs(self):
        ops = [_make_mapper(output_cols=['col_a']),
               _make_mapper(output_cols=['col_a'])]
        self.assertFalse(_are_ops_independent(ops))

    def test_reads_produced_col(self):
        ops = [_make_mapper(output_cols=['col_a']),
               _make_mapper(input_cols=['col_a'], output_cols=['col_b'])]
        self.assertFalse(_are_ops_independent(ops))

    def test_no_output_cols_declared(self):
        op = _make_mapper()
        op._output_columns = []
        self.assertFalse(_are_ops_independent([op]))


class TestMapperGroupBlocker(unittest.TestCase):

    def test_no_blocker(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a']),
               _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['b'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIsNone(result)

    def test_not_fusible(self):
        ops = [_make_mapper(num_gpus=1, fusible=False, vram=0.3, output_cols=['a'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("not explicitly opted", result)

    def test_not_independent(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a']),
               _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("not independent", result)

    def test_different_runtime_envs(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a'],
                            runtime_env={'pip': ['torch']}),
               _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['b'],
                            runtime_env={'pip': ['jax']})]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("runtime environments", result)

    def test_missing_vram_estimate(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=None, output_cols=['a'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("estimated_vram_fraction", result)

    def test_vram_over_limit(self):
        ops = [_make_mapper(num_gpus=1, fusible=True, vram=0.6, output_cols=['a']),
               _make_mapper(num_gpus=1, fusible=True, vram=0.6, output_cols=['b'])]
        result = _mapper_group_blocker(ops, 0.9)
        self.assertIn("exceeds", result)

    def test_invalid_vram_limit(self):
        with self.assertRaises(ValueError):
            _mapper_group_blocker([], 0.0)
        with self.assertRaises(ValueError):
            _mapper_group_blocker([], 1.5)


class TestFuseMapperGroup(unittest.TestCase):

    def test_empty_group(self):
        result = fuse_mapper_group([])
        self.assertEqual(result, [])

    def test_blocked_returns_original(self):
        ops = [_make_mapper(num_gpus=1, fusible=False, vram=0.3, output_cols=['a'])]
        result = fuse_mapper_group(ops)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ops[0])

    def test_successful_fusion(self):
        mock_fused_cls = MagicMock()
        mock_fused_instance = MagicMock()
        mock_fused_cls.return_value = mock_fused_instance
        mock_module = MagicMock()
        mock_module.FusedSequentialBatchOp = mock_fused_cls
        ops = [_make_mapper(name='op1', num_gpus=1, fusible=True, vram=0.3,
                            output_cols=['a'], num_cpus=2, batch_size=4),
               _make_mapper(name='op2', num_gpus=1, fusible=True, vram=0.3,
                            output_cols=['b'], num_cpus=4, batch_size=2)]
        with patch.dict('sys.modules', {'data_juicer.ops.fused_sequential_batch_op': mock_module}):
            result = fuse_mapper_group(ops, vram_limit=0.9)
        self.assertEqual(len(result), 1)
        mock_fused_cls.assert_called_once()


class TestFuseConsecutiveMappers(unittest.TestCase):

    def test_no_gpu_mappers(self):
        ops = [_make_mapper(num_gpus=0), _make_mapper(num_gpus=0)]
        result = fuse_consecutive_mappers(ops)
        self.assertEqual(len(result), 2)

    def test_single_gpu_mapper_passes_through(self):
        op = _make_mapper(num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])
        result = fuse_consecutive_mappers([op])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], op)

    def test_invalid_vram_limit_raises(self):
        with self.assertRaises(ValueError):
            fuse_consecutive_mappers([], vram_limit=0.0)

    @patch('data_juicer.ops.op_fusion.fuse_mapper_group')
    def test_consecutive_fusible_mappers(self, mock_fuse):
        mock_fuse.return_value = [MagicMock()]
        op1 = _make_mapper(name='g1', num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])
        op2 = _make_mapper(name='g2', num_gpus=1, fusible=True, vram=0.3, output_cols=['b'])
        result = fuse_consecutive_mappers([op1, op2])
        mock_fuse.assert_called_once()

    def test_non_gpu_breaks_group(self):
        op1 = _make_mapper(name='g1', num_gpus=1, fusible=True, vram=0.3, output_cols=['a'])
        cpu_op = _make_mapper(name='cpu', num_gpus=0)
        op2 = _make_mapper(name='g2', num_gpus=1, fusible=True, vram=0.3, output_cols=['b'])
        result = fuse_consecutive_mappers([op1, cpu_op, op2])
        # op1 alone (< 2 so passes through), cpu_op passes through, op2 alone passes through
        self.assertEqual(len(result), 3)


class TestFuseOperators(unittest.TestCase):

    def test_empty_ops(self):
        result = fuse_operators([])
        self.assertEqual(result, [])

    def test_non_filter_passes_through(self):
        mapper = _make_mapper(name='m1')
        result = fuse_operators([mapper])
        self.assertEqual(len(result), 1)

    def test_single_filter_no_fusion(self):
        f = _make_filter(name='f1')
        # Need to properly set up the filter for fuse_filter_group
        # Since fuse_filter_group checks INTER_VARS, and the mock won't be in any,
        # it just appends directly
        result = fuse_operators([f], probe_res=[None])
        self.assertEqual(len(result), 1)

    def test_mapper_fusion_disabled(self):
        mapper = _make_mapper(name='m1')
        result = fuse_operators([mapper], mapper_fusion=False)
        self.assertEqual(len(result), 1)


if __name__ == '__main__':
    unittest.main()
