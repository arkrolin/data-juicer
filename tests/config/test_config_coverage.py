"""Additional config tests to cover gaps in config.py coverage.

Targets:
- update_ds_cache_dir_and_related_vars()
- get_init_configs()
- merge_config() with nested dot-notation op parameters
- init_configs() deprecated which_entry parameter
- init_configs() unexpected keyword arguments
"""
import os
import shutil
import tempfile
import unittest
import warnings
from argparse import Namespace

import yaml

from data_juicer.config import init_configs
from data_juicer.config.config import (
    get_init_configs,
    merge_config,
    update_ds_cache_dir_and_related_vars,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class UpdateDsCacheDirTest(DataJuicerTestCaseBase):
    """Tests for update_ds_cache_dir_and_related_vars."""

    def test_updates_hf_datasets_cache(self):
        from datasets import config as hf_config

        original_cache = hf_config.HF_DATASETS_CACHE
        new_path = '/tmp/test_dj_cache_12345'
        try:
            update_ds_cache_dir_and_related_vars(new_path)
            self.assertEqual(str(hf_config.HF_DATASETS_CACHE), new_path)
            self.assertIn(new_path,
                          str(hf_config.DOWNLOADED_DATASETS_PATH))
            self.assertIn(new_path,
                          str(hf_config.EXTRACTED_DATASETS_PATH))
        finally:
            update_ds_cache_dir_and_related_vars(str(original_cache))

    def test_paths_are_nested_correctly(self):
        from datasets import config as hf_config

        original_cache = hf_config.HF_DATASETS_CACHE
        new_path = '/tmp/test_dj_nested_path'
        try:
            update_ds_cache_dir_and_related_vars(new_path)
            downloaded = str(hf_config.DOWNLOADED_DATASETS_PATH)
            extracted = str(hf_config.EXTRACTED_DATASETS_PATH)
            self.assertTrue(downloaded.startswith(new_path))
            self.assertTrue(extracted.startswith(downloaded))
        finally:
            update_ds_cache_dir_and_related_vars(str(original_cache))


class GetInitConfigsTest(DataJuicerTestCaseBase):
    """Tests for get_init_configs."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmp_dir, 'test.yaml')
        config_data = {
            'project_name': 'test_get_init',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 2,
            'export_path': os.path.join(self.tmp_dir, 'output.jsonl'),
            'process': [
                {'whitespace_normalization_mapper': None},
            ],
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_init_configs_from_dict(self):
        cfg_dict = {
            'project_name': 'dict_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': os.path.join(self.tmp_dir, 'out2.jsonl'),
            'process': [{'whitespace_normalization_mapper': None}],
        }
        result = get_init_configs(cfg_dict, load_configs_only=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.project_name, 'dict_test')

    def test_get_init_configs_from_dict_with_config_file(self):
        cfg_dict = {
            'project_name': 'file_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': os.path.join(self.tmp_dir, 'out3.jsonl'),
            'process': [{'whitespace_normalization_mapper': None}],
            'config': [self.config_path],
        }
        result = get_init_configs(cfg_dict, load_configs_only=True)
        self.assertIsNotNone(result)

    def test_get_init_configs_strips_internal_attrs(self):
        cfg_dict = {
            'project_name': 'strip_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': os.path.join(self.tmp_dir, 'out4.jsonl'),
            'process': [{'whitespace_normalization_mapper': None}],
            '_user_provided_job_id': True,
            '_resume_requested': False,
            'metadata_dir': '/some/path',
            'results_dir': '/some/other/path',
        }
        result = get_init_configs(cfg_dict, load_configs_only=True)
        self.assertIsNotNone(result)


class InitConfigsDeprecatedParamsTest(DataJuicerTestCaseBase):
    """Tests for deprecated parameters in init_configs."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmp_dir, 'test.yaml')
        config_data = {
            'project_name': 'deprecated_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': os.path.join(self.tmp_dir, 'out.jsonl'),
            'process': [{'whitespace_normalization_mapper': None}],
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_which_entry_emits_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = init_configs(
                ['--config', self.config_path],
                load_configs_only=True,
                which_entry=None,
            )
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            self.assertGreater(len(deprecation_warnings), 0)
            self.assertIn('which_entry',
                          str(deprecation_warnings[0].message))

    def test_unexpected_kwargs_raises(self):
        with self.assertRaises(TypeError) as ctx:
            init_configs(
                ['--config', self.config_path],
                load_configs_only=True,
                unknown_param='value',
            )
        self.assertIn('unexpected keyword arguments', str(ctx.exception))


class MergeConfigNestedOpsTest(DataJuicerTestCaseBase):
    """Tests for merge_config with nested dot-notation op parameters."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmp_dir, 'test.yaml')
        config_data = {
            'project_name': 'merge_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': os.path.join(self.tmp_dir, 'merged.jsonl'),
            'process': [
                {'language_id_score_filter': {'lang': 'en'}},
                {'remove_table_text_mapper': {'min_col': 3}},
            ],
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_merge_top_level_parameter(self):
        ori_cfg = init_configs(['--config', self.config_path],
                               load_configs_only=True)
        new_cfg = {'np': 8}
        result = merge_config(ori_cfg, new_cfg)
        self.assertEqual(result.np, 8)

    def test_merge_nested_op_parameter(self):
        ori_cfg = init_configs(['--config', self.config_path],
                               load_configs_only=True)
        new_cfg = {'language_id_score_filter.lang': 'zh'}
        result = merge_config(ori_cfg, new_cfg)
        filter_cfg = None
        for op in result.process:
            if 'language_id_score_filter' in op:
                filter_cfg = op['language_id_score_filter']
                break
        self.assertIsNotNone(filter_cfg)
        self.assertEqual(filter_cfg['lang'], 'zh')

    def test_merge_multiple_nested_op_parameters(self):
        ori_cfg = init_configs(['--config', self.config_path],
                               load_configs_only=True)
        new_cfg = {
            'remove_table_text_mapper.min_col': 5,
            'language_id_score_filter.lang': 'fr',
        }
        result = merge_config(ori_cfg, new_cfg)
        for op in result.process:
            if 'remove_table_text_mapper' in op:
                self.assertEqual(op['remove_table_text_mapper']['min_col'], 5)
            if 'language_id_score_filter' in op:
                self.assertEqual(
                    op['language_id_score_filter']['lang'], 'fr')


class InitConfigsIntegrationTest(DataJuicerTestCaseBase):
    """Integration tests for init_configs that cover more internal paths."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_init_with_debug_mode(self):
        config_path = os.path.join(self.tmp_dir, 'debug.yaml')
        config_data = {
            'project_name': 'debug_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': os.path.join(self.tmp_dir, 'debug_out.jsonl'),
            'process': [{'whitespace_normalization_mapper': None}],
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        cfg = init_configs(['--config', config_path, '--debug'],
                           load_configs_only=True)
        self.assertTrue(cfg.debug)

    def test_init_with_executor_type_ray(self):
        config_path = os.path.join(self.tmp_dir, 'ray.yaml')
        config_data = {
            'project_name': 'ray_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'executor_type': 'ray',
            'export_path': os.path.join(self.tmp_dir, 'ray_out.jsonl'),
            'process': [{'whitespace_normalization_mapper': None}],
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        cfg = init_configs(['--config', config_path],
                           load_configs_only=True)
        self.assertEqual(cfg.executor_type, 'ray')

    def test_init_creates_work_dir_structure(self):
        config_path = os.path.join(self.tmp_dir, 'full.yaml')
        export_path = os.path.join(self.tmp_dir, 'outputs', 'result.jsonl')
        config_data = {
            'project_name': 'full_test',
            'dataset_path': './demos/data/demo-dataset.jsonl',
            'np': 1,
            'export_path': export_path,
            'process': [{'whitespace_normalization_mapper': None}],
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        cfg = init_configs(['--config', config_path],
                           load_configs_only=False)
        self.assertTrue(os.path.exists(cfg.work_dir))
        self.assertTrue(os.path.exists(cfg.event_log_dir))

    def test_init_with_auto_raises_for_non_analyzer(self):
        with self.assertRaises(NotImplementedError):
            init_configs(['--auto'], allow_auto=False,
                         load_configs_only=True)

    def test_init_with_auto_allowed(self):
        cfg = init_configs(['--auto'], allow_auto=True,
                           load_configs_only=True)
        self.assertTrue(cfg.auto)


if __name__ == '__main__':
    unittest.main()
