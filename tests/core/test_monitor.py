import os
import unittest
import time
from loguru import logger
from data_juicer.core import Monitor
from data_juicer.core.monitor import resource_monitor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG
from unittest.mock import MagicMock, patch

class MonitorTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.work_dir = 'tmp/test_monitor/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')
        super().tearDown()

    def test_monitor_current_resources(self):
        resource_dict = Monitor.monitor_current_resources()
        logger.info(resource_dict)
        self.assertIn('timestamp', resource_dict)
        self.assertIn('CPU count', resource_dict)
        self.assertIn('Mem. util.', resource_dict)

    def test_analyze_resource_util_list(self):
        resource_samples = []
        for i in range(5):
            resource_samples.append(Monitor.monitor_current_resources())
            time.sleep(0.2)
        resource_util_list = [{
            'time': 1,
            'sampling interval': 0.2,
            'resource': resource_samples,
        }]
        analysis_res = Monitor.analyze_resource_util_list(resource_util_list)
        logger.info(analysis_res)
        item = analysis_res[0]
        self.assertIn('resource_analysis', item)
        resource_analysis = item['resource_analysis']
        cpu_util = resource_analysis['CPU util.']
        self.assertIn('max', cpu_util)
        self.assertIn('min', cpu_util)
        self.assertIn('avg', cpu_util)

        # test draw resource util list
        Monitor.draw_resource_util_graph(resource_util_list, self.work_dir)
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'func_0_CPU_util..jpg')))
        self.assertTrue(os.path.exists(os.path.join(self.work_dir, 'func_0_Used_mem..jpg')))

    def test_monitor_func(self):
        def test_func():
            for _ in range(5):
                time.sleep(0.2)

        ret, resource_util_dict = Monitor.monitor_func(test_func, sample_interval=0.3)
        self.assertIsNone(ret)
        self.assertIn("resource", resource_util_dict)
        self.assertIn("sampling interval", resource_util_dict)
        self.assertIn("time", resource_util_dict)

        self.assertEqual(resource_util_dict["sampling interval"], 0.3)
        resource_list = resource_util_dict["resource"]
        self.assertGreater(len(resource_list), 0)


class MonitorFuncArgsCoverageTest(DataJuicerTestCaseBase):
    """Tests for monitor_func() with different args dispatch branches."""

    @TEST_TAG("standalone")
    def test_monitor_func_with_dict_args(self):
        """monitor_func() with args as a dict dispatches via **kwargs."""

        def func_with_kwargs(a=0, b=0):
            return a + b

        ret, resource_util_dict = Monitor.monitor_func(
            func_with_kwargs, args={'a': 3, 'b': 7}, sample_interval=0.3
        )
        self.assertEqual(ret, 10)
        self.assertIn('resource', resource_util_dict)
        self.assertIn('time', resource_util_dict)

    @TEST_TAG("standalone")
    def test_monitor_func_with_list_args(self):
        """monitor_func() with args as a list dispatches via *args."""

        def func_with_positional(x, y):
            return x * y

        ret, resource_util_dict = Monitor.monitor_func(
            func_with_positional, args=[4, 5], sample_interval=0.3
        )
        self.assertEqual(ret, 20)
        self.assertIn('resource', resource_util_dict)

    @TEST_TAG("standalone")
    def test_monitor_func_with_tuple_args(self):
        """monitor_func() with args as a tuple dispatches via *args."""

        def func_with_positional(x, y):
            return x - y

        ret, resource_util_dict = Monitor.monitor_func(
            func_with_positional, args=(10, 3), sample_interval=0.3
        )
        self.assertEqual(ret, 7)
        self.assertIn('resource', resource_util_dict)

    @TEST_TAG("standalone")
    def test_monitor_func_with_single_value_args(self):
        """monitor_func() with args as a single non-iterable value (else branch)."""

        def func_with_single_arg(val):
            return val * 2

        ret, resource_util_dict = Monitor.monitor_func(
            func_with_single_arg, args=42, sample_interval=0.3
        )
        self.assertEqual(ret, 84)
        self.assertIn('resource', resource_util_dict)


class AnalyzeSingleResourceUtilCoverageTest(DataJuicerTestCaseBase):
    """Tests for analyze_single_resource_util() edge cases."""

    @TEST_TAG("standalone")
    def test_analyze_single_resource_util_none_values_skipped(self):
        """Records with None for a dynamic field should be skipped."""
        resource_util_dict = {
            'resource': [
                {
                    'CPU util.': 0.5,
                    'Used mem.': None,
                    'Free mem.': 200.0,
                },
                {
                    'CPU util.': 0.7,
                    'Used mem.': None,
                    'Free mem.': 180.0,
                },
            ]
        }
        result = Monitor.analyze_single_resource_util(resource_util_dict)
        analysis = result['resource_analysis']
        # Used mem. should not appear because all values were None
        self.assertNotIn('Used mem.', analysis)
        # CPU util. and Free mem. should be present
        self.assertIn('CPU util.', analysis)
        self.assertAlmostEqual(analysis['CPU util.']['max'], 0.7)
        self.assertAlmostEqual(analysis['CPU util.']['min'], 0.5)
        self.assertAlmostEqual(analysis['CPU util.']['avg'], 0.6)
        self.assertIn('Free mem.', analysis)
        self.assertAlmostEqual(analysis['Free mem.']['max'], 200.0)
        self.assertAlmostEqual(analysis['Free mem.']['min'], 180.0)

    @TEST_TAG("standalone")
    def test_analyze_single_resource_util_list_values_extended(self):
        """Records with list values (GPU metrics) should be extended, not appended."""
        resource_util_dict = {
            'resource': [
                {
                    'GPU util.': [0.3, 0.4],
                    'GPU free mem.': [1000.0, 2000.0],
                },
                {
                    'GPU util.': [0.5, 0.6],
                    'GPU free mem.': [900.0, 1800.0],
                },
            ]
        }
        result = Monitor.analyze_single_resource_util(resource_util_dict)
        analysis = result['resource_analysis']
        # GPU util. should have 4 values extended from two lists of 2
        self.assertIn('GPU util.', analysis)
        self.assertAlmostEqual(analysis['GPU util.']['max'], 0.6)
        self.assertAlmostEqual(analysis['GPU util.']['min'], 0.3)
        self.assertAlmostEqual(
            analysis['GPU util.']['avg'], (0.3 + 0.4 + 0.5 + 0.6) / 4
        )
        # GPU free mem. should similarly extend
        self.assertIn('GPU free mem.', analysis)
        self.assertAlmostEqual(analysis['GPU free mem.']['max'], 2000.0)
        self.assertAlmostEqual(analysis['GPU free mem.']['min'], 900.0)

    @TEST_TAG("standalone")
    def test_analyze_single_resource_util_mixed_none_and_list(self):
        """Mix of None and list values in the same key across records."""
        resource_util_dict = {
            'resource': [
                {
                    'GPU used mem.': None,
                },
                {
                    'GPU used mem.': [512.0, 1024.0],
                },
            ]
        }
        result = Monitor.analyze_single_resource_util(resource_util_dict)
        analysis = result['resource_analysis']
        # Only the second record contributes values
        self.assertIn('GPU used mem.', analysis)
        self.assertAlmostEqual(analysis['GPU used mem.']['max'], 1024.0)
        self.assertAlmostEqual(analysis['GPU used mem.']['min'], 512.0)


class DrawResourceUtilGraphCoverageTest(DataJuicerTestCaseBase):
    """Tests for draw_resource_util_graph() with GPU list fields."""

    def setUp(self):
        super().setUp()
        self.work_dir = '/tmp/test_monitor_coverage_draw'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')
        super().tearDown()

    @TEST_TAG("standalone")
    def test_draw_resource_util_graph_with_gpu_list_fields(self):
        """draw_resource_util_graph handles list values in metric fields (GPU)."""
        resource_list = [
            {
                'CPU util.': 0.5,
                'Used mem.': 4000.0,
                'Free mem.': 2000.0,
                'Available mem.': 3000.0,
                'Mem. util.': 0.6,
                'GPU free mem.': [1000.0, 2000.0],
                'GPU used mem.': [500.0, 800.0],
                'GPU util.': [0.3, 0.4],
            },
            {
                'CPU util.': 0.6,
                'Used mem.': 4200.0,
                'Free mem.': 1800.0,
                'Available mem.': 2800.0,
                'Mem. util.': 0.65,
                'GPU free mem.': [950.0, 1900.0],
                'GPU used mem.': [550.0, 900.0],
                'GPU util.': [0.35, 0.45],
            },
        ]
        resource_util_list = [{
            'resource': resource_list,
            'sampling interval': 0.5,
        }]
        # This should not raise even though GPU fields are lists
        Monitor.draw_resource_util_graph(resource_util_list, self.work_dir)
        # Check that scalar metric files were created
        self.assertTrue(
            os.path.exists(os.path.join(self.work_dir, 'func_0_CPU_util..jpg'))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.work_dir, 'func_0_Used_mem..jpg'))
        )
        # GPU list fields also produce files (matplotlib will plot the lists)
        self.assertTrue(
            os.path.exists(os.path.join(self.work_dir, 'func_0_GPU_util..jpg'))
        )


class ResourceMonitorBrokenPipeCoverageTest(DataJuicerTestCaseBase):
    """Tests for resource_monitor() BrokenPipeError handling."""

    @TEST_TAG("standalone")
    def test_resource_monitor_broken_pipe_error(self):
        """resource_monitor returns gracefully on BrokenPipeError."""

        class FaultyDict:
            """A dict-like object that raises BrokenPipeError on key access."""

            def __getitem__(self, key):
                if key == 'stop':
                    raise BrokenPipeError("pipe broken")
                raise KeyError(key)

            def __setitem__(self, key, value):
                pass

        mdict = FaultyDict()
        # resource_monitor should return without raising
        with patch(
            'data_juicer.core.monitor.Monitor.monitor_current_resources',
            return_value={'CPU util.': 0.5}
        ), patch('data_juicer.core.monitor.time.sleep', return_value=None):
            resource_monitor(mdict, interval=0.1)
        # If we get here, it handled BrokenPipeError gracefully

    @TEST_TAG("standalone")
    def test_resource_monitor_file_not_found_error(self):
        """resource_monitor returns gracefully on FileNotFoundError."""

        class FaultyDict:
            """A dict-like object that raises FileNotFoundError on key access."""

            def __getitem__(self, key):
                if key == 'stop':
                    raise FileNotFoundError("file not found")
                raise KeyError(key)

            def __setitem__(self, key, value):
                pass

        mdict = FaultyDict()
        with patch(
            'data_juicer.core.monitor.Monitor.monitor_current_resources',
            return_value={'CPU util.': 0.5}
        ), patch('data_juicer.core.monitor.time.sleep', return_value=None):
            resource_monitor(mdict, interval=0.1)
        # If we get here, it handled FileNotFoundError gracefully


if __name__ == '__main__':
    unittest.main()
