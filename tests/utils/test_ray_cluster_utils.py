import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.ray_cluster_utils import (
    ClusterTopology,
    detect_cluster_topology,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestClusterTopology(DataJuicerTestCaseBase):

    def test_gpus_per_node_normal(self):
        topo = ClusterTopology(
            num_nodes=4, total_cpus=64.0, total_gpus=8.0,
            available_cpus=32.0, available_gpus=4.0,
        )
        self.assertAlmostEqual(topo.gpus_per_node, 2.0)

    def test_gpus_per_node_zero_nodes(self):
        topo = ClusterTopology(
            num_nodes=0, total_cpus=0.0, total_gpus=0.0,
            available_cpus=0.0, available_gpus=0.0,
        )
        self.assertAlmostEqual(topo.gpus_per_node, 0.0)

    def test_gpus_per_node_cpu_only(self):
        topo = ClusterTopology(
            num_nodes=2, total_cpus=16.0, total_gpus=0.0,
            available_cpus=8.0, available_gpus=0.0,
        )
        self.assertAlmostEqual(topo.gpus_per_node, 0.0)

    def test_frozen_dataclass(self):
        topo = ClusterTopology(
            num_nodes=1, total_cpus=8.0, total_gpus=1.0,
            available_cpus=4.0, available_gpus=0.5,
        )
        with self.assertRaises(Exception):
            topo.num_nodes = 2


class TestDetectClusterTopology(DataJuicerTestCaseBase):

    def _patch_ray(self, mock_ray):
        return patch.dict('sys.modules', {'ray': mock_ray})

    def test_ray_not_initialized(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False
        with self._patch_ray(mock_ray):
            topo = detect_cluster_topology()
        self.assertEqual(topo.num_nodes, 1)
        self.assertEqual(topo.total_cpus, 0.0)
        self.assertEqual(topo.total_gpus, 0.0)

    def test_ray_initialized_multi_node(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {'Alive': True}, {'Alive': True}, {'Alive': False},
        ]
        mock_ray.cluster_resources.return_value = {'CPU': 32, 'GPU': 4}
        mock_ray.available_resources.return_value = {'CPU': 16, 'GPU': 2}
        with self._patch_ray(mock_ray):
            topo = detect_cluster_topology()
        self.assertEqual(topo.num_nodes, 2)
        self.assertAlmostEqual(topo.total_cpus, 32.0)
        self.assertAlmostEqual(topo.total_gpus, 4.0)
        self.assertAlmostEqual(topo.available_cpus, 16.0)
        self.assertAlmostEqual(topo.available_gpus, 2.0)

    def test_ray_initialized_no_gpu_key(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [{'Alive': True}]
        mock_ray.cluster_resources.return_value = {'CPU': 8}
        mock_ray.available_resources.return_value = {'CPU': 4}
        with self._patch_ray(mock_ray):
            topo = detect_cluster_topology()
        self.assertEqual(topo.num_nodes, 1)
        self.assertAlmostEqual(topo.total_gpus, 0.0)
        self.assertAlmostEqual(topo.available_gpus, 0.0)

    def test_ray_import_error_fallback(self):
        with patch.dict('sys.modules', {'ray': None}):
            topo = detect_cluster_topology()
        self.assertEqual(topo.num_nodes, 1)
        self.assertEqual(topo.total_cpus, 0.0)

    def test_ray_exception_fallback(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.side_effect = RuntimeError("connection lost")
        with self._patch_ray(mock_ray):
            topo = detect_cluster_topology()
        self.assertEqual(topo.num_nodes, 1)
        self.assertEqual(topo.total_cpus, 0.0)

    def test_all_nodes_dead_defaults_to_1(self):
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        mock_ray.nodes.return_value = [
            {'Alive': False}, {'Alive': False},
        ]
        mock_ray.cluster_resources.return_value = {'CPU': 0}
        mock_ray.available_resources.return_value = {}
        with self._patch_ray(mock_ray):
            topo = detect_cluster_topology()
        self.assertEqual(topo.num_nodes, 1)


if __name__ == '__main__':
    unittest.main()
