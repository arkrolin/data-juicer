import tempfile
import threading
import unittest
from types import SimpleNamespace

from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.dag_execution_strategies import (
    NonPartitionedDAGStrategy,
    PartitionedDAGStrategy,
    is_global_operation,
)
from data_juicer.core.executor.pipeline_dag import DAGNodeStatus
from data_juicer.ops.deduplicator.document_deduplicator import \
    DocumentDeduplicator
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.punctuation_normalization_mapper import \
    PunctuationNormalizationMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ConcreteTestExecutor(DAGExecutionMixin):
    """Concrete class that uses the mixin for testing."""

    def __init__(self, executor_type='default', num_partitions=None):
        DAGExecutionMixin.__init__(self)
        self.executor_type = executor_type
        if num_partitions is not None:
            self.num_partitions = num_partitions


class DagExecutionMixinTest(DataJuicerTestCaseBase):

    def _make_cfg(self, use_dag=True, process=None):
        work_dir = tempfile.mkdtemp()
        return SimpleNamespace(
            work_dir=work_dir,
            use_dag=use_dag,
            process=process or [],
        )

    # ------------------------------------------------------------------
    # Thread-local current_dag_node property
    # ------------------------------------------------------------------

    def test_current_dag_node_default_is_none(self):
        executor = ConcreteTestExecutor()
        self.assertIsNone(executor.current_dag_node)

    def test_current_dag_node_set_and_get(self):
        executor = ConcreteTestExecutor()
        executor.current_dag_node = 'op_001_test'
        self.assertEqual(executor.current_dag_node, 'op_001_test')

    def test_current_dag_node_thread_local_isolation(self):
        executor = ConcreteTestExecutor()
        executor.current_dag_node = 'main_thread_node'

        results = {}
        barrier = threading.Barrier(2)

        def thread_func(name, node_id):
            executor.current_dag_node = node_id
            barrier.wait()
            results[name] = executor.current_dag_node

        t1 = threading.Thread(target=thread_func, args=('t1', 'node_A'))
        t2 = threading.Thread(target=thread_func, args=('t2', 'node_B'))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have its own value
        self.assertEqual(results['t1'], 'node_A')
        self.assertEqual(results['t2'], 'node_B')
        # Main thread should retain its own value
        self.assertEqual(executor.current_dag_node, 'main_thread_node')

    # ------------------------------------------------------------------
    # _is_partitioned_executor
    # ------------------------------------------------------------------

    def test_is_partitioned_executor_false_for_default(self):
        executor = ConcreteTestExecutor(executor_type='default')
        self.assertFalse(executor._is_partitioned_executor())

    def test_is_partitioned_executor_true_for_ray_partitioned(self):
        executor = ConcreteTestExecutor(executor_type='ray_partitioned')
        self.assertTrue(executor._is_partitioned_executor())

    def test_is_partitioned_executor_false_for_ray(self):
        executor = ConcreteTestExecutor(executor_type='ray')
        self.assertFalse(executor._is_partitioned_executor())

    # ------------------------------------------------------------------
    # _create_execution_strategy
    # ------------------------------------------------------------------

    def test_create_execution_strategy_non_partitioned(self):
        executor = ConcreteTestExecutor(executor_type='default')
        cfg = self._make_cfg()
        strategy = executor._create_execution_strategy(cfg)
        self.assertIsInstance(strategy, NonPartitionedDAGStrategy)

    def test_create_execution_strategy_partitioned(self):
        executor = ConcreteTestExecutor(
            executor_type='ray_partitioned', num_partitions=3
        )
        cfg = self._make_cfg()
        strategy = executor._create_execution_strategy(cfg)
        self.assertIsInstance(strategy, PartitionedDAGStrategy)
        self.assertEqual(strategy.num_partitions, 3)

    # ------------------------------------------------------------------
    # _initialize_dag_execution with use_dag=False
    # ------------------------------------------------------------------

    def test_initialize_dag_execution_skips_when_use_dag_false(self):
        executor = ConcreteTestExecutor()
        cfg = self._make_cfg(use_dag=False)
        executor._initialize_dag_execution(cfg)
        self.assertTrue(executor.dag_initialized)
        self.assertIsNone(executor.pipeline_dag)
        self.assertIsNone(executor.dag_execution_strategy)

    def test_initialize_dag_execution_skips_when_already_initialized(self):
        executor = ConcreteTestExecutor()
        executor.dag_initialized = True
        cfg = self._make_cfg(use_dag=True)
        # Should return early without initializing pipeline_dag
        executor._initialize_dag_execution(cfg)
        self.assertIsNone(executor.pipeline_dag)

    # ------------------------------------------------------------------
    # _initialize_dag_execution with ops - full flow
    # ------------------------------------------------------------------

    def test_initialize_dag_execution_full_flow_non_partitioned(self):
        executor = ConcreteTestExecutor(executor_type='default')
        ops = [PunctuationNormalizationMapper(), TextLengthFilter()]
        cfg = self._make_cfg(use_dag=True)

        executor._initialize_dag_execution(cfg, ops=ops)

        self.assertTrue(executor.dag_initialized)
        self.assertIsNotNone(executor.pipeline_dag)
        self.assertIsInstance(
            executor.dag_execution_strategy, NonPartitionedDAGStrategy
        )
        self.assertIsNotNone(executor.dag_execution_start_time)

        # Verify nodes created
        nodes = executor.pipeline_dag.nodes
        self.assertEqual(len(nodes), 2)
        self.assertIn('op_001_punctuation_normalization_mapper', nodes)
        self.assertIn('op_002_text_length_filter', nodes)

        # Verify dependencies (sequential)
        second_node = nodes['op_002_text_length_filter']
        self.assertIn(
            'op_001_punctuation_normalization_mapper',
            second_node['dependencies'],
        )

    def test_initialize_dag_execution_full_flow_partitioned(self):
        executor = ConcreteTestExecutor(
            executor_type='ray_partitioned', num_partitions=2
        )
        ops = [PunctuationNormalizationMapper(), TextLengthFilter()]
        cfg = self._make_cfg(use_dag=True)

        executor._initialize_dag_execution(cfg, ops=ops)

        self.assertTrue(executor.dag_initialized)
        self.assertIsNotNone(executor.pipeline_dag)
        self.assertIsInstance(
            executor.dag_execution_strategy, PartitionedDAGStrategy
        )

        # Verify partition nodes: 2 ops x 2 partitions = 4 nodes
        nodes = executor.pipeline_dag.nodes
        self.assertEqual(len(nodes), 4)
        self.assertIn(
            'op_001_punctuation_normalization_mapper_partition_0', nodes
        )
        self.assertIn(
            'op_001_punctuation_normalization_mapper_partition_1', nodes
        )
        self.assertIn('op_002_text_length_filter_partition_0', nodes)
        self.assertIn('op_002_text_length_filter_partition_1', nodes)

    # ------------------------------------------------------------------
    # _mark_dag_node_started / completed / failed
    # ------------------------------------------------------------------

    def _setup_initialized_executor(self):
        """Helper to create an initialized executor with nodes."""
        executor = ConcreteTestExecutor(executor_type='default')
        ops = [PunctuationNormalizationMapper(), TextLengthFilter()]
        cfg = self._make_cfg(use_dag=True)
        executor._initialize_dag_execution(cfg, ops=ops)
        return executor

    def test_mark_dag_node_started(self):
        executor = self._setup_initialized_executor()
        node_id = 'op_001_punctuation_normalization_mapper'

        executor._mark_dag_node_started(node_id)

        node = executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.RUNNING.value)
        self.assertIsNotNone(node['start_time'])
        self.assertEqual(executor.current_dag_node, node_id)
        self.assertEqual(executor.current_dag_nodes[None], node_id)

    def test_mark_dag_node_completed(self):
        executor = self._setup_initialized_executor()
        node_id = 'op_001_punctuation_normalization_mapper'

        executor._mark_dag_node_started(node_id)
        executor._mark_dag_node_completed(node_id, duration=2.5)

        node = executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.COMPLETED.value)
        self.assertEqual(node['actual_duration'], 2.5)
        self.assertIsNone(executor.current_dag_node)
        self.assertNotIn(None, executor.current_dag_nodes)

    def test_mark_dag_node_failed(self):
        executor = self._setup_initialized_executor()
        node_id = 'op_002_text_length_filter'

        executor._mark_dag_node_started(node_id)
        executor._mark_dag_node_failed(node_id, 'Something went wrong', duration=0.3)

        node = executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.FAILED.value)
        self.assertEqual(node['error_message'], 'Something went wrong')
        self.assertIsNone(executor.current_dag_node)
        self.assertNotIn(None, executor.current_dag_nodes)

    def test_mark_dag_node_started_with_nonexistent_node(self):
        executor = self._setup_initialized_executor()
        # Should not raise - just returns early
        executor._mark_dag_node_started('nonexistent_node')
        self.assertIsNone(executor.current_dag_node)

    def test_mark_dag_node_completed_with_nonexistent_node(self):
        executor = self._setup_initialized_executor()
        # Should not raise
        executor._mark_dag_node_completed('nonexistent_node', duration=1.0)

    def test_mark_dag_node_failed_with_nonexistent_node(self):
        executor = self._setup_initialized_executor()
        # Should not raise
        executor._mark_dag_node_failed('nonexistent_node', 'error')

    # ------------------------------------------------------------------
    # _get_dag_node_for_operation
    # ------------------------------------------------------------------

    def test_get_dag_node_for_operation_non_partitioned(self):
        executor = self._setup_initialized_executor()

        node_id = executor._get_dag_node_for_operation(
            'punctuation_normalization_mapper', 0
        )
        self.assertEqual(node_id, 'op_001_punctuation_normalization_mapper')

        node_id = executor._get_dag_node_for_operation(
            'text_length_filter', 1
        )
        self.assertEqual(node_id, 'op_002_text_length_filter')

    def test_get_dag_node_for_operation_partitioned(self):
        executor = ConcreteTestExecutor(
            executor_type='ray_partitioned', num_partitions=2
        )
        ops = [PunctuationNormalizationMapper(), TextLengthFilter()]
        cfg = self._make_cfg(use_dag=True)
        executor._initialize_dag_execution(cfg, ops=ops)

        node_id = executor._get_dag_node_for_operation(
            'punctuation_normalization_mapper', 0, partition_id=1
        )
        self.assertEqual(
            node_id, 'op_001_punctuation_normalization_mapper_partition_1'
        )

    def test_get_dag_node_for_operation_no_strategy(self):
        executor = ConcreteTestExecutor()
        # Not initialized, strategy is None
        result = executor._get_dag_node_for_operation('some_op', 0)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _detect_convergence_points
    # ------------------------------------------------------------------

    def test_detect_convergence_points_with_global_op(self):
        executor = ConcreteTestExecutor(executor_type='default')
        ops = [
            PunctuationNormalizationMapper(),
            DocumentDeduplicator(),
            TextLengthFilter(),
        ]
        executor._dag_ops = ops
        cfg = self._make_cfg(use_dag=True)

        convergence_points = executor._detect_convergence_points(cfg)
        # DocumentDeduplicator is at index 1 and is a global operation
        self.assertIn(1, convergence_points)
        # Non-global ops should not be convergence points
        self.assertNotIn(0, convergence_points)
        self.assertNotIn(2, convergence_points)

    def test_detect_convergence_points_no_global_ops(self):
        executor = ConcreteTestExecutor(executor_type='default')
        ops = [PunctuationNormalizationMapper(), TextLengthFilter()]
        executor._dag_ops = ops
        cfg = self._make_cfg(use_dag=True)

        convergence_points = executor._detect_convergence_points(cfg)
        self.assertEqual(convergence_points, [])

    def test_detect_convergence_points_with_converge_after_flag(self):
        executor = ConcreteTestExecutor(executor_type='default')
        mapper = PunctuationNormalizationMapper()
        mapper.converge_after = True
        ops = [mapper, TextLengthFilter()]
        executor._dag_ops = ops
        cfg = self._make_cfg(use_dag=True)

        convergence_points = executor._detect_convergence_points(cfg)
        self.assertIn(0, convergence_points)

    # ------------------------------------------------------------------
    # Full lifecycle: start -> complete multiple nodes
    # ------------------------------------------------------------------

    def test_full_node_lifecycle(self):
        executor = self._setup_initialized_executor()

        # Start and complete first node
        node1 = 'op_001_punctuation_normalization_mapper'
        node2 = 'op_002_text_length_filter'

        executor._mark_dag_node_started(node1)
        self.assertEqual(
            executor.pipeline_dag.nodes[node1]['status'],
            DAGNodeStatus.RUNNING.value,
        )

        executor._mark_dag_node_completed(node1, duration=1.0)
        self.assertEqual(
            executor.pipeline_dag.nodes[node1]['status'],
            DAGNodeStatus.COMPLETED.value,
        )

        # Start and fail second node
        executor._mark_dag_node_started(node2)
        self.assertEqual(
            executor.pipeline_dag.nodes[node2]['status'],
            DAGNodeStatus.RUNNING.value,
        )

        executor._mark_dag_node_failed(node2, 'timeout error', duration=5.0)
        self.assertEqual(
            executor.pipeline_dag.nodes[node2]['status'],
            DAGNodeStatus.FAILED.value,
        )

        # Verify execution summary
        summary = executor.pipeline_dag.get_execution_summary()
        self.assertEqual(summary['completed_nodes'], 1)
        self.assertEqual(summary['failed_nodes'], 1)
        self.assertEqual(summary['total_nodes'], 2)

    # ------------------------------------------------------------------
    # Partitioned execution with convergence (scatter-gather)
    # ------------------------------------------------------------------

    def test_partitioned_with_convergence_creates_scatter_gather_nodes(self):
        executor = ConcreteTestExecutor(
            executor_type='ray_partitioned', num_partitions=2
        )
        ops = [
            PunctuationNormalizationMapper(),
            DocumentDeduplicator(),
            TextLengthFilter(),
        ]
        cfg = self._make_cfg(use_dag=True)
        executor._initialize_dag_execution(cfg, ops=ops)

        nodes = executor.pipeline_dag.nodes
        # Should have scatter-gather node for DocumentDeduplicator
        sg_node_id = 'sg_001_document_deduplicator'
        self.assertIn(sg_node_id, nodes)
        self.assertEqual(nodes[sg_node_id]['node_type'], 'scatter_gather')

    # ------------------------------------------------------------------
    # Default use_dag behavior (auto-detect based on executor_type)
    # ------------------------------------------------------------------

    def test_use_dag_defaults_false_for_standalone_executor(self):
        executor = ConcreteTestExecutor(executor_type='default')
        cfg = SimpleNamespace(
            work_dir=tempfile.mkdtemp(), process=[]
        )
        # No use_dag attribute - should default to False for standalone
        executor._initialize_dag_execution(cfg)
        self.assertTrue(executor.dag_initialized)
        self.assertIsNone(executor.pipeline_dag)

    def test_use_dag_defaults_true_for_partitioned_executor(self):
        executor = ConcreteTestExecutor(
            executor_type='ray_partitioned', num_partitions=2
        )
        ops = [PunctuationNormalizationMapper()]
        cfg = SimpleNamespace(
            work_dir=tempfile.mkdtemp(), process=[]
        )
        # No use_dag attribute - should default to True for partitioned
        executor._initialize_dag_execution(cfg, ops=ops)
        self.assertTrue(executor.dag_initialized)
        self.assertIsNotNone(executor.pipeline_dag)


if __name__ == '__main__':
    unittest.main()
