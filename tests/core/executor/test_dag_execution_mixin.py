import json
import os
import shutil
import tempfile
import threading
import time
import unittest

from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.dag_execution_strategies import (
    NonPartitionedDAGStrategy,
    PartitionedDAGStrategy,
)
from data_juicer.core.executor.pipeline_dag import DAGNodeStatus, PipelineDAG
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MockOperation:
    """Mock operation for testing."""

    def __init__(self, name, op_type='mapper'):
        self._name = name
        self._op_type = op_type


class MockConfig:
    """Mock configuration object."""

    def __init__(self, work_dir, process=None, use_dag=True):
        self.work_dir = work_dir
        self.process = process or []
        self.use_dag = use_dag


class ConcreteDAGExecutor(DAGExecutionMixin):
    """Concrete class using DAGExecutionMixin for testing."""

    def __init__(self, executor_type='default', num_partitions=None):
        super().__init__()
        self.executor_type = executor_type
        self.num_partitions = num_partitions
        self._logged_events = []

    def _is_partitioned_executor(self):
        return self.executor_type == 'ray_partitioned'

    def log_dag_build_start(self, ast_info):
        self._logged_events.append(('dag_build_start', ast_info))

    def log_dag_build_complete(self, dag_info):
        self._logged_events.append(('dag_build_complete', dag_info))

    def log_dag_execution_plan_saved(self, plan_path, dag_info):
        self._logged_events.append(('dag_plan_saved', plan_path, dag_info))

    def log_dag_node_start(self, node_id, node_info):
        self._logged_events.append(('dag_node_start', node_id, node_info))

    def log_dag_node_complete(self, node_id, node_info, duration):
        self._logged_events.append(
            ('dag_node_complete', node_id, node_info, duration))

    def log_dag_node_failed(self, node_id, node_info, error_message, duration):
        self._logged_events.append(
            ('dag_node_failed', node_id, node_info, error_message, duration))

    def log_op_start(self, partition_id, op_name, op_idx, metadata):
        self._logged_events.append(
            ('op_start', partition_id, op_name, op_idx, metadata))

    def log_op_complete(self, partition_id, op_name, op_idx, duration,
                        checkpoint_path, input_rows, output_rows):
        self._logged_events.append(
            ('op_complete', partition_id, op_name, op_idx, duration))

    def log_op_failed(self, partition_id, op_name, op_idx, error, retry_count):
        self._logged_events.append(
            ('op_failed', partition_id, op_name, op_idx, error))


class DAGExecutionMixinInitTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_init_sets_default_attributes(self):
        executor = ConcreteDAGExecutor()
        self.assertIsNone(executor.pipeline_dag)
        self.assertFalse(executor.dag_initialized)
        self.assertEqual(executor.current_dag_nodes, {})
        self.assertIsNone(executor.dag_execution_start_time)
        self.assertIsNone(executor.dag_execution_strategy)

    def test_initialize_disabled_for_standalone(self):
        executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=False)
        ops = [MockOperation('text_filter')]
        executor._initialize_dag_execution(cfg, ops)
        self.assertTrue(executor.dag_initialized)
        self.assertIsNone(executor.pipeline_dag)

    def test_initialize_non_partitioned(self):
        executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_length_filter', 'filter'),
               MockOperation('clean_email_mapper', 'mapper')]
        executor._initialize_dag_execution(cfg, ops)
        self.assertTrue(executor.dag_initialized)
        self.assertIsNotNone(executor.pipeline_dag)
        self.assertIsInstance(executor.dag_execution_strategy,
                             NonPartitionedDAGStrategy)
        self.assertIsNotNone(executor.dag_execution_start_time)

    def test_initialize_partitioned(self):
        executor = ConcreteDAGExecutor(executor_type='ray_partitioned',
                                       num_partitions=4)
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_length_filter', 'filter')]
        executor._initialize_dag_execution(cfg, ops)
        self.assertTrue(executor.dag_initialized)
        self.assertIsInstance(executor.dag_execution_strategy,
                             PartitionedDAGStrategy)

    def test_initialize_idempotent(self):
        executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter')]
        executor._initialize_dag_execution(cfg, ops)
        first_dag = executor.pipeline_dag
        executor._initialize_dag_execution(cfg, ops)
        self.assertIs(executor.pipeline_dag, first_dag)

    def test_initialize_partitioned_no_num_partitions_raises(self):
        executor = ConcreteDAGExecutor(executor_type='ray_partitioned',
                                       num_partitions=None)
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter')]
        with self.assertRaises(ValueError):
            executor._initialize_dag_execution(cfg, ops)


class DAGExecutionMixinNodeStateTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()
        self.executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_length_filter', 'filter'),
               MockOperation('clean_email_mapper', 'mapper')]
        self.executor._initialize_dag_execution(cfg, ops)
        self.node_ids = list(self.executor.pipeline_dag.nodes.keys())

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_mark_node_started(self):
        node_id = self.node_ids[0]
        self.executor._mark_dag_node_started(node_id)
        node = self.executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.RUNNING.value)
        self.assertEqual(self.executor.current_dag_node, node_id)

    def test_mark_node_completed(self):
        node_id = self.node_ids[0]
        self.executor._mark_dag_node_started(node_id)
        self.executor._mark_dag_node_completed(node_id, duration=1.5)
        node = self.executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.COMPLETED.value)
        self.assertIsNone(self.executor.current_dag_node)

    def test_mark_node_failed(self):
        node_id = self.node_ids[0]
        self.executor._mark_dag_node_started(node_id)
        self.executor._mark_dag_node_failed(node_id, 'test error', 0.5)
        node = self.executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.FAILED.value)
        self.assertIsNone(self.executor.current_dag_node)

    def test_mark_nonexistent_node_is_noop(self):
        self.executor._mark_dag_node_started('nonexistent_node')
        self.executor._mark_dag_node_completed('nonexistent_node')
        self.executor._mark_dag_node_failed('nonexistent_node', 'err')
        self.assertIsNone(self.executor.current_dag_node)

    def test_mark_node_logs_events(self):
        node_id = self.node_ids[0]
        self.executor._mark_dag_node_started(node_id)
        self.executor._mark_dag_node_completed(node_id, duration=2.0)
        starts = [e for e in self.executor._logged_events
                  if e[0] == 'dag_node_start']
        completes = [e for e in self.executor._logged_events
                     if e[0] == 'dag_node_complete']
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(completes), 1)
        self.assertEqual(starts[0][1], node_id)
        self.assertEqual(completes[0][1], node_id)

    def test_thread_safety_of_node_marking(self):
        node_id = self.node_ids[0]
        errors = []

        def mark_started():
            try:
                self.executor._mark_dag_node_started(node_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mark_started) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)
        node = self.executor.pipeline_dag.nodes[node_id]
        self.assertEqual(node['status'], DAGNodeStatus.RUNNING.value)


class DAGExecutionMixinMonitoringTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()
        self.executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=True)
        self.ops = [MockOperation('text_length_filter', 'filter'),
                    MockOperation('clean_email_mapper', 'mapper')]
        self.executor._initialize_dag_execution(cfg, self.ops)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_pre_execute_marks_all_nodes_started(self):
        self.executor._pre_execute_operations_with_dag_monitoring(self.ops)
        for node_id, node in self.executor.pipeline_dag.nodes.items():
            self.assertEqual(node['status'], DAGNodeStatus.RUNNING.value)

    def test_post_execute_marks_all_nodes_completed(self):
        self.executor._pre_execute_operations_with_dag_monitoring(self.ops)
        metrics = {'duration': 4.0, 'input_rows': 100, 'output_rows': 80}
        self.executor._post_execute_operations_with_dag_monitoring(
            self.ops, metrics=metrics)
        for node_id, node in self.executor.pipeline_dag.nodes.items():
            self.assertEqual(node['status'], DAGNodeStatus.COMPLETED.value)

    def test_post_execute_distributes_duration(self):
        self.executor._pre_execute_operations_with_dag_monitoring(self.ops)
        metrics = {'duration': 4.0, 'input_rows': 100, 'output_rows': 80}
        self.executor._post_execute_operations_with_dag_monitoring(
            self.ops, metrics=metrics)
        for node in self.executor.pipeline_dag.nodes.values():
            self.assertAlmostEqual(node.get('actual_duration', 0), 2.0,
                                   places=1)

    def test_post_execute_with_per_op_metrics(self):
        self.executor._pre_execute_operations_with_dag_monitoring(self.ops)
        metrics = {
            'duration': 5.0,
            'input_rows': 100,
            'output_rows': 80,
            'per_op_metrics': [
                {'duration': 3.0, 'input_rows': 100, 'output_rows': 90},
                {'duration': 2.0, 'input_rows': 90, 'output_rows': 80},
            ],
        }
        self.executor._post_execute_operations_with_dag_monitoring(
            self.ops, metrics=metrics)
        nodes = list(self.executor.pipeline_dag.nodes.values())
        self.assertAlmostEqual(nodes[0].get('actual_duration', 0), 3.0,
                               places=1)
        self.assertAlmostEqual(nodes[1].get('actual_duration', 0), 2.0,
                               places=1)

    def test_monitoring_noop_without_dag(self):
        executor = ConcreteDAGExecutor()
        executor._pre_execute_operations_with_dag_monitoring(self.ops)
        executor._post_execute_operations_with_dag_monitoring(self.ops)


class DAGExecutionMixinLogContextTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()
        self.executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=True)
        self.ops = [MockOperation('text_length_filter', 'filter')]
        self.executor._initialize_dag_execution(cfg, self.ops)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_log_op_start_with_dag_context(self):
        self.executor._log_operation_with_dag_context(
            'text_length_filter', 0, 'op_start', partition_id=0)
        starts = [e for e in self.executor._logged_events
                  if e[0] == 'op_start']
        self.assertEqual(len(starts), 1)

    def test_log_op_complete_with_dag_context(self):
        self.executor._log_operation_with_dag_context(
            'text_length_filter', 0, 'op_complete', partition_id=0,
            duration=1.5, input_rows=100, output_rows=90)
        completes = [e for e in self.executor._logged_events
                     if e[0] == 'op_complete']
        self.assertEqual(len(completes), 1)

    def test_log_op_failed_with_dag_context(self):
        self.executor._log_operation_with_dag_context(
            'text_length_filter', 0, 'op_failed', partition_id=0,
            error='something went wrong', retry_count=1)
        fails = [e for e in self.executor._logged_events
                 if e[0] == 'op_failed']
        self.assertEqual(len(fails), 1)


class DAGExecutionMixinExtractTypesTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.executor = ConcreteDAGExecutor()

    def test_extract_types_from_names(self):
        ops = [
            MockOperation('text_length_filter'),
            MockOperation('clean_email_mapper'),
            MockOperation('minhash_deduplicator'),
            MockOperation('topk_selector'),
            MockOperation('key_value_grouper'),
            MockOperation('nested_aggregator'),
        ]
        types = self.executor._extract_operation_types_from_ops(ops)
        self.assertIn('filter', types)
        self.assertIn('mapper', types)
        self.assertIn('deduplicator', types)
        self.assertIn('selector', types)
        self.assertIn('grouper', types)
        self.assertIn('aggregator', types)

    def test_extract_types_empty_ops(self):
        types = self.executor._extract_operation_types_from_ops([])
        self.assertEqual(types, [])


class DAGExecutionMixinStatusTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_status_not_initialized(self):
        executor = ConcreteDAGExecutor()
        status = executor.get_dag_execution_status()
        self.assertEqual(status['status'], 'not_initialized')

    def test_status_after_init(self):
        executor = ConcreteDAGExecutor()
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter')]
        executor._initialize_dag_execution(cfg, ops)
        status = executor.get_dag_execution_status()
        self.assertIn('summary', status)
        self.assertIn('execution_plan_length', status)

    def test_visualize_not_initialized(self):
        executor = ConcreteDAGExecutor()
        result = executor.visualize_dag_execution_plan()
        self.assertEqual(result, 'Pipeline DAG not initialized')

    def test_visualize_after_init(self):
        executor = ConcreteDAGExecutor()
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter')]
        executor._initialize_dag_execution(cfg, ops)
        result = executor.visualize_dag_execution_plan()
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, 'Pipeline DAG not initialized')

    def test_get_execution_plan_path_no_dag(self):
        executor = ConcreteDAGExecutor()
        executor.cfg = MockConfig(self.work_dir)
        path = executor.get_dag_execution_plan_path()
        self.assertIn('dag_execution_plan.json', path)

    def test_get_execution_plan_path_with_dag(self):
        executor = ConcreteDAGExecutor()
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter')]
        executor._initialize_dag_execution(cfg, ops)
        path = executor.get_dag_execution_plan_path()
        self.assertIn('dag_execution_plan.json', path)
        self.assertIn(self.work_dir, path)


class DAGExecutionMixinReconstructTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_reconstruct_without_event_logger(self):
        executor = ConcreteDAGExecutor()
        result = executor.reconstruct_dag_state_from_events('job_123')
        self.assertIsNone(result)

    def test_reconstruct_with_saved_plan(self):
        executor = ConcreteDAGExecutor()
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter'),
               MockOperation('clean_mapper')]
        executor._initialize_dag_execution(cfg, ops)

        class MockEventLogger:
            def get_events(self, event_type=None):
                return []

        executor.event_logger = MockEventLogger()
        result = executor.reconstruct_dag_state_from_events('job_123')
        self.assertIsNotNone(result)
        self.assertEqual(result['job_id'], 'job_123')
        self.assertIn('node_states', result)
        self.assertIn('statistics', result)
        self.assertIn('resumption', result)

    def test_reconstruct_no_plan_file(self):
        executor = ConcreteDAGExecutor()
        executor.pipeline_dag = PipelineDAG(self.work_dir)

        class MockEventLogger:
            def get_events(self, event_type=None):
                return []

        executor.event_logger = MockEventLogger()
        result = executor.reconstruct_dag_state_from_events('job_456')
        self.assertIsNone(result)

    def test_initialize_node_states_from_plan(self):
        executor = ConcreteDAGExecutor()
        dag_plan = {
            'nodes': {
                'op_001_filter': {
                    'op_name': 'filter',
                    'op_type': 'filter',
                    'execution_order': 0,
                    'dependencies': [],
                    'dependents': ['op_002_mapper'],
                },
                'op_002_mapper': {
                    'op_name': 'mapper',
                    'op_type': 'mapper',
                    'execution_order': 1,
                    'dependencies': ['op_001_filter'],
                    'dependents': [],
                },
            }
        }
        states = executor._initialize_node_states_from_plan(dag_plan)
        self.assertEqual(len(states), 2)
        self.assertEqual(states['op_001_filter']['status'],
                         DAGNodeStatus.PENDING.value)
        self.assertEqual(states['op_002_mapper']['dependencies'],
                         ['op_001_filter'])

    def test_calculate_dag_statistics(self):
        executor = ConcreteDAGExecutor()
        node_states = {
            'n1': {'status': DAGNodeStatus.COMPLETED.value},
            'n2': {'status': DAGNodeStatus.COMPLETED.value},
            'n3': {'status': DAGNodeStatus.FAILED.value},
            'n4': {'status': DAGNodeStatus.RUNNING.value},
            'n5': {'status': DAGNodeStatus.PENDING.value},
        }
        stats = executor._calculate_dag_statistics(node_states)
        self.assertEqual(stats['total_nodes'], 5)
        self.assertEqual(stats['completed_nodes'], 2)
        self.assertEqual(stats['failed_nodes'], 1)
        self.assertEqual(stats['running_nodes'], 1)
        self.assertEqual(stats['pending_nodes'], 1)
        self.assertAlmostEqual(stats['completion_percentage'], 40.0)

    def test_find_ready_nodes(self):
        executor = ConcreteDAGExecutor()
        node_states = {
            'n1': {'status': DAGNodeStatus.COMPLETED.value,
                   'dependencies': []},
            'n2': {'status': DAGNodeStatus.PENDING.value,
                   'dependencies': ['n1']},
            'n3': {'status': DAGNodeStatus.PENDING.value,
                   'dependencies': ['n2']},
        }
        ready = executor._find_ready_nodes(node_states)
        self.assertEqual(ready, ['n2'])

    def test_find_ready_nodes_all_completed(self):
        executor = ConcreteDAGExecutor()
        node_states = {
            'n1': {'status': DAGNodeStatus.COMPLETED.value,
                   'dependencies': []},
            'n2': {'status': DAGNodeStatus.COMPLETED.value,
                   'dependencies': ['n1']},
        }
        ready = executor._find_ready_nodes(node_states)
        self.assertEqual(ready, [])

    def test_determine_resumption_from_failed(self):
        executor = ConcreteDAGExecutor()
        node_states = {
            'n1': {'status': DAGNodeStatus.COMPLETED.value,
                   'execution_order': 0},
            'n2': {'status': DAGNodeStatus.FAILED.value,
                   'execution_order': 1},
            'n3': {'status': DAGNodeStatus.PENDING.value,
                   'execution_order': 2},
        }
        stats = {'failed_nodes': 1, 'running_nodes': 0,
                 'completed_nodes': 1, 'total_nodes': 3}
        ready = ['n3']
        result = executor._determine_resumption_strategy(
            node_states, ready, stats)
        self.assertTrue(result['can_resume'])
        self.assertEqual(result['resume_from_node'], 'n2')

    def test_determine_resumption_from_running(self):
        executor = ConcreteDAGExecutor()
        node_states = {
            'n1': {'status': DAGNodeStatus.COMPLETED.value,
                   'execution_order': 0},
            'n2': {'status': DAGNodeStatus.RUNNING.value,
                   'execution_order': 1},
        }
        stats = {'failed_nodes': 0, 'running_nodes': 1,
                 'completed_nodes': 1, 'total_nodes': 2}
        result = executor._determine_resumption_strategy(
            node_states, [], stats)
        self.assertTrue(result['can_resume'])
        self.assertEqual(result['resume_from_node'], 'n2')

    def test_determine_resumption_all_completed(self):
        executor = ConcreteDAGExecutor()
        node_states = {
            'n1': {'status': DAGNodeStatus.COMPLETED.value,
                   'execution_order': 0},
            'n2': {'status': DAGNodeStatus.COMPLETED.value,
                   'execution_order': 1},
        }
        stats = {'failed_nodes': 0, 'running_nodes': 0,
                 'completed_nodes': 2, 'total_nodes': 2}
        result = executor._determine_resumption_strategy(
            node_states, [], stats)
        self.assertFalse(result['can_resume'])

    def test_update_node_states_from_events(self):
        executor = ConcreteDAGExecutor()
        from data_juicer.core.executor.event_logging_mixin import EventType
        node_states = {
            'n1': {
                'status': DAGNodeStatus.PENDING.value,
                'start_time': None,
                'end_time': None,
                'actual_duration': 0.0,
                'error_message': None,
            },
        }
        events = [
            {
                'event_type': EventType.DAG_NODE_START.value,
                'metadata': {'dag_node_id': 'n1'},
                'timestamp': 1000.0,
            },
            {
                'event_type': EventType.DAG_NODE_COMPLETE.value,
                'metadata': {'dag_node_id': 'n1'},
                'timestamp': 1005.0,
                'duration': 5.0,
            },
        ]
        executor._update_node_states_from_events(node_states, events)
        self.assertEqual(node_states['n1']['status'],
                         DAGNodeStatus.COMPLETED.value)
        self.assertEqual(node_states['n1']['actual_duration'], 5.0)

    def test_handle_operation_event_with_dag_context(self):
        executor = ConcreteDAGExecutor()
        from data_juicer.core.executor.event_logging_mixin import EventType
        node_states = {
            'n1': {
                'status': DAGNodeStatus.PENDING.value,
                'start_time': None,
                'end_time': None,
                'actual_duration': 0.0,
                'error_message': None,
            },
        }
        event = {
            'event_type': EventType.OP_FAILED.value,
            'metadata': {'dag_context': {'dag_node_id': 'n1'}},
            'timestamp': 2000.0,
            'duration': 1.0,
            'error_message': 'OutOfMemory',
        }
        executor._handle_operation_event(event, node_states)
        self.assertEqual(node_states['n1']['status'],
                         DAGNodeStatus.FAILED.value)
        self.assertEqual(node_states['n1']['error_message'], 'OutOfMemory')


class DAGExecutionMixinCurrentNodePropertyTest(DataJuicerTestCaseBase):

    def test_current_dag_node_thread_local(self):
        executor = ConcreteDAGExecutor()
        results = {}

        def set_node(name):
            executor.current_dag_node = name
            time.sleep(0.01)
            results[threading.current_thread().name] = executor.current_dag_node

        t1 = threading.Thread(target=set_node, args=('node_a',), name='t1')
        t2 = threading.Thread(target=set_node, args=('node_b',), name='t2')
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertEqual(results['t1'], 'node_a')
        self.assertEqual(results['t2'], 'node_b')


class DAGExecutionMixinStrategyTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.work_dir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_create_non_partitioned_strategy(self):
        executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir)
        strategy = executor._create_execution_strategy(cfg)
        self.assertIsInstance(strategy, NonPartitionedDAGStrategy)

    def test_create_partitioned_strategy(self):
        executor = ConcreteDAGExecutor(executor_type='ray_partitioned',
                                       num_partitions=4)
        cfg = MockConfig(self.work_dir)
        strategy = executor._create_execution_strategy(cfg)
        self.assertIsInstance(strategy, PartitionedDAGStrategy)

    def test_get_dag_node_for_operation(self):
        executor = ConcreteDAGExecutor(executor_type='default')
        cfg = MockConfig(self.work_dir, use_dag=True)
        ops = [MockOperation('text_filter')]
        executor._initialize_dag_execution(cfg, ops)
        node_id = executor._get_dag_node_for_operation(
            'text_filter', 0, partition_id=0)
        self.assertIsNotNone(node_id)

    def test_get_dag_node_without_strategy(self):
        executor = ConcreteDAGExecutor()
        result = executor._get_dag_node_for_operation('op', 0)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
