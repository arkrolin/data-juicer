"""Tests to cover gaps in data_juicer/utils/job/snapshot.py (lines 128-143, 550-559, 643-728)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_juicer.utils.job.snapshot import (
    ProcessingSnapshotAnalyzer,
    ProcessingStatus,
    OperationStatus,
    PartitionStatus,
    JobSnapshot,
    create_snapshot,
    main,
)


class TestLoadEventsErrors(unittest.TestCase):
    """Cover lines 128-129: events file exists but can't be read."""

    def test_load_events_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events_file = Path(tmpdir) / "events.jsonl"
            events_file.write_text('{"event_type": "job_start"}\n')
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            with patch('builtins.open', side_effect=PermissionError("denied")):
                result = analyzer.load_events()
            self.assertEqual(result, [])

    def test_load_events_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer.load_events()
            self.assertEqual(result, [])


class TestLoadDagPlanErrors(unittest.TestCase):
    """Cover lines 138-143: dag file errors."""

    def test_dag_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer.load_dag_plan()
            self.assertEqual(result, {})

    def test_dag_file_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "dag_execution_plan.json").write_text("not json{")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer.load_dag_plan()
            self.assertEqual(result, {})


class TestLoadJobSummaryErrors(unittest.TestCase):
    """Cover lines 156-157: job summary errors."""

    def test_job_summary_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer.load_job_summary()
            self.assertEqual(result, {})

    def test_job_summary_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "job_summary.json").write_text("corrupt")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer.load_job_summary()
            self.assertEqual(result, {})


class TestOperationProgressInProgress(unittest.TestCase):
    """Cover lines 550-559: in-progress operation with start_time."""

    def test_in_progress_with_start_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            import time
            op = OperationStatus(
                operation_name="test_op",
                operation_idx=0,
                status=ProcessingStatus.IN_PROGRESS,
                start_time=time.time() - 0.5,
            )
            progress = analyzer._calculate_operation_progress(op)
            self.assertGreaterEqual(progress, 10.0)
            self.assertLessEqual(progress, 90.0)

    def test_in_progress_no_start_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            op = OperationStatus(
                operation_name="test_op",
                operation_idx=0,
                status=ProcessingStatus.IN_PROGRESS,
                start_time=None,
            )
            progress = analyzer._calculate_operation_progress(op)
            self.assertEqual(progress, 10.0)

    def test_checkpointed_operation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            op = OperationStatus(
                operation_name="test_op",
                operation_idx=0,
                status=ProcessingStatus.CHECKPOINTED,
            )
            progress = analyzer._calculate_operation_progress(op)
            self.assertEqual(progress, 100.0)

    def test_not_started_operation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            op = OperationStatus(
                operation_name="test_op",
                operation_idx=0,
                status=ProcessingStatus.NOT_STARTED,
            )
            progress = analyzer._calculate_operation_progress(op)
            self.assertEqual(progress, 0.0)


class TestFormatDuration(unittest.TestCase):
    """Cover line 590+: _format_duration helper."""

    def test_hours(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer._format_duration(3661)
            self.assertEqual(result, "1h 1m 1s")

    def test_minutes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer._format_duration(125)
            self.assertEqual(result, "2m 5s")

    def test_seconds_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer._format_duration(45)
            self.assertEqual(result, "45s")

    def test_none_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            result = analyzer._format_duration(None)
            self.assertIsNone(result)


class TestMainFunction(unittest.TestCase):
    """Cover lines 643-728: the CLI main() function."""

    def test_main_missing_dir(self):
        with patch('sys.argv', ['snapshot', '/tmp/nonexistent_snapshot_dir_xyz']):
            result = main()
            self.assertEqual(result, 1)

    def test_main_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 1000, "metadata": {"checkpoint_strategy": "every_op"}},
                {"event_type": "partition_creation_start", "partition_id": 0, "timestamp": 1001},
                {"event_type": "partition_creation_complete", "partition_id": 0, "timestamp": 1002, "metadata": {"sample_count": 100}},
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1003},
                {"event_type": "op_start", "partition_id": 0, "operation_idx": 0, "operation_name": "filter1", "timestamp": 1004},
                {"event_type": "op_complete", "partition_id": 0, "operation_idx": 0, "operation_name": "filter1", "timestamp": 1005, "metadata": {"input_rows": 100, "output_rows": 80}},
                {"event_type": "checkpoint_save", "partition_id": 0, "operation_idx": 0, "operation_name": "filter1", "timestamp": 1006},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 1007},
                {"event_type": "job_complete", "timestamp": 1008},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            with patch('sys.argv', ['snapshot', tmpdir]):
                result = main()
            self.assertEqual(result, 0)

    def test_main_human_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 1000, "metadata": {"checkpoint_strategy": "every_op"}},
                {"event_type": "partition_creation_start", "partition_id": 0, "timestamp": 1001},
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1002},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 1003},
                {"event_type": "job_complete", "timestamp": 1004},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            summary = {"status": "completed", "start_time": 1000, "end_time": 1004, "duration": 4}
            (Path(tmpdir) / "job_summary.json").write_text(json.dumps(summary))
            with patch('sys.argv', ['snapshot', tmpdir, '--human-readable']):
                result = main()
            self.assertEqual(result, 0)

    def test_main_exception_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid events file but corrupt dag file to trigger partial error
            (Path(tmpdir) / "events.jsonl").write_text("")
            with patch('sys.argv', ['snapshot', tmpdir]):
                with patch('data_juicer.utils.job.snapshot.ProcessingSnapshotAnalyzer.generate_snapshot',
                           side_effect=Exception("test error")):
                    result = main()
            self.assertEqual(result, 1)

    def test_main_human_readable_with_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 1000, "metadata": {"checkpoint_strategy": "every_op"}},
                {"event_type": "partition_creation_start", "partition_id": 0, "timestamp": 1001},
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1002},
                {"event_type": "op_start", "partition_id": 0, "operation_idx": 0, "operation_name": "filter1", "timestamp": 1003},
                {"event_type": "op_failed", "partition_id": 0, "operation_idx": 0, "operation_name": "filter1", "error_message": "OOM", "timestamp": 1004},
                {"event_type": "partition_failed", "partition_id": 0, "error_message": "OOM", "timestamp": 1005},
                {"event_type": "checkpoint_save", "partition_id": 1, "operation_idx": 0, "operation_name": "filter1", "timestamp": 1006},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            with patch('sys.argv', ['snapshot', tmpdir, '--human-readable']):
                result = main()
            self.assertEqual(result, 0)


class TestGenerateSnapshotWithJobSummary(unittest.TestCase):
    """Cover generate_snapshot paths using job_summary for timing."""

    def test_snapshot_uses_job_summary_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 1000, "metadata": {}},
                {"event_type": "job_complete", "timestamp": 2000},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            summary = {"start_time": 1000, "end_time": 2000, "duration": 1000}
            (Path(tmpdir) / "job_summary.json").write_text(json.dumps(summary))
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            snapshot = analyzer.generate_snapshot()
            self.assertEqual(snapshot.total_duration, 1000)

    def test_snapshot_fallback_to_events_timing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 500, "metadata": {}},
                {"event_type": "job_complete", "timestamp": 600},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            snapshot = analyzer.generate_snapshot()
            self.assertEqual(snapshot.job_start_time, 500)
            self.assertEqual(snapshot.job_end_time, 600)
            self.assertEqual(snapshot.total_duration, 100)


class TestFindLatestEventsFile(unittest.TestCase):
    """Cover _find_latest_events_file with timestamped files."""

    def test_picks_latest_by_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import time
            f1 = Path(tmpdir) / "events_20250101.jsonl"
            f1.write_text("")
            time.sleep(0.05)
            f2 = Path(tmpdir) / "events_20250102.jsonl"
            f2.write_text("")
            analyzer = ProcessingSnapshotAnalyzer(tmpdir)
            self.assertEqual(analyzer.events_file.name, "events_20250102.jsonl")


if __name__ == '__main__':
    unittest.main()
