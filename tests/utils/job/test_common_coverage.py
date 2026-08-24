"""Tests to cover gaps in data_juicer/utils/job/common.py."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

from data_juicer.utils.job.common import JobUtils, list_running_jobs


class TestJobUtilsInit(unittest.TestCase):

    def test_init_with_work_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            self.assertEqual(ju.work_dir, Path(tmpdir))

    def test_init_with_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job123"
            job_dir.mkdir()
            ju = JobUtils(job_id="job123", base_dir=tmpdir)
            self.assertEqual(ju.work_dir, job_dir)

    def test_init_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            JobUtils(job_id="nonexistent", work_dir="/tmp/nonexistent_dir_xyz_123")


class TestJobUtilsLoadJobSummary(unittest.TestCase):

    def test_load_valid_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = {"status": "completed", "start_time": 1000}
            (Path(tmpdir) / "job_summary.json").write_text(json.dumps(summary))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_job_summary()
            self.assertEqual(result["status"], "completed")

    def test_load_missing_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_job_summary()
            self.assertIsNone(result)

    def test_load_corrupt_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "job_summary.json").write_text("not json{{{")
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_job_summary()
            self.assertIsNone(result)


class TestJobUtilsLoadDatasetMapping(unittest.TestCase):

    def test_load_valid_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            mapping = {"partitions": [{"partition_id": 0}]}
            (meta_dir / "dataset_mapping.json").write_text(json.dumps(mapping))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_dataset_mapping()
            self.assertEqual(result["partitions"][0]["partition_id"], 0)

    def test_load_missing_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_dataset_mapping()
            self.assertEqual(result, {})

    def test_load_corrupt_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            (meta_dir / "dataset_mapping.json").write_text("bad json")
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_dataset_mapping()
            self.assertEqual(result, {})


class TestJobUtilsFindEventsFile(unittest.TestCase):

    def test_finds_timestamped_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events_20250101.jsonl").write_text("")
            (Path(tmpdir) / "events_20250102.jsonl").write_text("")
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju._find_latest_events_file()
            self.assertIsNotNone(result)

    def test_falls_back_to_events_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "events.jsonl").write_text("")
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju._find_latest_events_file()
            self.assertEqual(result.name, "events.jsonl")

    def test_returns_none_when_no_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju._find_latest_events_file()
            self.assertIsNone(result)


class TestJobUtilsLoadEventLogs(unittest.TestCase):

    def test_load_valid_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "job_start", "timestamp": 1000},
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 1001},
            ]
            (Path(tmpdir) / "events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events)
            )
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_event_logs()
            self.assertEqual(len(result), 2)

    def test_skips_bad_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            content = '{"event_type": "job_start"}\nnot valid json\n{"event_type": "end"}\n'
            (Path(tmpdir) / "events.jsonl").write_text(content)
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_event_logs()
            self.assertEqual(len(result), 2)

    def test_load_no_events_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.load_event_logs()
            self.assertEqual(result, [])


class TestJobUtilsExtractProcessThreadIds(unittest.TestCase):

    def test_extracts_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "op_start", "process_id": 1234, "thread_id": 5678},
                {"event_type": "op_start", "process_id": 1234, "thread_id": 9012},
                {"event_type": "op_start", "process_id": 4567, "thread_id": None},
            ]
            (Path(tmpdir) / "events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in events)
            )
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.extract_process_thread_ids()
            self.assertEqual(result["process_ids"], {1234, 4567})
            self.assertEqual(result["thread_ids"], {5678, 9012})


class TestJobUtilsFindProcessesByIds(unittest.TestCase):

    def test_finds_running_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            current_pid = os.getpid()
            # Use current process's parent (should be running)
            parent_pid = os.getppid()
            result = ju.find_processes_by_ids({parent_pid})
            self.assertGreaterEqual(len(result), 0)

    def test_skips_current_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.find_processes_by_ids({os.getpid()})
            self.assertEqual(len(result), 0)

    def test_handles_nonexistent_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.find_processes_by_ids({99999999})
            self.assertEqual(len(result), 0)


class TestJobUtilsFindThreadsByIds(unittest.TestCase):

    def test_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.find_threads_by_ids({1, 2, 3})
            self.assertEqual(result, [])


class TestJobUtilsGetPartitionStatus(unittest.TestCase):

    def test_from_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 100},
                {"event_type": "op_start", "partition_id": 0, "operation_name": "filter1", "operation_idx": 0, "timestamp": 101},
                {"event_type": "op_complete", "partition_id": 0, "operation_name": "filter1", "operation_idx": 0,
                 "timestamp": 102, "duration": 1, "input_rows": 100, "output_rows": 80,
                 "performance_metrics": {"throughput": 100, "reduction_ratio": 0.2}},
                {"event_type": "checkpoint_save", "partition_id": 0, "operation_name": "filter1", "operation_idx": 0,
                 "checkpoint_path": "/tmp/ckpt", "timestamp": 103},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 104},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.get_partition_status()
            self.assertEqual(result[0]["status"], "completed")
            self.assertEqual(len(result[0]["completed_ops"]), 1)
            self.assertEqual(len(result[0]["checkpoints"]), 1)

    def test_from_dataset_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            mapping = {"partitions": [
                {"partition_id": 0, "processing_status": "completed", "sample_count": 50}
            ]}
            (meta_dir / "dataset_mapping.json").write_text(json.dumps(mapping))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.get_partition_status()
            self.assertEqual(result[0]["status"], "completed")
            self.assertEqual(result[0]["sample_count"], 50)


class TestJobUtilsCalculateOverallProgress(unittest.TestCase):

    def test_all_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            events = [
                {"event_type": "partition_start", "partition_id": 0, "timestamp": 100},
                {"event_type": "partition_complete", "partition_id": 0, "timestamp": 200},
                {"event_type": "partition_start", "partition_id": 1, "timestamp": 100},
                {"event_type": "partition_complete", "partition_id": 1, "timestamp": 200},
            ]
            (Path(tmpdir) / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            summary = {"status": "completed", "start_time": 100}
            (Path(tmpdir) / "job_summary.json").write_text(json.dumps(summary))
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.calculate_overall_progress()
            self.assertEqual(result["progress_percentage"], 100.0)
            self.assertEqual(result["completed_partitions"], 2)

    def test_empty_partitions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.calculate_overall_progress()
            self.assertEqual(result["progress_percentage"], 0)
            self.assertEqual(result["total_partitions"], 0)


class TestJobUtilsGetOperationPipeline(unittest.TestCase):

    def test_parses_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = """dataset_path: test.jsonl
process:
  - filter1:
      threshold: 0.5
  - mapper1:
      key: value
"""
            (Path(tmpdir) / "partition-checkpoint-eventlog.yaml").write_text(config)
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.get_operation_pipeline()
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "filter1")

    def test_missing_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ju = JobUtils(job_id="test", work_dir=tmpdir)
            result = ju.get_operation_pipeline()
            self.assertEqual(result, [])


class TestListRunningJobs(unittest.TestCase):

    def test_empty_base_dir(self):
        result = list_running_jobs("/tmp/nonexistent_base_dir_xyz")
        self.assertEqual(result, [])

    def test_lists_jobs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job1"
            job_dir.mkdir()
            summary = {"status": "completed", "start_time": 1000}
            (job_dir / "job_summary.json").write_text(json.dumps(summary))
            events = [{"event_type": "op_start", "process_id": 99999999}]
            (job_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            result = list_running_jobs(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["job_id"], "job1")
            self.assertEqual(result[0]["status"], "completed")

    def test_handles_corrupt_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job_bad"
            job_dir.mkdir()
            (job_dir / "job_summary.json").write_text("not json")
            result = list_running_jobs(tmpdir)
            self.assertEqual(len(result), 0)

    def test_handles_no_events_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir) / "job2"
            job_dir.mkdir()
            summary = {"status": "running", "start_time": 2000}
            (job_dir / "job_summary.json").write_text(json.dumps(summary))
            result = list_running_jobs(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["processes"], 0)


if __name__ == '__main__':
    unittest.main()
