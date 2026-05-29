"""Tests for slurp.core.store atomic JSON store."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from slurp.core.store import JobStore, LogOffsetStore
from slurp.domain import JobRecord, JobStatus, LogOffsetRecord, ResourceRequest


class TestJobStore:
    """Tests for JobStore."""

    def test_read_missing_file(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        data = store.read()
        assert data == {"jobs": {}, "log_offsets": {}, "idempotency": {}}

    def test_write_and_read_jobs(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="123",
            name="test",
            status=JobStatus.PENDING,
            profile="default",
            submitted_at=datetime.now(UTC),
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
        )
        store.write_jobs({"123": record})
        data = store.read()
        assert "123" in data["jobs"]
        assert data["jobs"]["123"]["name"] == "test"

    def test_append_job(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="456",
            name="append",
            status=JobStatus.RUNNING,
            profile="default",
            submitted_at=datetime.now(UTC),
            command="python run.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
            idempotency_hash="abc123",
        )
        store.append_job(record)
        jobs = store.list_jobs()
        assert "456" in jobs
        assert jobs["456"].name == "append"
        # Idempotency entry should exist
        data = store.read()
        assert "abc123" in data["idempotency"]

    def test_update_job_status(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="789",
            name="update",
            status=JobStatus.PENDING,
            profile="default",
            submitted_at=datetime.now(UTC),
            command="python run.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
        )
        store.append_job(record)
        store.update_job_status("789", JobStatus.COMPLETED)
        job = store.get_job("789")
        assert job is not None
        assert job.status == JobStatus.COMPLETED

    def test_get_job_missing(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        assert store.get_job("999") is None

    def test_corruption_recovery(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        # Write corrupt JSON
        store.path.write_text("{not json")
        data = store.read()
        assert data == {"jobs": {}, "log_offsets": {}, "idempotency": {}}
        # Corrupt file should be moved aside
        corrupt_files = list(tmp_path.glob("*.corrupt.*"))
        assert len(corrupt_files) == 1

    def test_locking(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        fd = store._acquire_lock()
        assert fd >= 0
        store._release_lock(fd)
        # Lock file should exist
        assert store._lock_path.exists()

    def test_atomic_write(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        store._atomic_write({"jobs": {"1": {"name": "atomic"}}})
        assert store.path.exists()
        data = json.loads(store.path.read_text())
        assert data["jobs"]["1"]["name"] == "atomic"

    def test_idempotency_check(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="100",
            name="idempotent",
            status=JobStatus.PENDING,
            profile="default",
            submitted_at=datetime.now(UTC),
            command="python run.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
            idempotency_hash="hash1",
        )
        store.append_job(record)
        # Within window and still pending
        assert store.check_idempotency("hash1", window_seconds=60.0) == "100"
        # After status changes
        store.update_job_status("100", JobStatus.COMPLETED)
        assert store.check_idempotency("hash1", window_seconds=60.0) is None
        # Unknown hash
        assert store.check_idempotency("nope", window_seconds=60.0) is None

    def test_idempotency_window_expired(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        old_time = datetime.now(UTC).replace(year=2000)
        record = JobRecord(
            job_id="101",
            name="old",
            status=JobStatus.PENDING,
            profile="default",
            submitted_at=old_time,
            command="python run.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
            idempotency_hash="oldhash",
        )
        store.append_job(record)
        assert store.check_idempotency("oldhash", window_seconds=60.0) is None

    def test_prune_idempotency(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        data = store.read()
        data["idempotency"] = {
            "fresh": {
                "job_id": "1",
                "submitted_at": datetime.now(UTC).isoformat(),
            },
            "stale": {
                "job_id": "2",
                "submitted_at": "2000-01-01T00:00:00+00:00",
            },
            "bad": {
                "job_id": "3",
                "submitted_at": "not-a-date",
            },
        }
        store._atomic_write(data)
        store._prune_idempotency(data)
        assert "fresh" in data["idempotency"]
        assert "stale" not in data["idempotency"]
        assert "bad" not in data["idempotency"]


class TestLogOffsetStore:
    """Tests for LogOffsetStore."""

    def test_read_missing(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        assert store.read() == {}

    def test_write_and_read(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        offsets = {
            "job1": LogOffsetRecord(out=100, err=50),
        }
        store.write(offsets)
        data = store.read()
        assert "job1" in data
        assert data["job1"].out == 100
        assert data["job1"].err == 50

    def test_get_offset_missing(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        offset = store.get_offset("missing")
        assert offset.out == 0
        assert offset.err == 0

    def test_set_offset(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        store.set_offset("job2", 200, 75)
        offset = store.get_offset("job2")
        assert offset.out == 200
        assert offset.err == 75
        assert offset.last_read is not None

    def test_corruption_recovery(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        store.path.write_text("{bad json")
        data = store.read()
        assert data == {}

    def test_write_best_effort(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        # Write should not crash even on errors
        offsets = {"job1": LogOffsetRecord(out=10, err=5)}
        store.write(offsets)
        assert store.read() == offsets

    def test_concurrent_safe(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate write failure to verify best-effort behavior."""
        store = LogOffsetStore(path=tmp_path / "offsets.json")

        def fake_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
            raise OSError("disk full")

        monkeypatch.setattr(os, "fdopen", fake_mkstemp)
        # Should not raise
        offsets = {"job1": LogOffsetRecord(out=10, err=5)}
        store.write(offsets)
