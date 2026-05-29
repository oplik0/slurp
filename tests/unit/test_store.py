"""Unit tests for the atomic job store."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from slurp.core.store import JobStore, LogOffsetStore
from slurp.domain import JobRecord, JobStatus, ResourceRequest


class TestJobStore:
    def test_read_empty(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        data = store.read()
        assert data == {"jobs": {}, "log_offsets": {}, "idempotency": {}}

    def test_append_and_get(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="123",
            name="test",
            status=JobStatus.PENDING,
            profile="test",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
            idempotency_hash="abc",
            submitted_at=datetime.now(UTC),
        )
        store.append_job(record)
        fetched = store.get_job("123")
        assert fetched is not None
        assert fetched.job_id == "123"
        assert fetched.name == "test"

    def test_check_idempotency(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="123",
            name="test",
            status=JobStatus.PENDING,
            profile="test",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
            idempotency_hash="abc123",
            submitted_at=datetime.now(UTC),
        )
        store.append_job(record)
        assert store.check_idempotency("abc123") == "123"
        assert store.check_idempotency("nonexistent") is None

    def test_update_status(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        record = JobRecord(
            job_id="123",
            name="test",
            status=JobStatus.PENDING,
            profile="test",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
            submitted_at=datetime.now(UTC),
        )
        store.append_job(record)
        store.update_job_status("123", JobStatus.RUNNING)
        fetched = store.get_job("123")
        assert fetched is not None
        assert fetched.status == JobStatus.RUNNING

    def test_corruption_recovery(self, tmp_path: Path) -> None:
        store = JobStore(path=tmp_path / "jobs.json")
        store.path.write_text("invalid json{{{")
        data = store.read()
        assert data == {"jobs": {}, "log_offsets": {}, "idempotency": {}}
        assert not store.path.exists()  # moved to corrupt file


class TestLogOffsetStore:
    def test_read_empty(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        assert store.read() == {}

    def test_set_and_get(self, tmp_path: Path) -> None:
        store = LogOffsetStore(path=tmp_path / "offsets.json")
        store.set_offset("123", 100, 50)
        offset = store.get_offset("123")
        assert offset.out == 100
        assert offset.err == 50
