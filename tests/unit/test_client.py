"""Unit tests for SyncClient internal logic.

These tests mock SSHManager and use a real JobStore (pointed at tmp_path)
to verify client-level behaviour: reconciliation, caching, store recording.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from slurp.client import SyncClient
from slurp.domain import (
    Job,
    JobRecord,
    JobResult,
    JobStatus,
    Profile,
    ResourceRequest,
    SlurmJobInfo,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def profile() -> Profile:
    return Profile(
        name="test",
        hostname="hpc.local",
        username="user",
        partition="gpu",
        account="lab",
        sync=Profile.SyncConfig(local=".", remote="/remote/project"),
    )


@pytest.fixture
def mock_ssh() -> MagicMock:
    """Mock SSHManager so no real network calls happen."""
    m = MagicMock()
    m.run = AsyncMock()
    m._ensure_control_master = MagicMock()
    m.control_master_socket.return_value = Path("/tmp/fake.sock")
    m.close = MagicMock()
    return m


@pytest.fixture
def client(
    profile: Profile, mock_ssh: MagicMock, tmp_path: Path
) -> SyncClient:
    """A SyncClient with mocked SSH and a tmp-path store."""
    store_path = tmp_path / "jobs.json"
    offset_path = tmp_path / "log_offsets.json"
    with (
        patch("slurp.client.SSHManager", return_value=mock_ssh),
        patch("slurp.core.store.DEFAULT_STORE_PATH", store_path),
        patch("slurp.core.store.DEFAULT_OFFSET_PATH", offset_path),
    ):
        c = SyncClient(profile=profile)
        return c


def _make_record(
    job_id: str = "100",
    status: JobStatus = JobStatus.PENDING,
    name: str = "train",
    experiment: str | None = None,
) -> JobRecord:
    return JobRecord(
        job_id=job_id,
        name=name,
        status=status,
        profile="test",
        experiment=experiment,
        submitted_at=datetime.now(UTC),
        command="python train.py",
        resources=ResourceRequest(),
        working_dir="/remote",
    )


# ── _reconcile_store ──────────────────────────────────────────────────


class TestReconcileStore:
    """Tests for SLURM reconciliation of non-terminal jobs."""

    def test_updates_non_terminal_jobs(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """Non-terminal jobs should be queried via sacct and store updated."""
        client._store.append_job(_make_record("100", JobStatus.PENDING))
        client._store.append_job(_make_record("200", JobStatus.RUNNING))
        client._store.append_job(_make_record("300", JobStatus.COMPLETED))

        # Mock sacct_query to return updated statuses
        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {
                "100": SlurmJobInfo(job_id="100", state="RUNNING", exit_code="0:0"),
                "200": SlurmJobInfo(job_id="200", state="COMPLETED", exit_code="0:0"),
            }

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            records = client._reconcile_store()

        # Non-terminal jobs updated
        assert records["100"].status == JobStatus.RUNNING
        assert records["200"].status == JobStatus.COMPLETED
        # Terminal job not in sacct response — stays as-is
        assert records["300"].status == JobStatus.COMPLETED
        # Store was updated
        stored = client._store.get_job("100")
        assert stored is not None
        assert stored.status == JobStatus.RUNNING

    def test_does_not_query_terminal_jobs(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """Terminal jobs should not be included in the sacct query."""
        client._store.append_job(_make_record("100", JobStatus.COMPLETED))
        client._store.append_job(_make_record("200", JobStatus.FAILED))

        captured_ids: list[str] = []

        def fake_sacct(profile, job_ids, *, ssh_manager):
            captured_ids.extend(job_ids)
            return {}

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            client._reconcile_store()

        assert captured_ids == []

    def test_falls_back_on_ssh_failure(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """If sacct query fails, local cache is returned unchanged."""
        client._store.append_job(_make_record("100", JobStatus.PENDING))

        with patch("slurp.client.sacct_query", side_effect=Exception("SSH down")):
            records = client._reconcile_store()

        # Local cache returned as-is
        assert records["100"].status == JobStatus.PENDING

    def test_empty_store_returns_empty(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        records = client._reconcile_store()
        assert records == {}


# ── list_jobs reconciliation ───────────────────────────────────────────


class TestListJobsReconciliation:
    """Verify list_jobs reconciles before returning."""

    def test_list_jobs_refreshes_statuses(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        client._store.append_job(_make_record("100", JobStatus.PENDING))

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {
                "100": SlurmJobInfo(job_id="100", state="RUNNING", exit_code="0:0"),
            }

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            jobs = client.list_jobs()

        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.RUNNING

    def test_list_jobs_filters_by_experiment(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        client._store.append_job(
            _make_record("100", JobStatus.PENDING, experiment="exp-a")
        )
        client._store.append_job(
            _make_record("200", JobStatus.PENDING, experiment="exp-b")
        )

        with patch("slurp.client.sacct_query", return_value={}):
            jobs = client.list_jobs(experiment="exp-a")

        assert len(jobs) == 1
        assert jobs[0].job_id == "100"


# ── status() refresh ──────────────────────────────────────────────────


class TestStatusRefresh:
    """Verify status() refreshes from SLURM when job is in local store."""

    def test_status_refreshes_from_slurm(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        client._store.append_job(_make_record("100", JobStatus.PENDING))

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {
                "100": SlurmJobInfo(job_id="100", state="RUNNING", exit_code="0:0"),
            }

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            job = client.status("100")

        assert job is not None
        assert job.status == JobStatus.RUNNING
        # Store was updated too
        stored = client._store.get_job("100")
        assert stored is not None
        assert stored.status == JobStatus.RUNNING

    def test_status_not_found(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        mock_ssh.run.return_value = (0, "", "")

        with patch("slurp.client.sacct_query", return_value={}):
            job = client.status("999")

        assert job is None


# ── job_result() cache ────────────────────────────────────────────────


class TestJobResultCache:
    """Verify job_result() uses the cache populated by wait_job()."""

    def test_returns_cached_result_without_querying(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        job = Job(
            job_id="100",
            name="train",
            status=JobStatus.COMPLETED,
            profile="test",
            working_dir="/remote",
        )
        cached = JobResult(
            job_id="100",
            status=JobStatus.COMPLETED,
            exit_code=0,
            stdout="done",
        )
        job._result_cache["100"] = cached

        # If the cache is used, wait_job should never be called
        with patch.object(client, "wait_job") as mock_wait:
            result = client.job_result(job)
            mock_wait.assert_not_called()

        assert result is cached

    def test_calls_wait_job_when_cache_empty(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        job = Job(
            job_id="100",
            name="train",
            status=JobStatus.RUNNING,
            profile="test",
            working_dir="/remote",
        )

        expected = JobResult(
            job_id="100",
            status=JobStatus.COMPLETED,
            exit_code=0,
        )
        with patch.object(client, "wait_job", return_value=expected) as mock_wait:
            result = client.job_result(job)
            mock_wait.assert_called_once_with(job)

        assert result is expected


# ── submit_array() store recording ────────────────────────────────────


class TestSubmitArrayStore:
    """Verify submit_array records the job in the local store."""

    def test_array_job_recorded_in_store(
        self, client: SyncClient, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        # Mock sbatch submission and sync directly (bypass SSH layer)
        async def fake_sbatch(profile, script, *, working_dir, ssh_manager):
            return "50000"

        with (
            patch("slurp.client.sbatch_submit", side_effect=fake_sbatch),
            patch("slurp.client.sync_to_remote", new_callable=AsyncMock),
        ):
            array = client.submit_array(
                "python train.py --seed {seed}",
                configs=[{"seed": str(i)} for i in range(3)],
                gpus=1,
            )

        assert array.array_job_id == "50000"
        # Verify the job is in the local store
        record = client._store.get_job("50000")
        assert record is not None
        assert record.name == "array"
        assert record.status == JobStatus.PENDING
        assert record.profile == "test"
