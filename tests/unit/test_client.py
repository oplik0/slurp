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
from slurp.errors import SyncError

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
def client(profile: Profile, mock_ssh: MagicMock, tmp_path: Path) -> SyncClient:
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

    def test_updates_non_terminal_jobs(self, client: SyncClient, mock_ssh: MagicMock) -> None:
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

    def test_does_not_query_terminal_jobs(self, client: SyncClient, mock_ssh: MagicMock) -> None:
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

    def test_falls_back_on_ssh_failure(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """If sacct query fails, local cache is returned unchanged."""
        client._store.append_job(_make_record("100", JobStatus.PENDING))

        with patch("slurp.client.sacct_query", side_effect=Exception("SSH down")):
            records = client._reconcile_store()

        # Local cache returned as-is
        assert records["100"].status == JobStatus.PENDING

    def test_empty_store_returns_empty(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        records = client._reconcile_store()
        assert records == {}


# ── list_jobs reconciliation ───────────────────────────────────────────


class TestListJobsReconciliation:
    """Verify list_jobs reconciles before returning."""

    def test_list_jobs_refreshes_statuses(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        client._store.append_job(_make_record("100", JobStatus.PENDING))

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {
                "100": SlurmJobInfo(job_id="100", state="RUNNING", exit_code="0:0"),
            }

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            jobs = client.list_jobs()

        assert len(jobs) == 1
        assert jobs[0].status == JobStatus.RUNNING

    def test_list_jobs_filters_by_experiment(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        client._store.append_job(_make_record("100", JobStatus.PENDING, experiment="exp-a"))
        client._store.append_job(_make_record("200", JobStatus.PENDING, experiment="exp-b"))

        with patch("slurp.client.sacct_query", return_value={}):
            jobs = client.list_jobs(experiment="exp-a")

        assert len(jobs) == 1
        assert jobs[0].job_id == "100"


# ── status() refresh ──────────────────────────────────────────────────


class TestStatusRefresh:
    """Verify status() refreshes from SLURM when job is in local store."""

    def test_status_refreshes_from_slurm(self, client: SyncClient, mock_ssh: MagicMock) -> None:
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

    def test_status_not_found(self, client: SyncClient, mock_ssh: MagicMock) -> None:
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

    def test_calls_wait_job_when_cache_empty(self, client: SyncClient, mock_ssh: MagicMock) -> None:
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


class TestSubmitEnvResolution:
    """Verify $PROJECT/$SCRATCH in sync.remote are resolved to literal paths."""

    def test_submit_resolves_project_in_remote(
        self, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """A $PROJECT in sync.remote must be expanded before it reaches
        sbatch (which puts it in #SBATCH --output, where $VAR does NOT expand)
        and before it is stored in the JobRecord."""
        proj_profile = Profile(
            name="test",
            hostname="hpc.local",
            username="user",
            partition="gpu",
            account="lab",
            sync=Profile.SyncConfig(local=".", remote="$PROJECT/slurp-projects"),
        )
        store_path = tmp_path / "jobs.json"
        offset_path = tmp_path / "log_offsets.json"
        with (
            patch("slurp.client.SSHManager", return_value=mock_ssh),
            patch("slurp.core.store.DEFAULT_STORE_PATH", store_path),
            patch("slurp.core.store.DEFAULT_OFFSET_PATH", offset_path),
        ):
            client = SyncClient(profile=proj_profile)

        # resolve_remote_env SSHes `eval echo '${PROJECT:?unset}/...'; reply
        # with the resolved literal. Other SSH calls (none expected here since
        # sync/sbatch are patched) get an empty success.
        async def fake_run(profile, command, **kwargs):
            if command.startswith("eval echo"):
                return (0, "/p/projects/cproj/slurp-projects\n", "")
            return (0, "", "")

        mock_ssh.run = AsyncMock(side_effect=fake_run)

        async def fake_sbatch(profile, script, *, working_dir, ssh_manager):
            # working_dir fed to sbatch (and thus #SBATCH --output) must be
            # the resolved literal, never the raw $PROJECT template.
            assert working_dir == "/p/projects/cproj/slurp-projects", working_dir
            assert "$PROJECT" not in script, "unexpanded $VAR leaked into sbatch script"
            return "60000"

        with (
            patch("slurp.client.sbatch_submit", side_effect=fake_sbatch),
            patch("slurp.client.sync_to_remote", new_callable=AsyncMock),
        ):
            job = client.submit("python train.py", gpus=1)

        # Stored Job + JobRecord both carry the resolved literal, so later
        # pull/logs (which read record.working_dir) hit the right remote path.
        assert job.working_dir == "/p/projects/cproj/slurp-projects"
        record = client._store.get_job("60000")
        assert record is not None
        assert record.working_dir == "/p/projects/cproj/slurp-projects"

    def test_submit_literal_remote_makes_no_ssh_for_resolution(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """A remote with no $VAR must not SSH just to resolve it."""
        async def fake_sbatch(profile, script, *, working_dir, ssh_manager):
            return "60001"

        with (
            patch("slurp.client.sbatch_submit", side_effect=fake_sbatch),
            patch("slurp.client.sync_to_remote", new_callable=AsyncMock),
        ):
            client.submit("python train.py", gpus=1)

        # No eval echo command should have been issued.
        for call in mock_ssh.run.call_args_list:
            assert not call.args[1].startswith("eval echo"), call.args[1]


# ── _sync_venv ────────────────────────────────────────────────────────


class TestSyncVenv:
    """Tests for the remote venv sync hook (uv-sync strategy)."""

    @pytest.fixture
    def venv_profile(self) -> Profile:
        return Profile(
            name="test",
            hostname="hpc.local",
            username="user",
            partition="gpu",
            account="lab",
            sync=Profile.SyncConfig(local=".", remote="/remote/project"),
            venv=Profile.VenvConfig(
                strategy="uv-sync",
                path="/remote/.venv",
                extras=["cu121"],
            ),
        )

    @pytest.fixture
    def venv_client(self, venv_profile: Profile, mock_ssh: MagicMock, tmp_path: Path) -> SyncClient:
        store_path = tmp_path / "jobs.json"
        offset_path = tmp_path / "log_offsets.json"
        with (
            patch("slurp.client.SSHManager", return_value=mock_ssh),
            patch("slurp.core.store.DEFAULT_STORE_PATH", store_path),
            patch("slurp.core.store.DEFAULT_OFFSET_PATH", offset_path),
        ):
            return SyncClient(profile=venv_profile)

    def test_noop_when_venv_is_none(self, client: SyncClient) -> None:
        """Profile without venv config should be a complete no-op."""
        client._sync_venv("/remote/project")
        # No SSH calls should have been made
        client._ssh.run.assert_not_called()

    def test_noop_when_no_lockfile(self, venv_client: SyncClient, tmp_path: Path) -> None:
        """Without a local uv.lock, venv sync should skip."""
        # chdir to tmp_path (no uv.lock there)
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            venv_client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)
        venv_client._ssh.run.assert_not_called()

    def test_runs_uv_sync_when_lockfile_exists(
        self, venv_client: SyncClient, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """With uv.lock present, should SSH a uv sync command to the remote."""
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "uv.lock").write_text('version = "1"')
        mock_ssh.run.return_value = (0, "", "")
        try:
            venv_client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)

        mock_ssh.run.assert_called_once()
        args = mock_ssh.run.call_args
        cmd = args.args[1]  # second positional arg is the command
        assert "uv sync" in cmd
        assert "--frozen" in cmd
        assert "--extra cu121" in cmd
        assert "UV_PROJECT_ENVIRONMENT=/remote/.venv" in cmd
        assert "/remote/project" in cmd

    def test_all_extras_flag(self, tmp_path: Path, mock_ssh: MagicMock) -> None:
        """all_extras=True should pass --all-extras instead of individual extras."""
        profile = Profile(
            name="test",
            hostname="hpc.local",
            username="user",
            sync=Profile.SyncConfig(local=".", remote="/remote/project"),
            venv=Profile.VenvConfig(
                strategy="uv-sync",
                path="/remote/.venv",
                all_extras=True,
            ),
        )
        store_path = tmp_path / "jobs.json"
        offset_path = tmp_path / "log_offsets.json"
        with (
            patch("slurp.client.SSHManager", return_value=mock_ssh),
            patch("slurp.core.store.DEFAULT_STORE_PATH", store_path),
            patch("slurp.core.store.DEFAULT_OFFSET_PATH", offset_path),
        ):
            client = SyncClient(profile=profile)

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "uv.lock").write_text('version = "1"')
        mock_ssh.run.return_value = (0, "", "")
        try:
            client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)

        cmd = mock_ssh.run.call_args.args[1]
        assert "--all-extras" in cmd
        assert "--extra" not in cmd

    def test_raises_on_uv_not_found(
        self, venv_client: SyncClient, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """If uv is not installed on the remote, raise SyncError with a hint."""
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "uv.lock").write_text('version = "1"')
        mock_ssh.run.return_value = (127, "", "bash: uv: command not found")
        try:
            with pytest.raises(SyncError, match="uv not found"):
                venv_client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)

    def test_raises_on_sync_failure(
        self, venv_client: SyncClient, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """Non-zero exit from uv sync should raise SyncError."""
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "uv.lock").write_text('version = "1"')
        mock_ssh.run.return_value = (1, "", "error: failed to resolve packages")
        try:
            with pytest.raises(SyncError, match="venv sync failed"):
                venv_client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)

    def test_hash_check_command_includes_marker(
        self, venv_client: SyncClient, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """The SSH command should include the hash-check logic."""
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "uv.lock").write_text('version = "1"')
        mock_ssh.run.return_value = (0, "", "")
        try:
            venv_client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)

        cmd = mock_ssh.run.call_args.args[1]
        assert "sha256sum uv.lock" in cmd
        assert ".lockhash" in cmd
        assert "NEW_HASH" in cmd
        assert "OLD_HASH" in cmd

    def test_timeout_is_generous(
        self, venv_client: SyncClient, mock_ssh: MagicMock, tmp_path: Path
    ) -> None:
        """uv sync can take minutes — timeout should be at least 300s."""
        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        (tmp_path / "uv.lock").write_text('version = "1"')
        mock_ssh.run.return_value = (0, "", "")
        try:
            venv_client._sync_venv("/remote/project")
        finally:
            os.chdir(old_cwd)

        timeout = mock_ssh.run.call_args.kwargs.get("timeout", 30)
        assert timeout >= 300


# ── _sacct_lookup (array parent fallback) ─────────────────────────────


class TestSacctLookup:
    """Tests for _sacct_lookup — sacct with array-parent fallback.

    On JURECA, sacct -j <parent>_<task> returns nothing. The parent ID
    resolves, so we retry with it and prefer the specific task row.
    """

    def test_direct_hit_returns_immediately(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """Non-array job ID found on first sacct query."""

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {
                "100": SlurmJobInfo(job_id="100", state="RUNNING", exit_code="0:0"),
            }

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            info = client._sacct_lookup("100")

        assert info is not None
        assert info.status == JobStatus.RUNNING

    def test_array_task_falls_back_to_parent(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """Array task ID misses sacct → parent query returns the task row."""

        def fake_sacct(profile, job_ids, *, ssh_manager):
            if job_ids == ["15395489_5"]:
                return {}  # JURECA: direct query misses
            if job_ids == ["15395489"]:
                return {
                    "15395489_5": SlurmJobInfo(
                        job_id="15395489_5", state="COMPLETED", exit_code="0:0"
                    ),
                }
            return {}

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            info = client._sacct_lookup("15395489_5")

        assert info is not None
        assert info.status == JobStatus.COMPLETED

    def test_array_task_uses_parent_status_when_task_row_absent(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """When sacct collapses arrays, parent row is used as proxy."""

        def fake_sacct(profile, job_ids, *, ssh_manager):
            if job_ids == ["15395489_5"]:
                return {}
            if job_ids == ["15395489"]:
                return {
                    "15395489": SlurmJobInfo(job_id="15395489", state="COMPLETED", exit_code="0:0"),
                }
            return {}

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            info = client._sacct_lookup("15395489_5")

        assert info is not None
        assert info.status == JobStatus.COMPLETED

    def test_prefers_task_row_over_parent_row(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """When both task and parent rows exist, the task row wins."""

        def fake_sacct(profile, job_ids, *, ssh_manager):
            if job_ids == ["15395489_5"]:
                return {}  # direct miss
            if job_ids == ["15395489"]:
                return {
                    "15395489": SlurmJobInfo(job_id="15395489", state="COMPLETED", exit_code="0:0"),
                    "15395489_5": SlurmJobInfo(
                        job_id="15395489_5", state="FAILED", exit_code="1:0"
                    ),
                }
            return {}

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            info = client._sacct_lookup("15395489_5")

        assert info is not None
        assert info.status == JobStatus.FAILED
        assert info.exit_code_int == 1

    def test_returns_none_when_both_miss(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """Neither task ID nor parent resolves → None."""

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {}

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            info = client._sacct_lookup("15395489_5")

        assert info is None

    def test_non_array_id_no_parent_retry(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """Non-array job ID (no underscore) should not trigger parent retry."""

        call_count = 0

        def fake_sacct(profile, job_ids, *, ssh_manager):
            nonlocal call_count
            call_count += 1
            return {}

        with patch("slurp.client.sacct_query", side_effect=fake_sacct):
            info = client._sacct_lookup("100")

        assert info is None
        assert call_count == 1  # No parent retry for non-array IDs


# ── refresh_job squeue fallback ──────────────────────────────────────


class TestRefreshJobSqueueFallback:
    """Verify refresh_job falls back to squeue when sacct misses.

    Without this fallback, wait_job hangs indefinitely for array task
    IDs on clusters where sacct can't resolve <parent>_<task>.
    """

    def test_squeue_fallback_updates_status(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """sacct misses → squeue finds the job → status updated."""
        job = Job(
            job_id="15395489_5",
            name="train",
            status=JobStatus.PENDING,
            profile="test",
        )

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {}  # sacct misses (JURECA)

        def fake_squeue(profile, *, ssh_manager, user=None):
            return {"15395489_[0-11%20]": JobStatus.RUNNING}

        with (
            patch("slurp.client.sacct_query", side_effect=fake_sacct),
            patch("slurp.client.squeue_query", side_effect=fake_squeue),
        ):
            refreshed = client.refresh_job(job)

        assert refreshed.status == JobStatus.RUNNING

    def test_both_miss_returns_unchanged(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """Neither sacct nor squeue finds the job → return unchanged."""
        job = Job(
            job_id="99999",
            name="train",
            status=JobStatus.RUNNING,
            profile="test",
        )

        with (
            patch("slurp.client.sacct_query", return_value={}),
            patch("slurp.client.squeue_query", return_value={}),
        ):
            refreshed = client.refresh_job(job)

        assert refreshed.status == JobStatus.RUNNING
        assert refreshed is job  # same object, unchanged

    def test_sacct_hit_skips_squeue(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """sacct finds the job → squeue should not be queried."""
        job = Job(
            job_id="100",
            name="train",
            status=JobStatus.PENDING,
            profile="test",
        )

        def fake_sacct(profile, job_ids, *, ssh_manager):
            return {
                "100": SlurmJobInfo(job_id="100", state="COMPLETED", exit_code="0:0"),
            }

        with (
            patch("slurp.client.sacct_query", side_effect=fake_sacct),
            patch("slurp.client.squeue_query") as mock_squeue,
        ):
            refreshed = client.refresh_job(job)

        assert refreshed.status == JobStatus.COMPLETED
        mock_squeue.assert_not_called()

    def test_array_task_via_parent_sacct_skips_squeue(
        self, client: SyncClient, mock_ssh: MagicMock
    ) -> None:
        """Array task resolved via parent sacct → squeue not needed."""
        job = Job(
            job_id="15395489_5",
            name="train",
            status=JobStatus.PENDING,
            profile="test",
        )

        def fake_sacct(profile, job_ids, *, ssh_manager):
            if job_ids == ["15395489_5"]:
                return {}
            if job_ids == ["15395489"]:
                return {
                    "15395489_5": SlurmJobInfo(
                        job_id="15395489_5", state="COMPLETED", exit_code="0:0"
                    ),
                }
            return {}

        with (
            patch("slurp.client.sacct_query", side_effect=fake_sacct),
            patch("slurp.client.squeue_query") as mock_squeue,
        ):
            refreshed = client.refresh_job(job)

        assert refreshed.status == JobStatus.COMPLETED
        mock_squeue.assert_not_called()


# ── _status_from_squeue $USER default ─────────────────────────────────


class TestStatusFromSqueueUserDefault:
    """Verify squeue filters by $USER when profile.username is unset.

    Without the filter, squeue returns every job on the cluster —
    potentially thousands of lines on a shared HPC system.
    """

    def test_uses_username_when_set(self, client: SyncClient, mock_ssh: MagicMock) -> None:
        """Profile with username → squeue called with that username."""
        captured_user: list = []

        def fake_squeue(profile, *, ssh_manager, user=None):
            captured_user.append(user)
            return {}

        with patch("slurp.client.squeue_query", side_effect=fake_squeue):
            client._status_from_squeue("100")

        assert captured_user == ["user"]

    def test_defaults_to_dollar_user_when_unset(self, mock_ssh: MagicMock, tmp_path: Path) -> None:
        """Profile without username → squeue called with $USER literal."""
        no_user_profile = Profile(
            name="test",
            hostname="hpc.local",
            username=None,  # No explicit username
            partition="gpu",
            account="lab",
            sync=Profile.SyncConfig(local=".", remote="/remote/project"),
        )
        store_path = tmp_path / "jobs.json"
        offset_path = tmp_path / "log_offsets.json"
        captured_user: list = []

        def fake_squeue(profile, *, ssh_manager, user=None):
            captured_user.append(user)
            return {}

        with (
            patch("slurp.client.SSHManager", return_value=mock_ssh),
            patch("slurp.core.store.DEFAULT_STORE_PATH", store_path),
            patch("slurp.core.store.DEFAULT_OFFSET_PATH", offset_path),
            patch("slurp.client.squeue_query", side_effect=fake_squeue),
        ):
            c = SyncClient(profile=no_user_profile)
            c._status_from_squeue("100")

        assert captured_user == ["$USER"]
