"""Integration tests for the CLI.

These tests exercise the CLI end-to-end with mocked SSH/SLURM to simulate
a full workflow without requiring a real cluster.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from slurp.cli.main import app

runner = CliRunner()


class _MockSSHManager:
    """Mock SSHManager that works with isinstance checks."""

    _run_mock: AsyncMock = AsyncMock()
    _run_mock.return_value = (0, "Submitted batch job 99999", "")

    def __init__(self) -> None:
        self.run = self._run_mock

    def _ensure_control_master(self, profile: Any) -> None:
        pass

    def control_master_socket(self, profile: Any) -> Path:
        return Path("/tmp/fake.sock")

    def close(self, profile: Any | None = None) -> None:
        pass


@pytest.fixture
def mock_profiles_toml(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary profiles.toml and patch CONFIG_FILE."""
    config_dir = tmp_path / ".config" / "slurp"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "profiles.toml"
    config_file.write_text(
        """
[profiles.test]
hostname = "test.cluster"
username = "testuser"
partition = "test-part"
account = "test-acct"

[profiles.test.sync]
local = "."
remote = "/home/testuser/project"
"""
    )
    with patch("slurp.cli.config.CONFIG_FILE", config_file), patch(
        "slurp.client.Path.home", return_value=tmp_path
    ):
        yield config_file


@pytest.fixture
def mock_ssh() -> Generator[_MockSSHManager, None, None]:
    """Mock SSHManager to avoid real network connections."""
    _MockSSHManager._run_mock.reset_mock()
    _MockSSHManager._run_mock.return_value = (0, "Submitted batch job 99999", "")
    with patch("slurp.client.SSHManager", _MockSSHManager), patch(
        "slurp.core.ssh.SSHManager", _MockSSHManager
    ):
        yield _MockSSHManager()


@pytest.fixture
def mock_store(tmp_path: Path) -> Generator[Path, None, None]:
    """Mock job store to use a temporary directory."""
    store_path = tmp_path / "jobs.json"
    offset_path = tmp_path / "log_offsets.json"
    with patch("slurp.core.store.DEFAULT_STORE_PATH", store_path), patch(
        "slurp.core.store.DEFAULT_OFFSET_PATH", offset_path
    ):
        yield store_path


@pytest.fixture
def mock_rsync() -> Generator[MagicMock, None, None]:
    """Mock rsync subprocess calls to avoid real network I/O."""
    with patch("slurp.core.sync.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        yield mock_run


class TestSubmitWorkflow:
    def test_submit_and_status(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        result = runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--",
                "python",
                "train.py",
                "--lr",
                "0.01",
            ],
        )
        assert result.exit_code == 0
        assert "submitted" in result.output

        # Verify store was updated
        store = json.loads(mock_store.read_text())
        assert "jobs" in store
        assert len(store["jobs"]) == 1

    def test_submit_array_workflow(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        mock_ssh.run.return_value = (0, "Submitted batch job 100000", "")
        result = runner.invoke(
            app,
            [
                "submit-array",
                "--profile",
                "test",
                "python train.py --seed {seed}",
                "--seed",
                "1,2,3",
                "--gpus",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert "submitted" in result.output

    def test_cancel_job(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        mock_ssh.run.return_value = (0, "Submitted batch job 100001", "")
        runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--",
                "python",
                "train.py",
            ],
        )
        mock_ssh.run.return_value = (0, "", "")
        result = runner.invoke(
            app,
            [
                "cancel",
                "--profile",
                "test",
                "100001",
            ],
        )
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_sync_without_submit(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path
    ) -> None:
        with patch("slurp.core.sync.subprocess.run") as mock_rsync:
            mock_rsync.return_value = MagicMock(returncode=0, stderr="")
            result = runner.invoke(
                app,
                [
                    "sync",
                    "--profile",
                    "test",
                ],
            )
            assert result.exit_code == 0
            assert "Sync complete" in result.output

    def test_dry_run_output(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        result = runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--dry-run",
                "--gpus",
                "2",
                "--nodes",
                "2",
                "--",
                "python",
                "train.py",
            ],
        )
        assert result.exit_code == 0
        assert "#SBATCH" in result.output
        assert "--nodes=2" in result.output

    def test_list_and_watch(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        # Submit a few jobs
        mock_ssh.run.return_value = (0, "Submitted batch job 100002", "")
        runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--experiment",
                "exp-a",
                "--",
                "python",
                "train.py",
            ],
        )
        mock_ssh.run.return_value = (0, "Submitted batch job 100003", "")
        runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--experiment",
                "exp-b",
                "--",
                "python",
                "eval.py",
            ],
        )

        # List all
        result = runner.invoke(
            app,
            [
                "list",
                "--profile",
                "test",
            ],
        )
        assert result.exit_code == 0
        assert "100002" in result.output
        assert "100003" in result.output

        # List filtered by experiment
        result = runner.invoke(
            app,
            [
                "list",
                "--profile",
                "test",
                "--experiment",
                "exp-a",
            ],
        )
        assert result.exit_code == 0
        assert "100002" in result.output
        assert "100003" not in result.output

    def test_logs_output(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        mock_ssh.run.return_value = (0, "Submitted batch job 100004", "")
        runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--",
                "python",
                "train.py",
            ],
        )
        mock_ssh.run.return_value = (0, "epoch 1 loss 0.5\n", "")
        result = runner.invoke(
            app,
            [
                "logs",
                "--profile",
                "test",
                "100004",
                "--tail",
                "10",
            ],
        )
        assert result.exit_code == 0
        assert "epoch 1" in result.output or "epoch 1" in result.stderr

    def test_pull_output(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        mock_ssh.run.return_value = (0, "Submitted batch job 100005", "")
        runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--",
                "python",
                "train.py",
            ],
        )
        with patch("subprocess.run") as mock_rsync:
            mock_rsync.return_value = MagicMock(returncode=0, stderr="")
            result = runner.invoke(
                app,
                [
                    "pull",
                    "--profile",
                    "test",
                    "100005",
                ],
            )
            assert result.exit_code == 0
            assert "Pulled job 100005" in result.output

    def test_slurm_error_on_submit(
        self, mock_profiles_toml: Path, mock_ssh: _MockSSHManager, mock_store: Path, mock_rsync: MagicMock
    ) -> None:
        mock_ssh.run.return_value = (1, "", "sbatch: error: Batch job submission failed")
        result = runner.invoke(
            app,
            [
                "submit",
                "--profile",
                "test",
                "--",
                "python",
                "train.py",
            ],
        )
        assert result.exit_code == 1
        assert isinstance(result.exception, Exception) or "sbatch" in result.output or "SlurmError" in result.output
