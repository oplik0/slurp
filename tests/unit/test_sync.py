"""Tests for slurp.core.sync rsync and code sync utilities."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from slurp.core.sync import _is_git_dirty, snapshot_remote, sync_to_remote
from slurp.domain import Profile


class TestIsGitDirty:
    """Tests for _is_git_dirty."""

    def test_not_git_repo(self, tmp_path: Path) -> None:
        assert _is_git_dirty(tmp_path) is False

    def test_clean_repo(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        assert _is_git_dirty(tmp_path) is False

    def test_dirty_repo(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        (tmp_path / "file.txt").write_text("hello")
        assert _is_git_dirty(tmp_path) is True

    def test_git_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: MagicMock(returncode=0, stdout=b""))
        # But actually if git is not found, FileNotFoundError is raised
        monkeypatch.setattr(
            "subprocess.run",
            lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
        )
        assert _is_git_dirty(tmp_path) is False


class TestSyncToRemote:
    """Tests for sync_to_remote."""

    async def test_rsync_command_construction(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        profile = Profile(name="test", hostname="hpc", username="user", key_file="~/.ssh/id_rsa")
        local_dir = tmp_path / "src"
        local_dir.mkdir()
        (local_dir / "main.py").write_text("print(1)")

        recorded_cmd: list[str] | None = None

        def fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal recorded_cmd
            if cmd[0] == "rsync":
                recorded_cmd = cmd
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

        mock_ssh = MagicMock()
        mock_ssh.control_master_socket.return_value = Path("/tmp/control.sock")

        await sync_to_remote(profile, local_dir, "/remote/dir", ssh_manager=mock_ssh)

        assert recorded_cmd is not None
        assert recorded_cmd[0] == "rsync"
        assert "-avz" in recorded_cmd
        assert "--delete" in recorded_cmd
        assert "--filter=:- .gitignore" in recorded_cmd
        # SSH opts
        ssh_opts = recorded_cmd[recorded_cmd.index("-e") + 1]
        assert "-S /tmp/control.sock" in ssh_opts
        assert "-l user" in ssh_opts
        assert "-i" in ssh_opts
        # source and dest
        assert str(local_dir) + "/" in recorded_cmd
        assert "hpc:/remote/dir/" in recorded_cmd

    async def test_rsync_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        profile = Profile(name="test", hostname="hpc")
        local_dir = tmp_path / "src"
        local_dir.mkdir()

        def fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd, timeout=300)

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

        mock_ssh = MagicMock()
        mock_ssh.control_master_socket.return_value = Path("/tmp/control.sock")

        from slurp.errors import SyncError

        with pytest.raises(SyncError, match="rsync timed out"):
            await sync_to_remote(profile, local_dir, "/remote/dir", ssh_manager=mock_ssh)

    async def test_rsync_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        profile = Profile(name="test", hostname="hpc")
        local_dir = tmp_path / "src"
        local_dir.mkdir()

        def fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="permission denied")

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

        mock_ssh = MagicMock()
        mock_ssh.control_master_socket.return_value = Path("/tmp/control.sock")

        from slurp.errors import SyncError

        with pytest.raises(SyncError, match="rsync failed"):
            await sync_to_remote(profile, local_dir, "/remote/dir", ssh_manager=mock_ssh)

    async def test_git_dirty_warning(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        profile = Profile(name="test", hostname="hpc")
        local_dir = tmp_path / "src"
        local_dir.mkdir()
        subprocess.run(["git", "init"], cwd=local_dir, check=True, capture_output=True)
        (local_dir / "dirty.txt").write_text("x")

        real_subprocess_run = subprocess.run

        def fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            if cmd[0] == "rsync":
                return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")
            return real_subprocess_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

        mock_ssh = MagicMock()
        mock_ssh.control_master_socket.return_value = Path("/tmp/control.sock")

        # Should not raise
        await sync_to_remote(profile, local_dir, "/remote/dir", ssh_manager=mock_ssh)


class TestSnapshotRemote:
    """Tests for snapshot_remote."""

    async def test_snapshot_success(self) -> None:
        profile = Profile(name="test", hostname="hpc")

        mock_ssh = MagicMock()
        mock_ssh.run = AsyncMock(return_value="")

        result = await snapshot_remote(profile, "/remote/dir", "123", ssh_manager=mock_ssh)
        assert result == "/remote/dir/.slurp/runs/123"
        mock_ssh.run.assert_called_once()

    async def test_snapshot_failure(self) -> None:
        profile = Profile(name="test", hostname="hpc")

        mock_ssh = MagicMock()
        mock_ssh.run = AsyncMock(side_effect=Exception("disk full"))

        from slurp.errors import SyncError

        with pytest.raises(SyncError, match="Snapshot failed"):
            await snapshot_remote(profile, "/remote/dir", "123", ssh_manager=mock_ssh)
