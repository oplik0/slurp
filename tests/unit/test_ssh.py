"""Mock-based tests for slurp.core.ssh connection manager."""

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import asyncssh
import pytest

from slurp.core.ssh import SSHManager, get_manager
from slurp.domain import Profile
from slurp.errors import SSHError


class TestSSHManagerInit:
    """Tests for SSHManager initialization."""

    def test_init(self) -> None:
        manager = SSHManager()
        assert manager._connections == {}
        assert manager._lock is not None

    def test_socket_path(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")
        path = manager._socket_path(profile)
        assert path.name == "test-hpc.local.sock"


class TestEnsureControlMaster:
    """Tests for _ensure_control_master."""

    def test_existing_control_master(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            manager._ensure_control_master(profile)
            # First call is the check
            assert mock_run.call_count == 1
            args = mock_run.call_args[0][0]
            assert args[:2] == ["ssh", "-O"]

    def test_spawn_new_control_master(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local", username="user", key_file="~/.ssh/key")

        with patch("subprocess.run") as mock_run:
            # First call (check) fails
            mock_run.side_effect = [
                MagicMock(returncode=255),
                MagicMock(returncode=0),
            ]
            manager._ensure_control_master(profile)
            assert mock_run.call_count == 2
            # Second call spawns master
            args = mock_run.call_args[0][0]
            assert "-MNf" in args
            assert "-l" in args
            assert "user" in args
            assert "-i" in args

    def test_timeout_ignored(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=255),  # check fails
                MagicMock(side_effect=subprocess.TimeoutExpired(["ssh"], 15)),
            ]
            # Should not raise
            manager._ensure_control_master(profile)


class TestConnect:
    """Tests for async _connect."""

    async def test_success(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_conn = MagicMock()
        with patch("asyncssh.connect", new_callable=AsyncMock, return_value=mock_conn):
            conn = await manager._connect(profile)
            assert conn == mock_conn

    async def test_asyncssh_error(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        with (
            patch(
                "asyncssh.connect",
                new_callable=AsyncMock,
                side_effect=asyncssh.Error(1, "conn refused"),
            ),
            pytest.raises(SSHError, match="SSH connection to hpc.local failed"),
        ):
            await manager._connect(profile)

    async def test_os_error(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        with (
            patch("asyncssh.connect", new_callable=AsyncMock, side_effect=OSError("timeout")),
            pytest.raises(SSHError, match="SSH connection to hpc.local timed out"),
        ):
            await manager._connect(profile)


class TestGetConnection:
    """Tests for get_connection."""

    async def test_new_connection(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        with patch.object(manager, "_connect", new_callable=AsyncMock, return_value=mock_conn):
            conn = await manager.get_connection(profile)
            assert conn == mock_conn
            assert manager._connections["test"] == mock_conn

    async def test_reuse_existing(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        manager._connections["test"] = mock_conn

        conn = await manager.get_connection(profile)
        assert conn == mock_conn
        # Should not call _connect
        with patch.object(manager, "_connect", new_callable=AsyncMock) as mock_connect:
            conn2 = await manager.get_connection(profile)
            assert conn2 == mock_conn
            mock_connect.assert_not_called()

    async def test_reconnect_closed(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        old_conn = MagicMock()
        old_conn.is_closed.return_value = True
        manager._connections["test"] = old_conn

        new_conn = MagicMock()
        new_conn.is_closed.return_value = False
        with patch.object(manager, "_connect", new_callable=AsyncMock, return_value=new_conn):
            conn = await manager.get_connection(profile)
            assert conn == new_conn


class TestRun:
    """Tests for run method."""

    async def test_success(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_result = MagicMock()
        mock_result.exit_status = 0
        mock_result.stdout = "hello"
        mock_result.stderr = ""

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(return_value=mock_result)
        manager._connections["test"] = mock_conn

        code, out, err = await manager.run(profile, "echo hello")
        assert code == 0
        assert out == "hello"
        assert err == ""

    async def test_check_raises(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_result = MagicMock()
        mock_result.exit_status = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(return_value=mock_result)
        manager._connections["test"] = mock_conn

        with pytest.raises(SSHError, match="Remote command failed with exit code 1"):
            await manager.run(profile, "false", check=True)

    async def test_no_check(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_result = MagicMock()
        mock_result.exit_status = 1
        mock_result.stdout = ""
        mock_result.stderr = ""

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(return_value=mock_result)
        manager._connections["test"] = mock_conn

        code, out, err = await manager.run(profile, "false", check=False)
        assert code == 1

    async def test_timeout(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(side_effect=TimeoutError())
        manager._connections["test"] = mock_conn

        with pytest.raises(SSHError, match="SSH command timed out"):
            await manager.run(profile, "sleep 10", timeout=1.0)


class TestRunWithRetry:
    """Tests for run_with_retry."""

    async def test_success_first_try(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_result = MagicMock()
        mock_result.exit_status = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(return_value=mock_result)
        manager._connections["test"] = mock_conn

        code, out, err = await manager.run_with_retry(profile, "echo ok")
        assert code == 0

    async def test_retry_then_success(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_result = MagicMock()
        mock_result.exit_status = 0
        mock_result.stdout = "ok"
        mock_result.stderr = ""

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        # First call raises retryable SSHError, second succeeds
        mock_conn.run = AsyncMock(
            side_effect=[
                SSHError("transient", retryable=True),
                mock_result,
            ]
        )
        manager._connections["test"] = mock_conn

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.object(manager, "_connect", new_callable=AsyncMock, return_value=mock_conn),
        ):
            code, out, err = await manager.run_with_retry(profile, "echo ok")
            assert code == 0

    async def test_non_retryable_fails_immediately(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(side_effect=SSHError("fatal", retryable=False))
        manager._connections["test"] = mock_conn

        with pytest.raises(SSHError, match="fatal"):
            await manager.run_with_retry(profile, "echo ok")

    async def test_max_retries_exceeded(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.run = AsyncMock(side_effect=SSHError("transient", retryable=True))
        manager._connections["test"] = mock_conn

        with (
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch.object(manager, "_connect", new_callable=AsyncMock, return_value=mock_conn),
            pytest.raises(SSHError, match="transient"),
        ):
            await manager.run_with_retry(profile, "echo ok", max_retries=1)


class TestOpenProcess:
    """Tests for open_process."""

    async def test_success(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")

        mock_process = MagicMock()
        mock_conn = MagicMock()
        mock_conn.is_closed.return_value = False
        mock_conn.create_process = AsyncMock(return_value=mock_process)
        manager._connections["test"] = mock_conn

        proc = await manager.open_process(profile, "tail -f log")
        assert proc == mock_process


class TestClose:
    """Tests for close method."""

    def test_close_all(self) -> None:
        manager = SSHManager()
        mock_conn = MagicMock()
        manager._connections["test"] = mock_conn
        manager.close()
        mock_conn.close.assert_called_once()
        assert manager._connections == {}

    def test_close_specific(self) -> None:
        manager = SSHManager()
        mock_conn1 = MagicMock()
        mock_conn2 = MagicMock()
        manager._connections["a"] = mock_conn1
        manager._connections["b"] = mock_conn2
        manager.close(Profile(name="a", hostname="h1"))
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_not_called()
        assert "a" not in manager._connections


class TestControlMasterSocket:
    """Tests for control_master_socket."""

    def test_returns_socket_path(self) -> None:
        manager = SSHManager()
        profile = Profile(name="test", hostname="hpc.local")
        path = manager.control_master_socket(profile)
        assert path.name == "test-hpc.local.sock"


class TestGetManager:
    """Tests for global get_manager."""

    def test_singleton(self) -> None:
        m1 = get_manager()
        m2 = get_manager()
        assert m1 is m2
        assert isinstance(m1, SSHManager)
