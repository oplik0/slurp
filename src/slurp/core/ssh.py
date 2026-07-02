"""SSH transport: asyncssh-based connection manager with auto-reconnect."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

import asyncssh

from slurp.domain import Profile
from slurp.errors import SSHError

SOCKET_DIR = Path.home() / ".slurp" / "sockets"


class SSHManager:
    """Manages persistent SSH connections per profile."""

    def __init__(self) -> None:
        self._connections: dict[str, asyncssh.SSHClientConnection | None] = {}
        self._lock = asyncio.Lock()
        SOCKET_DIR.mkdir(parents=True, exist_ok=True)

    def _socket_path(self, profile: Profile) -> Path:
        slug = f"{profile.name}-{profile.hostname}"
        return SOCKET_DIR / f"{slug}.sock"

    def _ensure_control_master(self, profile: Profile) -> None:
        """Spawn an OpenSSH control master for rsync and other tools."""
        sock = self._socket_path(profile)
        # Check if already alive
        result = subprocess.run(
            ["ssh", "-O", "check", "-S", str(sock), profile.hostname],
            capture_output=True,
            timeout=0.5,
        )
        if result.returncode == 0:
            return
        # Spawn new control master
        cmd = [
            "ssh",
            "-MNf",
            "-S",
            str(sock),
            "-o",
            "ControlPersist=600",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]
        if profile.username:
            cmd.extend(["-l", profile.username])
        if profile.key_file:
            key_path = os.path.expanduser(profile.key_file)
            cmd.extend(["-i", key_path])
        if profile.proxy_jump:
            cmd.extend(["-J", profile.proxy_jump])
        cmd.append(profile.hostname)

        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
        except subprocess.TimeoutExpired:
            pass  # ssh -MNf may still succeed in background

    async def _connect(self, profile: Profile) -> asyncssh.SSHClientConnection:
        """Create a new asyncssh connection, reading OpenSSH config."""
        options: dict[str, Any] = {
            "known_hosts": None,  # Allow new hosts; OpenSSH config can override
        }
        # Only hand asyncssh a config path that exists; a missing file makes
        # it raise FileNotFoundError, which previously surfaced as a misleading
        # "connection timed out".
        config_path = Path.home() / ".ssh" / "config"
        if config_path.exists():
            options["config"] = [str(config_path)]
        if profile.username:
            options["username"] = profile.username
        if profile.key_file:
            options["client_keys"] = [os.path.expanduser(profile.key_file)]
        if profile.proxy_jump:
            options["tunnel"] = profile.proxy_jump

        try:
            conn = await asyncssh.connect(profile.hostname, **options)
            return conn
        except asyncssh.Error as exc:
            raise SSHError(
                f"SSH connection to {profile.hostname} failed: {exc}",
                hint="Check network, VPN, and SSH config. Verify the host is reachable.",
                retryable=True,
            )
        except OSError as exc:
            raise SSHError(
                f"SSH connection to {profile.hostname} timed out: {exc}",
                hint="Check network and try again.",
                retryable=True,
            )

    async def get_connection(self, profile: Profile) -> asyncssh.SSHClientConnection:
        """Return a live connection, reconnecting if necessary."""
        async with self._lock:
            key = profile.name
            conn = self._connections.get(key)
            if conn is not None and not conn.is_closed():
                return conn
            conn = await self._connect(profile)
            self._connections[key] = conn
            return conn

    async def run(
        self,
        profile: Profile,
        command: str,
        *,
        timeout: float | None = 30.0,
        stdin: bytes | None = None,
        check: bool = True,
    ) -> tuple[int, str, str]:
        """Execute a remote command and return (exit_code, stdout, stderr)."""
        conn = await self.get_connection(profile)
        try:
            result = await conn.run(command, timeout=timeout, check=False)
            stdout = result.stdout or "" if isinstance(result.stdout, str) else ""
            stderr = result.stderr or "" if isinstance(result.stderr, str) else ""
            if check and result.exit_status != 0:
                raise SSHError(
                    f"Remote command failed with exit code {result.exit_status}: {command}",
                    stderr_fragment=stderr,
                    retryable=True,
                )
            return result.exit_status or 0, stdout, stderr
        except TimeoutError:
            raise SSHError(
                f"SSH command timed out after {timeout}s: {command}",
                hint="The remote host may be overloaded or unreachable.",
                retryable=True,
            )

    async def run_with_retry(
        self,
        profile: Profile,
        command: str,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float | None = 30.0,
    ) -> tuple[int, str, str]:
        """Run with exponential backoff retry on SSHError."""
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await self.run(profile, command, timeout=timeout)
            except SSHError as exc:
                last_exc = exc
                if not exc.retryable or attempt >= max_retries:
                    raise
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(min(delay, 30.0))
                # Force reconnect
                async with self._lock:
                    self._connections.pop(profile.name, None)
        assert last_exc is not None
        raise last_exc

    async def open_process(
        self,
        profile: Profile,
        command: str,
        *,
        timeout: float | None = None,
    ) -> asyncssh.SSHClientProcess[str]:
        """Open a remote process for streaming."""
        conn = await self.get_connection(profile)
        try:
            return await conn.create_process(command, timeout=timeout)
        except Exception as exc:
            raise SSHError(
                f"Failed to open remote process: {exc}",
                retryable=True,
            ) from exc

    def control_master_socket(self, profile: Profile) -> Path:
        """Return the path to the OpenSSH control master socket."""
        return self._socket_path(profile)

    def close(self, profile: Profile | None = None) -> None:
        """Close connection(s)."""
        if profile is None:
            for conn in self._connections.values():
                if conn is not None:
                    conn.close()
            self._connections.clear()
        else:
            conn = self._connections.pop(profile.name, None)
            if conn is not None:
                conn.close()


# Global manager instance
_manager: SSHManager | None = None


def get_manager() -> SSHManager:
    global _manager
    if _manager is None:
        _manager = SSHManager()
    return _manager
