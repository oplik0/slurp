"""Code sync: rsync to remote working directory."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from slurp.domain import Profile
from slurp.errors import SyncError


def _is_git_dirty(local_dir: Path) -> bool:
    """Check if the local directory is inside a dirty git repo."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=local_dir,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


async def sync_to_remote(
    profile: Profile,
    local_dir: Path,
    remote_dir: str,
    *,
    ssh_manager: Any | None = None,
) -> None:
    """Sync local code to remote working directory via rsync over SSH."""
    from slurp.core.ssh import SSHManager

    if ssh_manager is None:
        ssh_manager = SSHManager()

    # Ensure control master is up for rsync to use
    ssh_manager._ensure_control_master(profile)
    sock = ssh_manager.control_master_socket(profile)

    ssh_opts = [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]
    if profile.username:
        ssh_opts.extend(["-l", profile.username])
    if profile.key_file:
        ssh_opts.extend(["-i", os.path.expanduser(profile.key_file)])
    ssh_opts.extend(["-S", str(sock)])

    ssh_cmd = "ssh " + " ".join(ssh_opts)

    cmd = [
        "rsync",
        "-avz",
        "--delete",
        "--filter=:- .gitignore",
        "-e",
        ssh_cmd,
        str(local_dir) + "/",
        f"{profile.hostname}:{remote_dir}/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise SyncError(
            "rsync timed out after 300s",
            hint="Your project may be very large. Consider using a .gitignore to exclude data.",
        ) from exc
    except FileNotFoundError as exc:
        raise SyncError(
            "rsync not found in PATH",
            hint="Install rsync: sudo apt-get install rsync",
        ) from exc

    if result.returncode != 0:
        raise SyncError(
            f"rsync failed (exit {result.returncode}): {result.stderr}",
            hint="Check disk quota, permissions, and network connectivity.",
        )

    # Check dirty git
    if _is_git_dirty(local_dir):
        import structlog

        logger = structlog.get_logger()
        logger.warning(
            "Git workspace has uncommitted changes.",
            hint="Use --snapshot if reproducibility matters.",
        )


async def snapshot_remote(
    profile: Profile,
    remote_dir: str,
    job_id: str,
    *,
    ssh_manager: Any | None = None,
) -> str:
    """Copy the remote working dir to a snapshot directory."""
    from slurp.core.ssh import SSHManager

    if ssh_manager is None:
        ssh_manager = SSHManager()

    snapshot_dir = f"{remote_dir}/.slurp/runs/{job_id}"
    try:
        await ssh_manager.run(
            profile,
            f"mkdir -p {snapshot_dir} && cp -a {remote_dir}/. {snapshot_dir}/",
            timeout=60.0,
        )
    except Exception as exc:
        raise SyncError(
            f"Snapshot failed: {exc}",
            hint="Check remote disk space and permissions.",
        ) from exc
    return snapshot_dir
