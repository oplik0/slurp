"""Code sync: rsync to remote working directory."""

from __future__ import annotations

import os
import re
import shlex
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


def _rsync_ssh_options(profile: Profile, ssh_manager: Any) -> list[str]:
    """Build ssh options for rsync, using an existing control master socket."""
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
    return ssh_opts


def _run_rsync(cmd: list[str]) -> None:
    """Run a local rsync subprocess and raise SyncError on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise SyncError(
            "rsync timed out after 300s",
            hint="Your project may be very large. Consider using a .gitignore to exclude data.",
        )
    except FileNotFoundError:
        raise SyncError(
            "rsync not found in PATH",
            hint="Install rsync: sudo apt-get install rsync",
        )

    if result.returncode != 0:
        raise SyncError(
            f"rsync failed (exit {result.returncode}): {result.stderr}",
            hint="Check disk quota, permissions, and network connectivity.",
        )


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

    ssh_cmd = "ssh " + " ".join(_rsync_ssh_options(profile, ssh_manager))
    cmd = [
        "rsync",
        "-avz",
        "--delete",
        "--include=._slurp/***",  # sync task payloads despite .gitignore
        "--filter=:- .gitignore",
        "-e",
        ssh_cmd,
        str(local_dir) + "/",
        f"{profile.hostname}:{remote_dir}/",
    ]
    _run_rsync(cmd)

    # Check dirty git
    if _is_git_dirty(local_dir):
        import structlog

        logger = structlog.get_logger()
        logger.warning(
            "Git workspace has uncommitted changes.",
            hint="Use --snapshot if reproducibility matters.",
        )


async def rsync_from_remote(
    profile: Profile,
    remote_src: str,
    local_dst: Path,
    *,
    ssh_manager: Any | None = None,
) -> None:
    """Pull files from a remote directory to a local path via rsync over SSH."""
    from slurp.core.ssh import SSHManager

    if ssh_manager is None:
        ssh_manager = SSHManager()

    local_dst.mkdir(parents=True, exist_ok=True)
    ssh_cmd = "ssh " + " ".join(_rsync_ssh_options(profile, ssh_manager))
    cmd = [
        "rsync",
        "-avz",
        "-e",
        ssh_cmd,
        f"{profile.hostname}:{remote_src}/",
        str(local_dst) + "/",
    ]
    _run_rsync(cmd)


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
            (
                f"mkdir -p {snapshot_dir} && "
                f"cd {remote_dir} && "
                f"find . -mindepth 1 -maxdepth 1 ! -name '.slurp' "
                f"-exec cp -a {{}} {snapshot_dir}/ \\;"
            ),
            timeout=60.0,
        )
    except Exception as exc:
        raise SyncError(
            f"Snapshot failed: {exc}",
            hint="Check remote disk space and permissions.",
        )
    return snapshot_dir


async def resolve_remote_env(
    profile: Profile,
    path: str,
    *,
    ssh_manager: Any,
) -> str:
    """Expand ``$VAR``/``${VAR}`` in a remote path to a literal via the login shell.

    SLURM ``#SBATCH`` directives and rsync do not expand shell environment
    variables, so a ``sync.remote`` like ``"$PROJECT/slurp-projects"`` must be
    resolved to a literal absolute path before it is used as ``working_dir``
    (which feeds ``--output``, ``cd``, and rsync). Without this, SLURM opens a
    directory literally named ``$PROJECT`` and rsync syncs into one.

    Unset variables raise ``SyncError`` rather than expanding to empty -- a
    silent ``/slurp-projects`` would quietly write into the filesystem root.
    """
    if "$" not in path:
        return path
    # Rewrite $VAR / ${VAR} into ${VAR:?unset} so an unset var makes the
    # remote shell exit non-zero (instead of expanding to an empty string).
    safe = re.sub(r"\$\{?(\w+)\}?", r"${\1:?unset}", path)
    # eval re-parses the shlex-quoted path so ${VAR:?unset} is expanded by the
    # remote login shell. shlex.quote single-quotes the path, preventing local
    # expansion and injection; eval then expands ${VAR} on the remote side.
    cmd = f"eval echo {shlex.quote(safe)}"
    exit_code, stdout, _ = await ssh_manager.run(
        profile, cmd, timeout=10.0, check=False
    )
    if exit_code != 0:
        raise SyncError(
            f"Unset environment variable in remote path {path!r}",
            hint=(
                "Run `echo $PROJECT $SCRATCH` on the login node to check. "
                "Training accounts on JURECA have neither set."
            ),
        )
    # ssh_manager is typed Any, so stdout is Any -- coerce to str for mypy.
    resolved = str(stdout).strip()
    if not resolved:
        raise SyncError(
            f"Resolved remote path is empty: {path!r}",
            hint="Check $PROJECT/$SCRATCH on the login node.",
        )
    return resolved
