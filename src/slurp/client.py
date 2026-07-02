"""SyncClient — synchronous public API facade over async core modules."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from slurp.core.slurm import (
    build_torchrun_command,
    generate_array_wrapper,
    generate_sbatch_script,
    log_path,
    sacct_query,
    sbatch_submit,
    scancel,
    squeue_query,
)
from slurp.core.ssh import SSHManager
from slurp.core.store import JobStore, LogOffsetStore
from slurp.core.sync import rsync_from_remote, snapshot_remote, sync_to_remote
from slurp.domain import (
    ArrayJob,
    Job,
    JobRecord,
    JobResult,
    JobStatus,
    Profile,
    ResourceRequest,
    SlurmJobInfo,
    slugify_command,
)
from slurp.errors import (
    JobFailedError,
    ProfileError,
    SlurmError,
    SyncError,
)

logger = structlog.get_logger()


def _resolve_profile(name: str | None = None) -> Profile:
    """Load a profile from ~/.config/slurp/profiles.toml."""
    import tomllib

    config_path = Path.home() / ".config" / "slurp" / "profiles.toml"
    if not config_path.exists():
        if name:
            raise ProfileError(
                f"Profile '{name}' not found and no profiles.toml exists.",
                hint="Run: slurp config add-profile <name>",
            )
        raise ProfileError(
            "No profiles configured.",
            hint="Run: slurp config add-profile <name>",
        )

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    profiles = data.get("profiles", {})
    if not profiles:
        raise ProfileError("profiles.toml is empty.", hint="Add a profile first.")

    if name:
        raw = profiles.get(name)
        if raw is None:
            raise ProfileError(
                f"Profile '{name}' not found.",
                hint=f"Available: {', '.join(profiles.keys())}",
            )
        return Profile(name=name, **raw)

    # Default: try 'default', then first available
    if "default" in profiles:
        return Profile(name="default", **profiles["default"])
    first_name = next(iter(profiles))
    return Profile(name=first_name, **profiles[first_name])


def _parse_timeout(timeout: str | float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, (int, float)):
        return float(timeout)
    timeout = str(timeout).strip().lower()
    if timeout.endswith("h"):
        return float(timeout[:-1]) * 3600
    if timeout.endswith("m"):
        return float(timeout[:-1]) * 60
    if timeout.endswith("s"):
        return float(timeout[:-1])
    return float(timeout)


def _idempotency_hash(
    command: str, resources: ResourceRequest, working_dir: str, profile: str
) -> str:
    payload = json.dumps(
        {
            "command": command,
            "resources": resources.model_dump(),
            "working_dir": working_dir,
            "profile": profile,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class SyncClient:
    """Synchronous client for submitting and managing SLURM jobs."""

    def __init__(self, profile: str | Profile | None = None) -> None:
        if isinstance(profile, Profile):
            self.profile = profile
        else:
            self.profile = _resolve_profile(profile)
        self._ssh = SSHManager()
        self._store = JobStore()
        self._offset_store = LogOffsetStore()
        # Persistent background event loop. A cached asyncssh connection is
        # bound to the loop that created it; if we used asyncio.run() per
        # _run() call (fresh loop each time), the second call would reuse a
        # connection attached to a now-dead loop and raise
        # "Future attached to a different loop" -- which the sacct/squeue
        # wrappers swallow, manifesting as wait_job() timeouts.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_background_loop, name="slurp-asyncio", daemon=True
        )
        self._thread.start()

    def _run_background_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def __enter__(self) -> SyncClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close SSH connections and tear down the background loop."""
        if not self._loop.is_running():
            return
        # Close connections on their own loop (thread-safe context).
        try:
            asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop).result(
                timeout=5.0
            )
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    async def _shutdown_async(self) -> None:
        """Close cached SSH connections and the loop's default executor.

        asyncssh offloads blocking I/O (DNS resolution, key loading) to the
        loop's default ThreadPoolExecutor. Stopping the loop alone leaves those
        worker threads alive -- one per SyncClient -- so they must be drained
        explicitly or long-running processes accumulate threads.
        """
        self._ssh.close(profile=self.profile)
        try:
            await asyncio.wait_for(self._loop.shutdown_default_executor(), timeout=2.0)
        except (TimeoutError, NotImplementedError, RuntimeError):
            pass

    def _run(self, coro: Any) -> Any:
        """Run an async coroutine from sync code on the persistent loop.

        Falls back to a fresh asyncio.run() in a worker thread when called
        from inside a running event loop (e.g. Jupyter), because blocking the
        caller's loop with future.result() would deadlock.
        """
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _sync_venv(self, remote_dir: str) -> None:
        """Ensure the remote venv is up-to-date via ``uv sync``.

        Runs on the login node (which has network access) after rsync
        but before sbatch.  The venv is cached on the remote and only
        rebuilt when ``uv.lock`` changes.

        No-op when ``profile.venv`` is None or ``uv.lock`` is absent
        from the working directory.
        """
        venv = self.profile.venv
        if venv is None:
            return

        # Check locally — rsync just ran (or will run), so if uv.lock
        # exists in the sync source it exists on the remote. Use the profile's
        # sync.local dir (not cwd) so this works when the driver script runs
        # from a subdir (e.g. examples/) but uv.lock lives at the repo root.
        local_dir = (
            Path(self.profile.sync.local)
            if self.profile.sync and self.profile.sync.local
            else Path.cwd()
        )
        local_lockfile = local_dir / "uv.lock"
        if not local_lockfile.exists():
            logger.debug("venv_sync_skipped", reason="no uv.lock in working dir")
            return

        if venv.strategy != "uv-sync":
            logger.warning(
                "venv_sync_skipped",
                strategy=venv.strategy,
                reason="unknown strategy",
            )
            return

        venv_path = venv.path
        if venv.all_extras:
            extras_args = "--all-extras"
        else:
            extras_args = " ".join(f"--extra {e}" for e in venv.extras)

        # Pin the remote venv's Python to the LOCAL interpreter's major.minor.
        # cloudpickle serializes function code objects with version-specific
        # CPython internals; a payload pickled on 3.11 and unpickled on 3.14
        # segfaults. The driver and the workers MUST share a Python version.
        import sys

        python_pin = f"--python {sys.version_info[0]}.{sys.version_info[1]}"

        # Hash-check + conditional rebuild in one SSH call.
        # If the lockfile hash matches the marker, this is a no-op.
        #
        # uv's cache and managed-Python downloads can be large (~100MB); on
        # clusters with tight HOME quotas (JURECA: ~GBs) they fail mid-extract
        # with "Disk quota exceeded". Prefer the project scratch filesystem
        # (/p/scratch/<account>/<user>) when the account is set, falling back
        # to $HOME. The venv's bin/python symlinks to the managed install, so
        # compute nodes see it via the shared filesystem.
        account = self.profile.account or ""
        username = self.profile.username or ""
        uv_base = (
            f"/p/scratch/{account}/{username}/.uv" if account and username else '"$HOME/.uv"'
        )
        cmd = (
            f"cd {remote_dir} && "
            f"mkdir -p {uv_base}/cache {uv_base}/python && "
            f"export UV_CACHE_DIR={uv_base}/cache && "
            f"export UV_PYTHON_INSTALL_DIR={uv_base}/python && "
            f'NEW_HASH=$(sha256sum uv.lock 2>/dev/null | cut -d" " -f1) && '
            f'OLD_HASH=$(cat {venv_path}/.lockhash 2>/dev/null || echo "") && '
            f'if [ "$NEW_HASH" != "$OLD_HASH" ]; then '
            f'UV_PROJECT_ENVIRONMENT={venv_path} uv sync --frozen --no-dev {python_pin} {extras_args} && '
            f'mkdir -p {venv_path} && '
            f'echo "$NEW_HASH" > {venv_path}/.lockhash; '
            f"fi"
        )

        logger.info("venv_sync_start", venv_path=venv_path, remote_dir=remote_dir)
        exit_code, _, stderr = self._run(
            self._ssh.run(self.profile, cmd, check=False, timeout=600.0)
        )

        if exit_code != 0:
            stderr_str = stderr.strip() if stderr else ""
            if "command not found" in stderr_str or "uv: not found" in stderr_str:
                raise SyncError(
                    "uv not found on remote login node.",
                    hint="Install uv on the remote: pip install uv",
                )
            raise SyncError(
                f"venv sync failed (exit {exit_code}): {stderr_str}",
                hint="Check uv.lock is valid and the remote has network access.",
            )

        logger.info("venv_sync_complete", venv_path=venv_path)

    def submit(
        self,
        command: str,
        *,
        name: str | None = None,
        gpus: int | None = None,
        nodes: int = 1,
        cpus: int | None = None,
        mem: str | None = None,
        time: str | None = None,
        partition: str | None = None,
        account: str | None = None,
        constraint: str | None = None,
        qos: str | None = None,
        mail_type: str | None = None,
        slurm_kwargs: dict[str, str] | None = None,
        working_dir: str | None = None,
        experiment: str | None = None,
        sync: bool = True,
        snapshot: bool = False,
        depends_on: list[Job] | None = None,
        depends_on_type: str = "afterok",
    ) -> Job:
        """Submit a job to SLURM."""
        profile = self.profile
        resources = ResourceRequest(
            gpus=gpus if gpus is not None else 0,
            nodes=nodes,
            time=time or "2:00:00",
            mem=mem,
            cpus=cpus if cpus is not None else 8,
            partition=partition or profile.partition,
            account=account or profile.account,
            constraint=constraint,
            qos=qos,
            mail_type=mail_type,
            job_name=name,
            slurm_kwargs=slurm_kwargs or {},
        )

        local_dir = Path.cwd()
        remote_dir = working_dir or str(local_dir)

        if profile.sync and profile.sync.remote:
            remote_dir = profile.format_remote()
            if profile.sync.local:
                local_dir = Path(profile.sync.local)

        # Idempotency check
        hash_key = _idempotency_hash(command, resources, remote_dir, profile.name)
        dup_id = self._store.check_idempotency(hash_key)
        if dup_id:
            raise SlurmError(
                f"Job {dup_id} with identical spec was submitted recently.",
                hint="Use a different name or wait 30 seconds.",
            )

        # Sync
        if sync:
            self._run(sync_to_remote(profile, local_dir, remote_dir, ssh_manager=self._ssh))

        # Venv sync (login node, after rsync)
        self._sync_venv(remote_dir)

        # Build command
        job_name = resources.job_name or slugify_command(command)
        if nodes > 1:
            command = build_torchrun_command(
                command, profile, nodes=nodes, gpus_per_node=resources.gpus
            )

        # Generate script
        dep_ids = [j.job_id for j in (depends_on or []) if j.job_id]
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command=command,
            working_dir=remote_dir,
            job_name=job_name,
            snapshot=snapshot,
            depends_on=dep_ids or None,
            depends_on_type=depends_on_type,
        )

        # Submit
        job_id = self._run(
            sbatch_submit(profile, script, working_dir=remote_dir, ssh_manager=self._ssh)
        )

        # Snapshot immediately after submit so the job finds its working directory
        # when SLURM schedules it. The SBATCH script already cd's into the snapshot path.
        if snapshot:
            self._run(
                snapshot_remote(profile, remote_dir, job_id, ssh_manager=self._ssh)
            )

        # Create Job
        job = Job(
            job_id=job_id,
            name=job_name,
            status=JobStatus.PENDING,
            profile=profile.name,
            experiment=experiment,
            command=command,
            resources=resources,
            working_dir=remote_dir,
        )

        # Store
        record = JobRecord(
            job_id=job_id,
            name=job_name,
            status=JobStatus.PENDING,
            profile=profile.name,
            experiment=experiment,
            submitted_at=datetime.now(UTC),
            command=command,
            resources=resources,
            working_dir=remote_dir,
            idempotency_hash=hash_key,
            idempotency_time=datetime.now(UTC),
        )
        self._store.append_job(record)

        logger.info("Job submitted", job_id=job_id, name=job_name)
        return job

    def submit_array(
        self,
        template: str,
        *,
        configs: list[dict[str, str]],
        throttle: int = 20,
        **kwargs: Any,
    ) -> ArrayJob:
        """Submit a SLURM job array."""
        profile = self.profile
        working_dir = kwargs.pop("working_dir", str(Path.cwd()))
        experiment = kwargs.pop("experiment", None)
        name = kwargs.pop("name", None)
        sync = kwargs.pop("sync", True)
        snapshot = kwargs.pop("snapshot", False)

        resources = ResourceRequest(
            gpus=kwargs.pop("gpus", 0),
            nodes=kwargs.pop("nodes", 1),
            time=kwargs.pop("time", None) or "2:00:00",
            mem=kwargs.pop("mem", None),
            cpus=kwargs.pop("cpus", 8),
            partition=kwargs.pop("partition", None) or profile.partition,
            account=kwargs.pop("account", None) or profile.account,
            constraint=kwargs.pop("constraint", None),
            qos=kwargs.pop("qos", None),
            mail_type=kwargs.pop("mail_type", None),
            job_name=name,
            slurm_kwargs=kwargs.pop("slurm_kwargs", None) or {},
        )

        local_dir = Path.cwd()
        remote_dir = working_dir
        if profile.sync and profile.sync.remote:
            remote_dir = profile.format_remote()
            if profile.sync.local:
                local_dir = Path(profile.sync.local)

        if sync:
            self._run(sync_to_remote(profile, local_dir, remote_dir, ssh_manager=self._ssh))

        # Venv sync (login node, after rsync)
        self._sync_venv(remote_dir)

        script = generate_array_wrapper(
            resources=resources,
            profile=profile,
            template=template,
            configs=configs,
            working_dir=remote_dir,
            job_name=name,
            throttle=throttle,
        )

        job_id = self._run(
            sbatch_submit(profile, script, working_dir=remote_dir, ssh_manager=self._ssh)
        )

        if snapshot:
            self._run(
                snapshot_remote(profile, remote_dir, job_id, ssh_manager=self._ssh)
            )

        array = ArrayJob(
            array_job_id=job_id,
            name=name or "array",
            profile=profile.name,
            experiment=experiment,
            task_count=len(configs),
            throttle=throttle,
        )

        # Record in local store so the job appears in list/watch
        record = JobRecord(
            job_id=job_id,
            name=name or "array",
            status=JobStatus.PENDING,
            profile=profile.name,
            experiment=experiment,
            submitted_at=datetime.now(UTC),
            command=template,
            resources=resources,
            working_dir=remote_dir,
        )
        self._store.append_job(record)

        logger.info("Array job submitted", array_job_id=job_id, tasks=len(configs))
        return array

    def _sacct_lookup(self, job_id: str) -> SlurmJobInfo | None:
        """Query sacct for a single job, with array-parent fallback.

        On clusters that collapse uniform job arrays into a single sacct
        row (JURECA), ``sacct -j <parent>_<task>`` returns nothing while
        ``sacct -j <parent>`` succeeds.  When the direct lookup misses and
        *job_id* looks like an array task (``parent_task``), retry with
        the parent ID and prefer the specific task row if present.
        """
        info: dict[str, SlurmJobInfo] = self._run(
            sacct_query(self.profile, [job_id], ssh_manager=self._ssh)
        )
        if job_id in info:
            return info[job_id]
        # Array task ID that sacct can't resolve directly.
        if "_" in job_id:
            parent = job_id.rsplit("_", 1)[0]
            parent_info: dict[str, SlurmJobInfo] = self._run(
                sacct_query(self.profile, [parent], ssh_manager=self._ssh)
            )
            if job_id in parent_info:
                return parent_info[job_id]
            if parent in parent_info:
                return parent_info[parent]
        return None

    def refresh_job(self, job: Job) -> Job:
        """Refresh a job's status from SLURM.

        Tries sacct first (with array-parent fallback for task IDs),
        then falls back to squeue for live state — mirroring
        :meth:`status`.  Without the squeue fallback, ``wait_job``
        hangs indefinitely for array task IDs on clusters where sacct
        can't resolve ``<parent>_<task>`` (JURECA).
        """
        slurm_info = self._sacct_lookup(job.job_id)
        if slurm_info:
            new_status = slurm_info.status
            if new_status != job.status:
                self._store.update_job_status(job.job_id, new_status)
            return job.model_copy(update={"status": new_status})
        # sacct miss — fall back to squeue for live state (accounting DB
        # lag, pending jobs, or collapsed array task IDs on JURECA).
        squeue_job = self._status_from_squeue(job.job_id)
        if squeue_job is not None:
            if squeue_job.status != job.status:
                self._store.update_job_status(job.job_id, squeue_job.status)
            return job.model_copy(update={"status": squeue_job.status})
        return job

    def _reconcile_store(self) -> dict[str, JobRecord]:
        """Reconcile local job cache with SLURM for all non-terminal jobs.

        Queries ``sacct`` for every tracked job that is not in a terminal
        state, updates the local store, and returns the full record dict
        (with refreshed statuses).  If the SSH/SLURM query fails, falls
        back to whatever the local store has — stale data is better than
        no data.
        """
        records = self._store.list_jobs()
        non_terminal = [
            jid for jid, rec in records.items() if not rec.status.is_terminal
        ]
        if non_terminal:
            try:
                infos = self._run(
                    sacct_query(self.profile, non_terminal, ssh_manager=self._ssh)
                )
                for jid, info in infos.items():
                    rec = records.get(jid)
                    if rec and info.status != rec.status:
                        self._store.update_job_status(jid, info.status)
                        records[jid] = rec.model_copy(update={"status": info.status})
            except Exception:
                logger.warning("sacct reconciliation failed, using local cache")
        return records

    def wait_job(
        self,
        job: Job,
        *,
        timeout: str | float | None = None,
        follow_logs: bool = False,
        poll_interval: float = 5.0,
    ) -> JobResult:
        """Block until job reaches terminal state."""
        timeout_seconds = _parse_timeout(timeout)
        start = datetime.now(UTC)
        poll_interval = max(poll_interval, 1.0)
        out_offset = 0
        err_offset = 0

        while True:
            job = self.refresh_job(job)
            if job.status.is_terminal:
                break
            if timeout_seconds and (datetime.now(UTC) - start).total_seconds() > timeout_seconds:
                raise TimeoutError(f"Job {job.job_id} did not finish within {timeout}.")

            if follow_logs:
                out_path, err_path = log_path(job.job_id, job.name, job.working_dir)
                try:
                    out_data, new_out = self._run(
                        self._tail_remote(out_path, out_offset)
                    )
                    if out_data:
                        sys.stdout.write(out_data)
                        sys.stdout.flush()
                        out_offset = new_out
                except Exception:
                    pass
                try:
                    err_data, new_err = self._run(
                        self._tail_remote(err_path, err_offset)
                    )
                    if err_data:
                        sys.stderr.write(err_data)
                        sys.stderr.flush()
                        err_offset = new_err
                except Exception:
                    pass

            # Save offsets
            self._offset_store.set_offset(job.job_id, out_offset, err_offset)
            import time

            time.sleep(poll_interval)

        # Fetch final result
        slurm_info = self._sacct_lookup(job.job_id)
        exit_code = slurm_info.exit_code_int if slurm_info else None
        status = slurm_info.status if slurm_info else job.status

        # Read full logs
        out_path, err_path = log_path(job.job_id, job.name, job.working_dir)
        try:
            _, stdout, _ = self._run(
                self._ssh.run(self.profile, f"cat '{out_path}'", check=False, timeout=15.0)
            )
        except Exception:
            stdout = ""
        try:
            _, stderr, _ = self._run(
                self._ssh.run(self.profile, f"cat '{err_path}'", check=False, timeout=15.0)
            )
        except Exception:
            stderr = ""

        stdout = stdout[: 1024 * 1024]
        stderr = stderr[: 1024 * 1024]

        result = JobResult(
            job_id=job.job_id,
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            max_rss_mb=None,
            wall_time=0.0,
        )

        # Cache the result on the Job so repeated calls to result() are idempotent.
        job._result_cache[job.job_id] = result

        if status in (JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED) or (
            status == JobStatus.COMPLETED and exit_code != 0
        ):
            raise JobFailedError(
                job_id=job.job_id,
                exit_code=exit_code,
                status=status.value,
                stdout_tail=stdout[-1024:],
                stderr_tail=stderr[-1024:],
            )

        return result

    async def _tail_remote(self, path: str, offset: int) -> tuple[str, int]:
        """Return new data and new offset."""
        try:
            _, stdout, _ = await self._ssh.run(
                self.profile, f"tail -c +{offset + 1} '{path}'", check=False, timeout=5.0
            )
        except Exception:
            return "", offset
        return stdout, offset + len(stdout.encode())

    def job_logs(
        self,
        job: Job,
        *,
        follow: bool = False,
        tail: int = 100,
        stream: str = "both",
    ) -> Iterator[str]:
        """Yield log lines for a job.

        Args:
            stream: One of "stdout", "stderr", or "both" (default).
        """
        out_path, err_path = log_path(job.job_id, job.name, job.working_dir)
        if follow:
            yield from self._follow_logs(out_path, err_path, stream=stream)
        else:
            yield from self._tail_logs(out_path, err_path, tail, stream=stream)

    def _tail_logs(
        self,
        out_path: str,
        err_path: str,
        tail: int,
        *,
        stream: str = "both",
    ) -> Iterator[str]:
        """Yield last N lines of stdout and/or stderr."""
        if stream in ("both", "stdout"):
            try:
                _, stdout, _ = self._run(
                    self._ssh.run(self.profile, f"tail -n {tail} '{out_path}'", check=False, timeout=10.0)
                )
                if stdout:
                    yield stdout
            except Exception:
                pass
        if stream in ("both", "stderr"):
            try:
                _, stderr, _ = self._run(
                    self._ssh.run(self.profile, f"tail -n {tail} '{err_path}'", check=False, timeout=10.0)
                )
                if stderr:
                    yield stderr
            except Exception:
                pass

    def _follow_logs(
        self,
        out_path: str,
        err_path: str,
        *,
        stream: str = "both",
    ) -> Iterator[str]:
        """Blocking follow of log files."""
        out_offset = 0
        err_offset = 0
        while True:
            if stream in ("both", "stdout"):
                try:
                    out_data, out_offset = self._run(self._tail_remote(out_path, out_offset))
                    if out_data:
                        yield out_data
                except Exception:
                    break
            if stream in ("both", "stderr"):
                try:
                    err_data, err_offset = self._run(self._tail_remote(err_path, err_offset))
                    if err_data:
                        yield err_data
                except Exception:
                    break
            import time

            time.sleep(2.0)

    def cancel_job(self, job: Job) -> Job:
        """Cancel a job and return updated handle."""
        self._run(scancel(self.profile, job.job_id, ssh_manager=self._ssh))
        self._store.update_job_status(job.job_id, JobStatus.CANCELLED)
        return job.model_copy(update={"status": JobStatus.CANCELLED})

    def cancel_job_by_id(self, job_id: str) -> None:
        """Cancel a job by ID string (no status check, best-effort).

        Accepts array task IDs (``15395489_5``) as well as plain job IDs.
        ``scancel`` itself handles already-completed or non-existent jobs
        gracefully.
        """
        self._run(scancel(self.profile, job_id, ssh_manager=self._ssh))

    def job_result(self, job: Job) -> JobResult:
        """Idempotent accessor for job result."""
        cached = job._result_cache.get(job.job_id)
        if cached is not None:
            return cached
        return self.wait_job(job)

    def watch(self, experiment: str | None = None, refresh: float = 5.0) -> None:
        """Live watch of jobs."""
        from rich.live import Live
        from rich.table import Table

        refresh = max(refresh, 2.0)
        with Live(auto_refresh=False) as live:
            while True:
                table = Table(title="slurp watch")
                table.add_column("Job ID", style="cyan")
                table.add_column("Name", style="magenta")
                table.add_column("Status", style="green")
                table.add_column("Experiment", style="yellow")
                table.add_column("Time", style="blue")

                records = self._reconcile_store()
                for record in records.values():
                    if experiment and record.experiment != experiment:
                        continue
                    status = record.status.value
                    if record.status == JobStatus.RUNNING:
                        status = f"[bold green]{status}[/bold green]"
                    elif record.status.is_terminal:
                        status = f"[dim]{status}[/dim]"
                    table.add_row(
                        record.job_id,
                        record.name,
                        status,
                        record.experiment or "",
                        record.submitted_at.strftime("%H:%M:%S"),
                    )

                live.update(table)
                try:
                    import time

                    time.sleep(refresh)
                except KeyboardInterrupt:
                    break

    def list_jobs(
        self,
        *,
        experiment: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List tracked jobs, optionally filtered.

        Reconciles non-terminal jobs with SLURM via ``sacct`` before
        returning, so statuses reflect the real cluster state.
        """
        records = self._reconcile_store()
        jobs: list[Job] = []
        for record in records.values():
            if experiment and record.experiment != experiment:
                continue
            if status and record.status.value != status:
                continue
            jobs.append(
                Job(
                    job_id=record.job_id,
                    name=record.name,
                    status=record.status,
                    profile=record.profile,
                    experiment=record.experiment,
                    command=record.command,
                    resources=record.resources,
                    working_dir=record.working_dir,
                )
            )
            if len(jobs) >= limit:
                break
        return jobs

    def status(self, job_id: str) -> Job | None:
        """Get status for a single job, refreshing from SLURM."""
        record = self._store.get_job(job_id)
        if record:
            job = Job(
                job_id=record.job_id,
                name=record.name,
                status=record.status,
                profile=record.profile,
                experiment=record.experiment,
                command=record.command,
                resources=record.resources,
                working_dir=record.working_dir,
            )
            # Refresh from SLURM so the user sees current state
            return self.refresh_job(job)
        # Not in local store — try sacct directly
        info = self._run(sacct_query(self.profile, [job_id], ssh_manager=self._ssh))
        slurm_info = info.get(job_id)
        if slurm_info:
            return Job(
                job_id=job_id,
                name="unknown",
                status=slurm_info.status,
                profile=self.profile.name,
            )
        # sacct misses three cases: (1) pending/just-submitted jobs (accounting
        # DB lag — squeue sees them within ~1s, sacct takes 5s+), (2) array
        # task IDs "<parent>_<task>" on clusters that collapse uniform arrays
        # into one sacct row (JURECA: sacct -j 15395211_0 returns nothing; only
        # the parent 15395211 resolves). Fall back to squeue, which is live.
        return self._status_from_squeue(job_id)

    def _status_from_squeue(self, job_id: str) -> Job | None:
        """Look up a job's live state via squeue (sacct fallback).

        Returns None only if the job is in neither squeue nor sacct — i.e. it
        genuinely doesn't exist (or is so old it aged out of both).
        """
        # When the profile has no explicit username (relies on SSH config
        # for the remote user), pass $USER so squeue filters by the remote
        # user instead of returning every job on the cluster.
        user = self.profile.username or "$USER"
        states = self._run(
            squeue_query(self.profile, ssh_manager=self._ssh, user=user)
        )
        if not states:
            return None
        # Direct hit (non-array pending/running job).
        if job_id in states:
            return Job(
                job_id=job_id, name="unknown", status=states[job_id],
                profile=self.profile.name,
            )
        # Array task: squeue collapses to "<parent>_[0-11%20]". Match by parent
        # prefix — if the array is still active, the task is pending/running.
        parent = job_id.rsplit("_", 1)[0] if "_" in job_id else job_id
        for sq_id, st in states.items():
            if sq_id == parent or sq_id.startswith(f"{parent}_") or sq_id.startswith(
                f"{parent}["
            ):
                return Job(
                    job_id=job_id, name="unknown", status=st,
                    profile=self.profile.name,
                )
        return None

    def sync(self, local_dir: Path | None = None, remote_dir: str | None = None) -> None:
        """Sync code without submitting."""
        profile = self.profile
        local_dir = local_dir or Path.cwd()
        remote_dir = remote_dir or str(local_dir)
        if profile.sync and profile.sync.remote:
            remote_dir = profile.format_remote()
            if profile.sync.local:
                local_dir = Path(profile.sync.local)
        self._run(sync_to_remote(profile, local_dir, remote_dir, ssh_manager=self._ssh))

    def pull(self, job_id: str, local_dir: str | None = None) -> None:
        """Download job results from remote to local."""
        record = self._store.get_job(job_id)
        if not record:
            raise SyncError(
                f"Job {job_id} not found in local store.",
                hint="Submit or track the job before pulling results.",
            )
        remote_dir = record.working_dir
        local_dest = Path(local_dir or f"./outputs/{job_id}")

        self._run(
            rsync_from_remote(
                self.profile,
                remote_dir,
                local_dest,
                ssh_manager=self._ssh,
            )
        )

    # ArrayJob helpers
    def watch_array(self, array: ArrayJob) -> None:
        """Watch an array job."""
        self.watch()

    def array_logs(
        self, array: ArrayJob, *, task_id: int | None = None, follow: bool = False, tail: int = 100
    ) -> Iterator[str]:
        """Yield logs for array tasks."""
        for i in range(array.task_count):
            if task_id is not None and i != task_id:
                continue
            out_path, err_path = log_path(
                array.array_job_id, array.name, "", task_id=i
            )
            if task_id is None:
                yield f"--- Task {i} ---"
            yield from self._tail_logs(out_path, err_path, tail)

    def cancel_array(self, array: ArrayJob) -> ArrayJob:
        self._run(scancel(self.profile, array.array_job_id, ssh_manager=self._ssh))
        return array

    def cancel_array_task(self, array: ArrayJob, task_id: int) -> ArrayJob:
        self._run(scancel(self.profile, f"{array.array_job_id}_{task_id}", ssh_manager=self._ssh))
        return array

    def array_results(
        self,
        array: ArrayJob,
        *,
        timeout: str | float | None = None,
        poll_interval: float = 5.0,
    ) -> list[JobResult]:
        """Wait for all array tasks and return results."""
        results: list[JobResult] = []
        for i in range(array.task_count):
            job = Job(
                job_id=f"{array.array_job_id}_{i}",
                name=array.name,
                status=JobStatus.PENDING,
                profile=array.profile,
            )
            result = self.wait_job(job, timeout=timeout, poll_interval=poll_interval)
            results.append(result)
        return results

    def array_tasks(self, array: ArrayJob) -> list[Job]:
        return [
            Job(
                job_id=f"{array.array_job_id}_{i}",
                name=array.name,
                status=JobStatus.PENDING,
                profile=array.profile,
            )
            for i in range(array.task_count)
        ]
