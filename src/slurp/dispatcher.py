"""Dispatcher: strategy object for submitting task calls.

The dispatcher decides what ``.distribute()`` / ``.remote()`` actually does:

- ``SlurmDispatcher``  — serializes the call, rsyncs, submits via sbatch.
- ``LocalDispatcher``  — calls the function directly in-process (testing /
  SLURM-less environments).
- ``JobBundling``     — wraps another dispatcher, buffers calls, and
  flushes them as SLURM job arrays.

Auto-detection: if no profile is configured, ``LocalDispatcher`` is used.
This gives you the slurminade property of "same code runs with or without
SLURM."
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import cloudpickle
import structlog

from slurp.domain import ResourceRequest

logger = structlog.get_logger()

# Directory for payloads, results, and shared objects.
# Lives in the working directory so rsync syncs it.
_SLURP_DIR = "._slurp"
_PAYLOAD_DIR = f"{_SLURP_DIR}/payloads"
_RESULT_DIR = f"{_SLURP_DIR}/results"
_OBJECT_DIR = f"{_SLURP_DIR}/objects"


def _ensure_dirs(base_dir: Path | str | None = None) -> None:
    """Create the local ._slurp subdirectories.

    Args:
        base_dir: Where to create ._slurp/. Defaults to the synced directory
            (profile.sync.local) so payloads reach the remote via rsync.
            Falls back to CWD if no profile is configured.
    """
    base = Path(base_dir) if base_dir else _local_base_dir()
    for d in (_PAYLOAD_DIR, _RESULT_DIR, _OBJECT_DIR):
        (base / d).mkdir(parents=True, exist_ok=True)


def _write_payload(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    base_dir: Path | str | None = None,
) -> str:
    """Serialize a function call to a payload file. Returns the UUID."""
    payload_id = uuid.uuid4().hex
    base = Path(base_dir) if base_dir else _local_base_dir()
    _ensure_dirs(base)
    path = base / f"{_PAYLOAD_DIR}/{payload_id}.pkl"
    with open(path, "wb") as f:
        cloudpickle.dump((func, args, kwargs), f)
    return payload_id


def _local_base_dir() -> Path:
    """Return the directory that gets synced to the remote.

    This is where ._slurp/ should live so payloads, results, and shared
    objects are available on the remote after rsync.  Falls back to CWD
    when no profile is configured (local execution).
    """
    try:
        from slurp.client import _resolve_profile

        profile = _resolve_profile()
        if profile.sync and profile.sync.local:
            return Path(profile.sync.local)
    except Exception:
        pass
    return Path.cwd()


def _cancel_job_ids(job_ids: list[str], profile_name: str | None) -> None:
    """Best-effort scancel for a list of job IDs.

    Never raises — if cancellation fails, the KeyboardInterrupt should
    still propagate to the user.
    """
    if not job_ids:
        return
    from slurp.client import SyncClient

    logger.info("keyboard_interrupt_cancelling", count=len(job_ids), job_ids=job_ids)
    try:
        with SyncClient(profile=profile_name) as client:
            for jid in job_ids:
                try:
                    client.cancel_job_by_id(jid)
                except Exception:
                    pass  # Best effort per job
    except Exception:
        pass  # Never mask the KeyboardInterrupt


class Ref:
    """Future handle for a remote function call.

    Wraps a SLURM job ID and knows how to retrieve the deserialized
    result.  Created by ``SlurmDispatcher``; consumed by ``slurp.get()``.
    """

    def __init__(
        self,
        job_id: str,
        result_id: str | None,
        profile: str,
        working_dir: str,
    ) -> None:
        self.job_id = job_id
        self.result_id = result_id
        self.profile = profile
        self.working_dir = working_dir
        self._cached: Any = None
        self._resolved = False

    @property
    def result_path(self) -> str | None:
        if self.result_id is None:
            return None
        return f"{self.working_dir}/{_RESULT_DIR}/{self.result_id}.pkl"

    def ready(self) -> bool:
        """Non-blocking check: is the result available?"""
        from slurp.client import SyncClient

        with SyncClient(profile=self.profile) as client:
            job = client.status(self.job_id)
            if job is None:
                return False
            return job.status.is_terminal

    def get(self, *, timeout: str | float | None = None) -> Any:
        """Block until the job finishes, then return the deserialized result.

        Raises:
            The original exception if the function failed on the remote.
            TimeoutError if the job doesn't finish within *timeout*.
        """
        if self._resolved:
            return self._cached

        from slurp.client import SyncClient

        with SyncClient(profile=self.profile) as client:
            job = client.status(self.job_id)
            if job is None:
                raise RuntimeError(f"Job {self.job_id} not found")
            try:
                result = client.wait_job(job, timeout=timeout)
            except KeyboardInterrupt:
                # Best-effort cancel — don't mask the interrupt.
                try:
                    client.cancel_job_by_id(self.job_id)
                    logger.info("cancelled_on_interrupt", job_id=self.job_id)
                except Exception:
                    pass
                raise

            if self.result_path is None:
                # Fire-and-forget job (no result file). Return the JobResult.
                self._cached = result
                self._resolved = True
                return result

            # Read result file from remote via base64
            import base64

            _, stdout, _ = client._run(  # noqa: SLF001
                client._ssh.run(  # noqa: SLF001
                    client.profile,
                    f"base64 '{self.result_path}'",
                    check=False,
                    timeout=30.0,
                )
            )
            if not stdout.strip():
                raise RuntimeError(
                    f"Result file {self.result_path} is empty or missing. "
                    f"Job status: {result.status}"
                )
            data = base64.b64decode(stdout)
            status, payload = cloudpickle.loads(data)
            # Unpack: ("ok", value) or ("error", exc, tb_str)
            if status == "ok":
                self._cached = payload
                self._resolved = True
                return payload
            # error
            exc = payload
            raise exc

    def cancel(self) -> None:
        """Cancel the underlying SLURM job."""
        from slurp.client import SyncClient

        with SyncClient(profile=self.profile) as client:
            client.cancel_job_by_id(self.job_id)


class ObjectRef:
    """Reference to a shared object created by ``slurp.put()``.

    Pickled alongside task arguments. The worker resolves it by loading
    the file from the shared filesystem.
    """

    def __init__(self, object_id: str, working_dir: str) -> None:
        if working_dir:
            self._slurp_object_path = f"{working_dir}/{_OBJECT_DIR}/{object_id}.pkl"
        else:
            self._slurp_object_path = f"{_OBJECT_DIR}/{object_id}.pkl"


class Dispatcher(ABC):
    """Abstract base: decides how .distribute()/.remote() are executed."""

    @abstractmethod
    def dispatch(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        resources: dict[str, Any],
        *,
        collect_result: bool,
    ) -> Ref | None:
        """Submit a function call.

        Args:
            collect_result: If True, the worker writes a result file and
                a Ref is returned. If False, fire-and-forget (returns None).
        """

    @abstractmethod
    def join(self) -> None:
        """Block until all pending fire-and-forget calls finish."""

    @abstractmethod
    def is_local(self) -> bool:
        """True if calls run in-process (no SLURM)."""


class LocalDispatcher(Dispatcher):
    """Calls functions directly. For testing / SLURM-less environments."""

    def __init__(self) -> None:
        self._pending: list[tuple[Any, Any]] = []

    def dispatch(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        resources: dict[str, Any],  # noqa: ARG002
        *,
        collect_result: bool,
    ) -> Ref | None:
        logger.info("local_dispatch", func=getattr(func, "__name__", str(func)))
        result = func(*args, **kwargs)
        if not collect_result:
            self._pending.append((None, None))
            return None
        # Return a fake Ref that holds the result directly
        ref = Ref(
            job_id="local",
            result_id=None,
            profile="local",
            working_dir="",
        )
        ref._cached = result  # noqa: SLF001
        ref._resolved = True  # noqa: SLF001
        return ref

    def join(self) -> None:
        self._pending.clear()

    def is_local(self) -> bool:
        return True


class SlurmDispatcher(Dispatcher):
    """Submits via SyncClient.submit(). Uses sbatch + slurp.worker."""

    def __init__(self, profile: str | None = None) -> None:
        self._profile = profile
        self._pending_job_ids: list[str] = []

    def dispatch(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        resources: dict[str, Any],
        *,
        collect_result: bool,
    ) -> Ref | None:
        from slurp.client import SyncClient

        client = SyncClient(profile=self._profile)
        profile = client.profile

        # Determine local + remote working dir
        if profile.sync and profile.sync.remote:
            working_dir = profile.format_remote()
            local_dir = Path(profile.sync.local) if profile.sync.local else Path.cwd()
        else:
            working_dir = str(Path.cwd())
            local_dir = Path.cwd()

        # Serialize — write to local_dir so the payload is inside the
        # synced directory and reaches the remote via rsync.
        payload_id = _write_payload(func, args, kwargs, base_dir=local_dir)

        # Build command
        remote_payload = f"{_PAYLOAD_DIR}/{payload_id}.pkl"
        result_id = None
        if collect_result:
            result_id = payload_id  # reuse UUID
            remote_result = f"{_RESULT_DIR}/{result_id}.pkl"
            command = f"python -m slurp.worker {remote_payload} {remote_result}"
        else:
            command = f"python -m slurp.worker {remote_payload}"

        # Pop profile from resources if present (client already has it)
        resources.pop("profile", None)

        # Submit
        job = client.submit(command, **resources)

        if not collect_result:
            self._pending_job_ids.append(job.job_id)
            return None

        return Ref(
            job_id=job.job_id,
            result_id=result_id,
            profile=profile.name,
            working_dir=working_dir,
        )

    def join(self) -> None:
        if not self._pending_job_ids:
            return
        from slurp.client import SyncClient

        try:
            with SyncClient(profile=self._profile) as client:
                for job_id in self._pending_job_ids:
                    job = client.status(job_id)
                    if job is not None and not job.status.is_terminal:
                        client.wait_job(job)
        except KeyboardInterrupt:
            _cancel_job_ids(self._pending_job_ids, self._profile)
            raise
        self._pending_job_ids.clear()

    def is_local(self) -> bool:
        return False


class JobBundling(Dispatcher):
    """Buffers calls and flushes them as SLURM job arrays.

    Usage::

        with slurp.JobBundling(max_size=20):
            for i in range(100):
                train.distribute(i)
        slurp.join()

    Calls are grouped by their resource specification — only tasks with
    identical resources can be bundled into the same job array.
    """

    def __init__(self, max_size: int, *, profile: str | None = None) -> None:
        self.max_size = max_size
        self._profile = profile
        self._buffer: dict[
            tuple[str, ...], list[tuple[Any, tuple[Any, ...], dict[str, Any]]]
        ] = defaultdict(list)
        self._pending_job_ids: list[str] = []
        self._inner: Dispatcher | None = None

    def __enter__(self) -> JobBundling:
        self._inner = _get_dispatcher()
        _set_dispatcher(self)
        return self

    def __exit__(self, *exc: object) -> None:
        _set_dispatcher(self._inner)
        if exc[0] is None:
            self.flush()  # flush() handles KeyboardInterrupt internally
        elif isinstance(exc[0], KeyboardInterrupt) and self._pending_job_ids:
            # Ctrl+C inside the with block — cancel any previously flushed jobs.
            _cancel_job_ids(self._pending_job_ids, self._profile)

    def dispatch(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        resources: dict[str, Any],
        *,
        collect_result: bool,  # noqa: ARG002
    ) -> Ref | None:
        # Buffer the call, keyed by resource signature
        key = self._resource_key(resources)
        self._buffer[key].append((func, args, kwargs))
        return None

    def _resource_key(self, resources: dict[str, Any]) -> tuple[str, ...]:
        """Hashable key from resource dict for grouping."""
        return tuple(sorted(f"{k}={v}" for k, v in resources.items()))

    def flush(self) -> list[str]:
        """Submit all buffered calls as job arrays. Returns job IDs."""
        # If inner dispatcher is local, just dispatch each call directly
        if self._inner is not None and self._inner.is_local():
            for _key, calls in self._buffer.items():
                for func, args, kwargs in calls:
                    self._inner.dispatch(
                        func, args, kwargs, {}, collect_result=False
                    )
            self._buffer.clear()
            return []

        from slurp.client import SyncClient
        from slurp.core.slurm import generate_sbatch_script, sbatch_submit
        from slurp.core.ssh import SSHManager

        job_ids: list[str] = []
        # Tracks the most recently submitted job ID that hasn't been
        # appended to _pending_job_ids yet. If KeyboardInterrupt fires
        # in that window, the except handler cancels it. There remains a
        # 1-bytecode irreducible race (between sbatch_submit returning
        # and the STORE_FAST for job_id) that cannot be closed in pure
        # Python — the signal is processed between bytecodes.
        _untracked: str | None = None

        try:
            for _key, calls in self._buffer.items():
                if not calls:
                    continue

                # Chunk into groups of max_size
                for chunk_start in range(0, len(calls), self.max_size):
                    chunk = calls[chunk_start : chunk_start + self.max_size]

                    # Submit as a job array
                    client = SyncClient(profile=self._profile)
                    profile = client.profile

                    if profile.sync and profile.sync.remote:
                        working_dir = profile.format_remote()
                        local_dir = (
                            Path(profile.sync.local) if profile.sync.local else Path.cwd()
                        )
                    else:
                        working_dir = str(Path.cwd())
                        local_dir = Path.cwd()

                    # Write payloads under local_dir so they get synced to the
                    # remote and land at <remote>/._slurp/payloads/.
                    bundle_uuid = uuid.uuid4().hex
                    _ensure_dirs(local_dir)
                    for i, (func, args, kwargs) in enumerate(chunk):
                        payload_id = f"{bundle_uuid}_{i}"
                        path = local_dir / f"{_PAYLOAD_DIR}/{payload_id}.pkl"
                        with open(path, "wb") as f:
                            cloudpickle.dump((func, args, kwargs), f)

                    # Build the array command
                    n_tasks = len(chunk)
                    command = (
                        f"python -m slurp.worker "
                        f"{_PAYLOAD_DIR}/{bundle_uuid}_${{SLURM_ARRAY_TASK_ID}}.pkl"
                    )

                    # Sync repo + payloads to the remote working dir.
                    from slurp.core.sync import sync_to_remote

                    client._run(  # noqa: SLF001
                        sync_to_remote(
                            profile, local_dir, working_dir, ssh_manager=client._ssh  # noqa: SLF001
                        )
                    )

                    # Venv sync (login node, after rsync so uv.lock is present remotely).
                    client._sync_venv(working_dir)  # noqa: SLF001

                    resources_obj = ResourceRequest(
                        gpus=0,
                        nodes=1,
                        time="2:00:00",
                        partition=profile.partition,
                        account=profile.account,
                    )

                    script = generate_sbatch_script(
                        resources=resources_obj,
                        profile=profile,
                        command=command,
                        working_dir=working_dir,
                        array_size=n_tasks,
                    )

                    ssh = SSHManager()
                    _untracked = None
                    job_id = client._run(  # noqa: SLF001
                        sbatch_submit(profile, script, working_dir=working_dir, ssh_manager=ssh)
                    )
                    # Track for cleanup BEFORE anything else — if
                    # KeyboardInterrupt fires between sbatch_submit
                    # returning and this append, _untracked holds the
                    # ID and the except handler cancels it.
                    _untracked = job_id
                    self._pending_job_ids.append(job_id)
                    _untracked = None
                    job_ids.append(job_id)
                    logger.info(
                        "bundled_submit",
                        job_id=job_id,
                        tasks=n_tasks,
                    )
        except KeyboardInterrupt:
            if _untracked is not None:
                _cancel_job_ids([_untracked], self._profile)
            _cancel_job_ids(self._pending_job_ids, self._profile)
            raise
        finally:
            self._buffer.clear()
        return job_ids

    def join(self) -> None:
        self.flush()
        if self._pending_job_ids:
            from slurp.client import SyncClient

            try:
                with SyncClient(profile=self._profile) as client:
                    for job_id in self._pending_job_ids:
                        job = client.status(job_id)
                        if job is not None and not job.status.is_terminal:
                            client.wait_job(job)
            except KeyboardInterrupt:
                _cancel_job_ids(self._pending_job_ids, self._profile)
                raise
            self._pending_job_ids.clear()

    def is_local(self) -> bool:
        return self._inner.is_local() if self._inner else False


# ── Global dispatcher management ──────────────────────────────────

_dispatcher: Dispatcher | None = None


def get_dispatcher() -> Dispatcher:
    """Get the current dispatcher, creating one if needed."""
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher

    # Auto-detect: try SLURM, fall back to local
    try:
        _dispatcher = SlurmDispatcher()
        logger.info("dispatcher", kind="slurm")
    except Exception:
        _dispatcher = LocalDispatcher()
        logger.info("dispatcher", kind="local")
    return _dispatcher


def set_dispatcher(d: Dispatcher) -> None:
    """Set the global dispatcher."""
    global _dispatcher
    _dispatcher = d


def _get_dispatcher() -> Dispatcher:
    return get_dispatcher()


def _set_dispatcher(d: Dispatcher | None) -> None:
    global _dispatcher
    if d is not None:
        _dispatcher = d


__all__ = [
    "Dispatcher",
    "SlurmDispatcher",
    "LocalDispatcher",
    "JobBundling",
    "Ref",
    "ObjectRef",
    "get_dispatcher",
    "set_dispatcher",
]
