"""@task decorator and module-level helpers.

This is the high-level API layering on top of :mod:`slurp.dispatcher` and
:mod:`slurp.client`.  It provides three usage patterns from one decorator:

**Slurminade-style (fire-and-forget)**::

    @slurp.task(gpus=1)
    def train(lr, epochs):
        ...

    train.distribute(lr=0.001, epochs=100)
    slurp.join()

**Ray-style (with result)**::

    ref = train.remote(lr=0.001, epochs=100)
    result = slurp.get(ref)

**Batch (job array)**::

    refs = train.remote_batch([{"lr": "0.001"}, {"lr": "0.01"}])
    results = slurp.get(refs)

**Local call (testing)**::

    train(lr=0.001, epochs=3)  # runs in-process, no SLURM
"""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, overload

import cloudpickle
import structlog

from slurp.dispatcher import (
    _OBJECT_DIR,
    _RESULT_DIR,
    ObjectRef,
    Ref,
    _ensure_dirs,
    get_dispatcher,
)
from slurp.guard import check as guard_check

logger = structlog.get_logger()

P = ParamSpec("P")
R = TypeVar("R")


class TaskFunction:
    """Wrapper around a function that can be called locally or submitted to SLURM.

    Created by the ``@slurp.task`` decorator.  The wrapper stores resource
    defaults and provides three submission modes:

    - ``__call__`` — direct local call (for testing)
    - ``.distribute()`` — fire-and-forget submit (slurminade-style)
    - ``.remote()`` — submit and return a :class:`Ref` (Ray-style)
    - ``.remote_batch()`` — submit as a job array, return list of Refs
    """

    def __init__(
        self,
        func: Callable[..., Any],
        resources: dict[str, Any] | None = None,
    ) -> None:
        self._func = func
        self._resources = resources or {}
        functools.update_wrapper(self, func, updated=())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the function directly (local, in-process)."""
        return self._func(*args, **kwargs)

    def _merge_resources(self, overrides: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self._resources)
        merged.update(overrides)
        return merged

    def distribute(self, *args: Any, **kwargs: Any) -> None:
        """Submit fire-and-forget. Returns immediately, no result retrieval.

        The function's return value is discarded. Write results to files
        inside the function if you need them later.
        """
        guard_check()
        dispatcher = get_dispatcher()
        dispatcher.dispatch(
            self._func,
            args,
            kwargs,
            self._resources,
            collect_result=False,
        )

    def remote(self, *args: Any, **kwargs: Any) -> Ref:
        """Submit and return a :class:`Ref` for result retrieval.

        Use ``slurp.get(ref)`` to block and retrieve the return value.
        """
        guard_check()
        dispatcher = get_dispatcher()
        ref = dispatcher.dispatch(
            self._func,
            args,
            kwargs,
            self._resources,
            collect_result=True,
        )
        if ref is None:
            # LocalDispatcher returns a pre-resolved Ref
            from slurp.dispatcher import LocalDispatcher

            if isinstance(dispatcher, LocalDispatcher):
                # This shouldn't happen — LocalDispatcher returns Ref for collect
                raise RuntimeError("LocalDispatcher returned None for collect_result=True")
        return ref  # type: ignore[return-value]

    def remote_batch(
        self,
        configs: list[dict[str, Any]],
        **resource_overrides: Any,
    ) -> list[Ref]:
        """Submit as a SLURM job array. One sbatch call, N tasks.

        Each config dict maps placeholder names to values. The function
        is called with each config as keyword arguments.

        Returns one :class:`Ref` per task.
        """
        guard_check()
        dispatcher = get_dispatcher()

        # For local dispatcher, just call each one
        if dispatcher.is_local():
            refs: list[Ref] = []
            for cfg in configs:
                ref = dispatcher.dispatch(
                    self._func,
                    (),
                    cfg,
                    self._resources,
                    collect_result=True,
                )
                if ref is None:
                    raise RuntimeError("LocalDispatcher returned None for collect_result=True")
                refs.append(ref)
            return refs

        # For SLURM dispatcher, submit as a job array
        return self._submit_array(configs, **resource_overrides)

    def _submit_array(
        self,
        configs: list[dict[str, Any]],
        **resource_overrides: Any,
    ) -> list[Ref]:
        """Submit configs as a SLURM job array via the worker."""
        from slurp.client import SyncClient
        from slurp.core.slurm import generate_sbatch_script, sbatch_submit
        from slurp.core.ssh import SSHManager
        from slurp.dispatcher import _PAYLOAD_DIR
        from slurp.domain import ResourceRequest

        merged = self._merge_resources(resource_overrides)

        client = SyncClient(profile=merged.pop("profile", None))
        profile = client.profile

        if profile.sync and profile.sync.remote:
            working_dir = profile.format_remote()
            local_dir = Path(profile.sync.local) if profile.sync.local else Path.cwd()
        else:
            working_dir = str(Path.cwd())
            local_dir = Path.cwd()

        # Write a payload for each config BEFORE sync so they reach the remote.
        # Payloads go under local_dir (not CWD) so they're inside the synced
        # directory and land at <remote>/._slurp/payloads/ after rsync.
        batch_uuid = uuid.uuid4().hex
        _ensure_dirs(local_dir)
        for i, cfg in enumerate(configs):
            payload_id = f"{batch_uuid}_{i}"
            path = Path(local_dir) / f"{_PAYLOAD_DIR}/{payload_id}.pkl"
            with open(path, "wb") as f:
                cloudpickle.dump((self._func, (), dict(cfg)), f)

        # Sync repo + payloads to the remote working dir. Without this the
        # worker has no code and no payloads (the sbatch script cd's into a
        # remote dir that was previously empty).
        from slurp.core.sync import sync_to_remote

        client._run(  # noqa: SLF001
            sync_to_remote(profile, local_dir, working_dir, ssh_manager=client._ssh)  # noqa: SLF001
        )

        # Venv sync (login node, after rsync so uv.lock is present remotely).
        client._sync_venv(working_dir)

        n_tasks = len(configs)
        command = (
            f"python -m slurp.worker "
            f"{_PAYLOAD_DIR}/{batch_uuid}_${{SLURM_ARRAY_TASK_ID}}.pkl "
            f"{_RESULT_DIR}/{batch_uuid}_${{SLURM_ARRAY_TASK_ID}}.pkl"
        )

        resources = ResourceRequest(
            gpus=merged.pop("gpus", 0),
            nodes=merged.pop("nodes", 1),
            time=merged.pop("time", "2:00:00") or "2:00:00",
            mem=merged.pop("mem", None),
            cpus=merged.pop("cpus", 8),
            partition=merged.pop("partition", None) or profile.partition,
            account=merged.pop("account", None) or profile.account,
            constraint=merged.pop("constraint", None),
            qos=merged.pop("qos", None),
            job_name=merged.pop("name", None),
            slurm_kwargs=merged.pop("slurm_kwargs", None) or {},
        )

        throttle = merged.pop("throttle", 20)
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command=command,
            working_dir=working_dir,
            array_size=n_tasks,
            throttle=throttle,
        )

        ssh = SSHManager()
        job_id = client._run(  # noqa: SLF001
            sbatch_submit(profile, script, working_dir=working_dir, ssh_manager=ssh)
        )

        logger = structlog.get_logger()
        logger.info("array_submitted", job_id=job_id, tasks=n_tasks)

        # Create one Ref per task
        refs = []
        for i in range(n_tasks):
            refs.append(
                Ref(
                    job_id=f"{job_id}_{i}",
                    result_id=f"{batch_uuid}_{i}",
                    profile=profile.name,
                    working_dir=working_dir,
                )
            )
        return refs

    def options(self, **overrides: Any) -> TaskFunction:
        """Return a copy with overridden resource defaults.

        Similar to Ray's ``.options()`` — allows per-call resource
        overrides without changing the decorator defaults::

            train.options(gpus=8).remote(lr=0.001)
        """
        new_resources = self._merge_resources(overrides)
        return TaskFunction(self._func, new_resources)

    def with_options(self, **overrides: Any) -> TaskFunction:
        """Alias for :meth:`options` (slurminade compatibility)."""
        return self.options(**overrides)


# ── Module-level helpers ──────────────────────────────────────────


def get(
    refs: Ref | list[Ref],
    *,
    timeout: str | float | None = None,
) -> Any:
    """Block until result(s) are ready and return the deserialized value(s).

    If a single Ref is passed, returns a single value.
    If a list of Refs is passed, returns a list of values.

    If the user presses Ctrl+C while waiting, all pending jobs are
    scancelled before :class:`KeyboardInterrupt` propagates.

    Raises:
        The original exception if the function failed on the remote.
        TimeoutError if a job doesn't finish within *timeout*.
    """
    if isinstance(refs, Ref):
        return refs.get(timeout=timeout)
    try:
        return [r.get(timeout=timeout) for r in refs]
    except KeyboardInterrupt:
        _cancel_refs(refs)
        raise


def _cancel_refs(refs: list[Ref]) -> None:
    """Best-effort scancel all pending jobs on KeyboardInterrupt.

    Skips refs that have already been resolved (completed successfully).
    """
    from slurp.client import SyncClient

    pending = [r for r in refs if r.job_id and not r._resolved]  # noqa: SLF001
    if not pending:
        return
    logger.info(
        "keyboard_interrupt_cancelling",
        count=len(pending),
        job_ids=[r.job_id for r in pending],
    )
    try:
        with SyncClient(profile=pending[0].profile) as client:
            for ref in pending:
                try:
                    client.cancel_job_by_id(ref.job_id)
                except Exception:
                    pass  # Best effort per job
    except Exception:
        pass  # Best effort — never mask the KeyboardInterrupt


def join() -> None:
    """Block until all pending ``.distribute()`` calls finish.

    Does not raise on job failure — check ``slurp.list_jobs()`` or
    ``slurp logs <job_id>`` for error details.
    """
    get_dispatcher().join()


def put(obj: Any) -> ObjectRef:
    """Serialize an object to a file for sharing across tasks.

    The object is pickled to ``._slurp/objects/<uuid>.pkl`` under the
    synced directory (``profile.sync.local``).  When passed as an argument
    to ``.remote()``, the worker loads it transparently.

    This is the SLURM equivalent of Ray's object store — it uses the shared
    filesystem instead of in-memory transfer.  For large objects (model
    weights), prefer passing file paths directly.
    """
    object_id = uuid.uuid4().hex
    from slurp.dispatcher import _local_base_dir

    base = _local_base_dir()
    _ensure_dirs(base)
    path = base / f"{_OBJECT_DIR}/{object_id}.pkl"
    with open(path, "wb") as f:
        cloudpickle.dump(obj, f)

    logger.info("object_put", object_id=object_id, path=str(path))

    # ObjectRef uses a relative path because the worker cd's to the
    # working directory before executing.
    return ObjectRef(object_id, "")


# ── The decorator ──────────────────────────────────────────────────


@overload
def task(func: Callable[P, R]) -> TaskFunction: ...


@overload
def task(
    **resources: Any,
) -> Callable[[Callable[P, R]], TaskFunction]: ...


def task(
    func: Callable[..., Any] | None = None,
    **resources: Any,
) -> Any:
    """Decorate a function so it can be distributed to SLURM.

    Usage::

        @slurp.task(gpus=4, time="2:00:00")
        def train(lr, epochs):
            ...

        # Local call (testing)
        train(lr=0.001, epochs=3)

        # Fire-and-forget
        train.distribute(lr=0.001, epochs=100)
        slurp.join()

        # With result
        ref = train.remote(lr=0.001, epochs=100)
        result = slurp.get(ref)

    Args:
        **resources: Resource defaults (gpus, nodes, cpus, mem, time,
            partition, account, constraint, qos, etc.)
    """
    if func is not None:
        # Used as @slurp.task without parens
        return TaskFunction(func)

    # Used as @slurp.task(gpus=4, ...)
    def decorator(f: Callable[..., Any]) -> TaskFunction:
        return TaskFunction(f, resources)

    return decorator


__all__ = [
    "task",
    "TaskFunction",
    "Ref",
    "ObjectRef",
    "get",
    "join",
    "put",
]
