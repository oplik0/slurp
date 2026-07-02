"""Guard: prevents recursive distribution from inside a running SLURM job.

When a ``@task``-decorated function runs on a compute node, calling
``.distribute()`` or ``.remote()`` from within it would submit *another*
SLURM job — which would submit another — ad infinitum.  This module
detects that situation via ``$SLURM_JOB_ID`` and raises a clear error
instead of letting the cluster eat itself.

The guard can be explicitly bypassed for legitimate recursive submission
(e.g. a coordinator task that spawns sub-tasks) via
``slurp.guard.disable()``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

_ENABLED = True


def is_inside_slurm_job() -> bool:
    """Return True if we are currently running inside a SLURM job."""
    return bool(os.environ.get("SLURM_JOB_ID"))


def check() -> None:
    """Raise RuntimeError if we are inside a SLURM job and the guard is active.

    Called by ``TaskFunction.distribute()`` and ``.remote()`` before
    submitting.
    """
    if _ENABLED and is_inside_slurm_job():
        raise RuntimeError(
            "Recursive distribution detected: .distribute()/.remote() called "
            "from inside SLURM job "
            f"{os.environ['SLURM_JOB_ID']}. "
            "This would submit nested jobs recursively. "
            "If this is intentional, wrap the call in "
            "`with slurp.guard.disabled():`."
        )


@contextmanager
def disabled() -> Iterator[None]:
    """Temporarily disable the recursive distribution guard.

    Example::

        @slurp.task(gpus=1)
        def coordinator():
            # This task runs on SLURM and spawns sub-tasks.
            with slurp.guard.disabled():
                for item in work_items:
                    worker.distribute(item)
    """
    global _ENABLED
    old = _ENABLED
    _ENABLED = False
    try:
        yield
    finally:
        _ENABLED = old


# Re-exported as slurp.guard.disable for ergonomics
disable = disabled

# Expose everything for `from slurp.guard import *` or `slurp.guard.check()`
__all__ = ["check", "disabled", "disable", "is_inside_slurm_job"]
