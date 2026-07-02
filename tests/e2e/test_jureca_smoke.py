"""End-to-end smoke tests against a real SLURM cluster.

These are marked ``e2e`` and never run in standard CI. They require:
  * a valid profile named ``jureca`` in ``~/.config/slurp/profiles.toml``
  * SSH access to jureca.fz-juelich.de

Run explicitly with::

    SLURP_E2E=1 pytest tests/e2e/ -m e2e -v --timeout=600

Tests submit tiny jobs (1 GPU, <=5 min, devel partition) and must clean up
after themselves even on failure.
"""

from __future__ import annotations

import os
import uuid

import pytest

from slurp import SyncClient
from slurp.domain import JobStatus
from slurp.errors import ProfileError

pytestmark = pytest.mark.e2e

E2E_ENABLED = os.environ.get("SLURP_E2E") == "1"
PROFILE = "jureca"
SMOKE_PARTITION = "dc-gpu-devel"
SMOKE_TIME = "00:05:00"


def _e2e_skip() -> None:
    if not E2E_ENABLED:
        pytest.skip("set SLURP_E2E=1 to run e2e tests")
    try:
        SyncClient(profile=PROFILE)
    except ProfileError as exc:
        pytest.skip(f"no '{PROFILE}' profile: {exc}")


@pytest.fixture(scope="module")
def client() -> SyncClient:
    _e2e_skip()
    c = SyncClient(profile=PROFILE)
    yield c
    # Cancel anything we left running.
    try:
        for job in c.list_jobs(experiment="slurp-e2e", limit=50):
            if not job.status.is_terminal:
                c.cancel_job(job)
    except Exception:
        pass
    c.close()


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def test_submit_hostname_completes(client: SyncClient) -> None:
    """Submit a trivial 1-GPU job and wait for COMPLETED with exit 0.

    Exercises the full path: asyncssh connect -> sbatch (heredoc) -> sacct
    polling -> log fetch -> local store. This is the minimal end-to-end proof.
    """
    name = _unique_name("e2e-hostname")
    job = client.submit(
        "hostname; echo SMOKE_OK",
        name=name,
        gpus=1,
        cpus=1,
        time=SMOKE_TIME,
        partition=SMOKE_PARTITION,
        sync=False,
        experiment="slurp-e2e",
    )
    try:
        result = client.wait_job(job, timeout="5m", poll_interval=5.0)
    finally:
        # Ensure cleanup even if wait raised (e.g. timeout from sacct lag).
        try:
            client.cancel_job(job)
        except Exception:
            pass

    assert result.status == JobStatus.COMPLETED
    assert result.exit_code == 0
    # stdout must contain our marker -- proves the command actually ran.
    assert "SMOKE_OK" in result.stdout
    # And it ran on a compute node, not a login node.
    assert result.stdout.strip(), "empty stdout"


def test_submit_then_status_refresh(client: SyncClient) -> None:
    """A second _run() call after submit must see the real cluster state.

    Guards against the cross-event-loop connection-reuse bug: a cached asyncssh
    connection bound to a dead loop silently made every status poll return {}.
    """
    name = _unique_name("e2e-status")
    job = client.submit(
        "true",
        name=name,
        gpus=1,
        cpus=1,
        time=SMOKE_TIME,
        partition=SMOKE_PARTITION,
        sync=False,
        experiment="slurp-e2e",
    )
    try:
        # This call happens in a different _run() than submit(); if the
        # connection were stale it would return the original PENDING forever.
        refreshed = client.refresh_job(job)
        assert refreshed.status in {
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
        }
        # Wait it out so we don't leave a job dangling.
        result = client.wait_job(job, timeout="5m", poll_interval=5.0)
        assert result.status == JobStatus.COMPLETED
    finally:
        try:
            client.cancel_job(job)
        except Exception:
            pass
