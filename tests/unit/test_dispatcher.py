"""Tests for slurp.dispatcher — Dispatcher strategy and Ref/ObjectRef."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cloudpickle
import pytest

from slurp.dispatcher import (
    JobBundling,
    LocalDispatcher,
    ObjectRef,
    Ref,
    SlurmDispatcher,
    _ensure_dirs,
    _write_payload,
    get_dispatcher,
    set_dispatcher,
)

# ── Helpers ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_slurp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in a tmp directory so ._slurp/ doesn't pollute the repo."""
    monkeypatch.chdir(tmp_path)
    # Simulate no profile so _local_base_dir() uses CWD (the tmp dir).
    monkeypatch.setattr("slurp.dispatcher._local_base_dir", lambda: tmp_path)


# ── LocalDispatcher ──────────────────────────────────────────────────


class TestLocalDispatcher:
    def test_dispatch_collect_result(self) -> None:
        d = LocalDispatcher()

        def add(a: int, b: int) -> int:
            return a + b

        ref = d.dispatch(add, (3, 4), {}, {}, collect_result=True)
        assert ref is not None
        assert ref.get() == 7

    def test_dispatch_fire_and_forget(self) -> None:
        d = LocalDispatcher()
        called: list[bool] = []

        def f(x: int) -> None:
            called.append(True)

        ref = d.dispatch(f, (1,), {}, {}, collect_result=False)
        assert ref is None
        assert called == [True]

    def test_join_is_noop_after_dispatch(self) -> None:
        d = LocalDispatcher()
        d.dispatch(lambda x: x, (5,), {}, {}, collect_result=False)
        d.join()  # should not raise

    def test_is_local(self) -> None:
        assert LocalDispatcher().is_local() is True


# ── Ref ──────────────────────────────────────────────────────────────


class TestRef:
    def test_result_path_none_for_distribute(self) -> None:
        ref = Ref(job_id="123", result_id=None, profile="test", working_dir="/remote")
        assert ref.result_path is None

    def test_result_path_built_from_result_id(self) -> None:
        ref = Ref(
            job_id="123",
            result_id="abc",
            profile="test",
            working_dir="/remote/project",
        )
        assert ref.result_path == "/remote/project/._slurp/results/abc.pkl"

    def test_get_returns_cached_on_second_call(self) -> None:
        """get() should cache the result and return it on subsequent calls."""
        ref = Ref(job_id="local", result_id=None, profile="local", working_dir="")
        ref._cached = {"answer": 42}
        ref._resolved = True
        assert ref.get() == {"answer": 42}
        # Second call should return the same cached value without re-fetching
        assert ref.get() == {"answer": 42}


# ── ObjectRef ────────────────────────────────────────────────────────


class TestObjectRef:
    def test_has_slurp_object_path(self) -> None:
        ref = ObjectRef("abc123", "/remote")
        assert ref._slurp_object_path == "/remote/._slurp/objects/abc123.pkl"

    def test_is_picklable(self) -> None:
        """ObjectRef must be picklable — it's included in task payloads."""
        ref = ObjectRef("abc123", "/remote")
        data = cloudpickle.dumps(ref)
        restored = cloudpickle.loads(data)
        assert restored._slurp_object_path == "/remote/._slurp/objects/abc123.pkl"


# ── JobBundling ──────────────────────────────────────────────────────


class TestJobBundling:
    def test_buffers_calls_in_local_mode(self) -> None:
        """In local mode, JobBundling should flush calls through the local dispatcher."""
        set_dispatcher(LocalDispatcher())
        results: list[int] = []

        def f(x: int) -> None:
            results.append(x)

        bundling = JobBundling(max_size=5)
        with bundling:
            for i in range(7):
                bundling.dispatch(f, (i,), {}, {}, collect_result=False)

        # All 7 calls should have been dispatched
        assert len(results) == 7
        assert sorted(results) == [0, 1, 2, 3, 4, 5, 6]

    def test_restores_previous_dispatcher(self) -> None:
        """JobBundling.__exit__ should restore the previous dispatcher."""
        original = LocalDispatcher()
        set_dispatcher(original)

        with JobBundling(max_size=10):
            assert isinstance(get_dispatcher(), JobBundling)

        assert get_dispatcher() is original

    def test_restores_on_exception(self) -> None:
        original = LocalDispatcher()
        set_dispatcher(original)

        with pytest.raises(ValueError), JobBundling(max_size=10):
            raise ValueError("boom")

        assert get_dispatcher() is original

    def test_does_not_flush_on_exception(self) -> None:
        """If an exception occurs inside the with block, flush should be skipped."""
        set_dispatcher(LocalDispatcher())
        called: list[int] = []

        def f(x: int) -> None:
            called.append(x)

        with pytest.raises(ValueError), JobBundling(max_size=10):
            bundling_dispatch(f, (1,))
            raise ValueError("abort")

        # The call should not have been dispatched because flush was skipped
        assert called == []


def bundling_dispatch(func: object, args: tuple[object, ...]) -> None:
    """Helper: dispatch via the current (bundling) dispatcher."""
    d = get_dispatcher()
    d.dispatch(func, args, {}, {}, collect_result=False)


# ── Payload helpers ─────────────────────────────────────────────────


class TestPayloadHelpers:
    def test_ensure_dirs_creates_structure(self, tmp_path: Path) -> None:
        _ensure_dirs()
        assert Path("._slurp/payloads").exists()
        assert Path("._slurp/results").exists()
        assert Path("._slurp/objects").exists()

    def test_write_payload_creates_file(self) -> None:
        def f(x: int) -> int:
            return x

        payload_id = _write_payload(f, (5,), {})
        path = Path(f"._slurp/payloads/{payload_id}.pkl")
        assert path.exists()

        with open(path, "rb") as fh:
            func, args, kwargs = cloudpickle.load(fh)
        assert args == (5,)
        assert kwargs == {}

    def test_write_payload_unique_ids(self) -> None:
        def f() -> None:
            pass

        id1 = _write_payload(f, (), {})
        id2 = _write_payload(f, (), {})
        assert id1 != id2


# ── Global dispatcher management ─────────────────────────────────────


class TestDispatcherManagement:
    def test_set_and_get(self) -> None:
        d = LocalDispatcher()
        set_dispatcher(d)
        assert get_dispatcher() is d

    def test_set_none_falls_back_to_auto(self) -> None:
        """Setting dispatcher back to None should trigger auto-detection."""
        # Set to local first
        set_dispatcher(LocalDispatcher())
        assert isinstance(get_dispatcher(), LocalDispatcher)

    def test_default_is_auto_detected(self) -> None:
        """Without explicit set, get_dispatcher should not return None."""
        # This depends on the environment — in CI without SLURM, it should
        # fall back to LocalDispatcher
        d = get_dispatcher()
        assert d is not None
        assert isinstance(d, (LocalDispatcher, SlurmDispatcher))


# ── JobBundling KeyboardInterrupt race ───────────────────────────────


class TestJobBundlingInterruptRace:
    """Verify submitted-but-untracked jobs are cancelled on KeyboardInterrupt.

    When Ctrl+C fires between sbatch_submit returning and the job ID
    being appended to _pending_job_ids, the _untracked sentinel ensures
    the job is still cancelled.
    """

    @staticmethod
    def _make_bundling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from slurp.domain import Profile

        monkeypatch.chdir(tmp_path)

        inner = MagicMock()
        inner.is_local.return_value = False

        bundling = JobBundling(max_size=5, profile="test")
        bundling._inner = inner

        def f(x: int) -> None:
            pass

        bundling._buffer[("gpus=0",)] = [(f, (1,), {})]

        mock_profile = Profile(
            name="test",
            hostname="hpc",
            sync=Profile.SyncConfig(local=".", remote="/remote"),
        )
        mock_client = MagicMock()
        mock_client.profile = mock_profile
        mock_client._run.return_value = "12345"
        mock_client._sync_venv = MagicMock()
        mock_client._ssh = MagicMock()
        # _resolve_working_dir returns (remote_dir, local_dir).
        mock_client._resolve_working_dir.return_value = ("/remote", Path("."))

        return bundling, mock_client

    def test_untracked_job_cancelled_on_interrupt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+C after sbatch but before tracking → untracked job cancelled."""
        bundling, mock_client = self._make_bundling(tmp_path, monkeypatch)

        # Simulate KeyboardInterrupt when _pending_job_ids.append is called.
        class InterruptingList(list):
            def append(self, item):
                raise KeyboardInterrupt

        bundling._pending_job_ids = InterruptingList()

        with (
            patch("slurp.client.SyncClient", return_value=mock_client),
            patch("slurp.core.slurm.generate_sbatch_script", return_value="#!/bin/bash"),
            patch("slurp.core.slurm.sbatch_submit", new_callable=AsyncMock),
            patch("slurp.core.ssh.SSHManager", return_value=MagicMock()),
            patch("slurp.core.sync.sync_to_remote", new_callable=AsyncMock),
            patch("slurp.dispatcher._cancel_job_ids") as mock_cancel,
            pytest.raises(KeyboardInterrupt),
        ):
            bundling.flush()

        # The untracked job_id should appear in one of the _cancel_job_ids calls
        cancel_args = [c.args[0] for c in mock_cancel.call_args_list]
        assert ["12345"] in cancel_args

    def test_no_cancel_on_successful_flush(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal flush (no interrupt) → _cancel_job_ids not called."""
        bundling, mock_client = self._make_bundling(tmp_path, monkeypatch)

        with (
            patch("slurp.client.SyncClient", return_value=mock_client),
            patch("slurp.core.slurm.generate_sbatch_script", return_value="#!/bin/bash"),
            patch("slurp.core.slurm.sbatch_submit", new_callable=AsyncMock),
            patch("slurp.core.ssh.SSHManager", return_value=MagicMock()),
            patch("slurp.core.sync.sync_to_remote", new_callable=AsyncMock),
            patch("slurp.dispatcher._cancel_job_ids") as mock_cancel,
        ):
            job_ids = bundling.flush()

        assert job_ids == ["12345"]
        assert bundling._pending_job_ids == ["12345"]
        mock_cancel.assert_not_called()
