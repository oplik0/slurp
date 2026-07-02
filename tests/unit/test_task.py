"""Tests for slurp.task — @task decorator, TaskFunction, get/join/put."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cloudpickle
import pytest

import slurp
from slurp.dispatcher import LocalDispatcher, Ref, set_dispatcher
from slurp.task import TaskFunction

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _local_dispatcher() -> None:
    """Use LocalDispatcher for all tests — no SLURM needed."""
    set_dispatcher(LocalDispatcher())


@pytest.fixture(autouse=True)
def _isolate_slurp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in a tmp directory."""
    monkeypatch.chdir(tmp_path)
    # Simulate no profile so _local_base_dir() uses CWD (the tmp dir).
    monkeypatch.setattr("slurp.dispatcher._local_base_dir", lambda: tmp_path)


# ── Decorator ────────────────────────────────────────────────────────


class TestTaskDecorator:
    def test_no_parens(self) -> None:
        @slurp.task
        def f(x: int) -> int:
            return x

        assert isinstance(f, TaskFunction)
        assert f(5) == 5

    def test_with_resources(self) -> None:
        @slurp.task(gpus=4, time="1:00:00")
        def f(x: int) -> int:
            return x

        assert isinstance(f, TaskFunction)
        assert f._resources == {"gpus": 4, "time": "1:00:00"}

    def test_preserves_function_name(self) -> None:
        @slurp.task
        def my_function(x: int) -> int:
            return x

        assert my_function.__wrapped__.__name__ == "my_function"  # type: ignore[attr-defined]

    def test_preserves_docstring(self) -> None:
        @slurp.task
        def documented(x: int) -> int:
            """A documented function."""
            return x

        assert documented.__doc__ == "A documented function."


# ── Direct call ─────────────────────────────────────────────────────


class TestDirectCall:
    def test_runs_locally(self) -> None:
        @slurp.task(gpus=4)
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_runs_with_kwargs(self) -> None:
        @slurp.task
        def greet(name: str, greeting: str = "hello") -> str:
            return f"{greeting}, {name}"

        assert greet(name="world") == "hello, world"
        assert greet("alice", greeting="hi") == "hi, alice"

    def test_exception_propagates(self) -> None:
        @slurp.task
        def boom() -> None:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            boom()


# ── .distribute() ───────────────────────────────────────────────────


class TestDistribute:
    def test_fire_and_forget_local(self) -> None:
        called: list[int] = []

        @slurp.task(gpus=1)
        def f(x: int) -> None:
            called.append(x)

        f.distribute(x=42)
        assert called == [42]

    def test_distribute_with_args_and_kwargs(self) -> None:
        calls: list[tuple[int, int, int]] = []

        @slurp.task
        def f(a: int, b: int, c: int = 0) -> None:
            calls.append((a, b, c))

        f.distribute(1, 2, c=3)
        assert calls == [(1, 2, 3)]

    def test_distribute_returns_none(self) -> None:
        @slurp.task
        def f() -> None:
            pass

        f.distribute()  # returns None
        assert True


# ── .remote() ───────────────────────────────────────────────────────


class TestRemote:
    def test_returns_ref(self) -> None:
        @slurp.task(gpus=1)
        def square(x: int) -> int:
            return x * x

        ref = square.remote(x=5)
        assert isinstance(ref, Ref)

    def test_get_returns_value(self) -> None:
        @slurp.task(gpus=1)
        def square(x: int) -> int:
            return x * x

        ref = square.remote(x=7)
        result = slurp.get(ref)
        assert result == 49

    def test_get_with_timeout(self) -> None:
        @slurp.task
        def slow() -> int:
            return 42

        ref = slow.remote()
        result = slurp.get(ref, timeout=10)
        assert result == 42

    def test_get_list_of_refs(self) -> None:
        @slurp.task
        def double(x: int) -> int:
            return x * 2

        refs = [double.remote(x=i) for i in range(3)]
        results = slurp.get(refs)
        assert results == [0, 2, 4]

    def test_get_caches_result(self) -> None:
        call_count: list[int] = []

        @slurp.task
        def f(x: int) -> int:
            call_count.append(1)
            return x

        ref = f.remote(x=10)
        r1 = slurp.get(ref)
        r2 = slurp.get(ref)
        assert r1 == r2 == 10
        # In local mode, the function is called once per .remote()
        assert len(call_count) == 1


# ── .remote_batch() ─────────────────────────────────────────────────


class TestRemoteBatch:
    def test_local_batch(self) -> None:
        @slurp.task(gpus=1)
        def square(x: int) -> int:
            return x * x

        refs = square.remote_batch([{"x": 1}, {"x": 2}, {"x": 3}])
        assert len(refs) == 3
        results = slurp.get(refs)
        assert results == [1, 4, 9]

    def test_empty_configs_raises(self) -> None:
        @slurp.task
        def f(x: int) -> int:
            return x

        # remote_batch with empty list should produce empty refs
        refs = f.remote_batch([])
        assert refs == []


# ── .options() ──────────────────────────────────────────────────────


class TestOptions:
    def test_returns_new_task_function(self) -> None:
        @slurp.task(gpus=1)
        def f(x: int) -> int:
            return x

        g = f.options(gpus=8)
        assert isinstance(g, TaskFunction)
        assert g is not f

    def test_overrides_resources(self) -> None:
        @slurp.task(gpus=1, time="1:00:00")
        def f(x: int) -> int:
            return x

        g = f.options(gpus=8)
        assert g._resources["gpus"] == 8
        assert g._resources["time"] == "1:00:00"  # original preserved

    def test_with_options_alias(self) -> None:
        @slurp.task(gpus=1)
        def f(x: int) -> int:
            return x

        g = f.with_options(gpus=4)
        assert g._resources["gpus"] == 4

    def test_options_then_remote(self) -> None:
        @slurp.task(gpus=1)
        def square(x: int) -> int:
            return x * x

        ref = square.options(gpus=8).remote(x=5)
        assert slurp.get(ref) == 25


# ── join() ──────────────────────────────────────────────────────────


class TestJoin:
    def test_join_after_distribute(self) -> None:
        called: list[int] = []

        @slurp.task
        def f(x: int) -> None:
            called.append(x)

        f.distribute(x=1)
        f.distribute(x=2)
        slurp.join()  # should not raise
        assert sorted(called) == [1, 2]

    def test_join_with_no_pending(self) -> None:
        slurp.join()  # should not raise


# ── put() ───────────────────────────────────────────────────────────


class TestPut:
    def test_creates_file(self) -> None:
        obj = {"weights": [1, 2, 3], "name": "model"}
        ref = slurp.put(obj)
        assert hasattr(ref, "_slurp_object_path")
        assert Path(ref._slurp_object_path).exists()

    def test_file_contains_object(self) -> None:
        obj = [1, 2, 3]
        ref = slurp.put(obj)
        with open(ref._slurp_object_path, "rb") as f:
            loaded = cloudpickle.load(f)
        assert loaded == obj

    def test_unique_ids(self) -> None:
        ref1 = slurp.put("a")
        ref2 = slurp.put("b")
        assert ref1._slurp_object_path != ref2._slurp_object_path

    def test_object_ref_is_picklable(self) -> None:
        """ObjectRef must survive cloudpickle (it's included in payloads)."""
        ref = slurp.put({"x": 1})
        data = cloudpickle.dumps(ref)
        restored = cloudpickle.loads(data)
        assert restored._slurp_object_path == ref._slurp_object_path


# ── Guard integration ───────────────────────────────────────────────


class TestGuardIntegration:
    def test_distribute_raises_inside_slurm(self) -> None:
        """distribute() should raise when inside a SLURM job (guard active)."""
        import os

        @slurp.task
        def f(x: int) -> None:
            pass

        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "123"}),
            pytest.raises(RuntimeError, match="Recursive distribution"),
        ):
            f.distribute(x=1)

    def test_remote_raises_inside_slurm(self) -> None:
        import os

        @slurp.task
        def f(x: int) -> int:
            return x

        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "456"}),
            pytest.raises(RuntimeError, match="Recursive distribution"),
        ):
            f.remote(x=1)

    def test_guard_disabled_allows_distribute(self) -> None:
        import os

        @slurp.task
        def f(x: int) -> None:
            pass

        with patch.dict(os.environ, {"SLURM_JOB_ID": "789"}), slurp.guard.disabled():
            f.distribute(x=1)  # should not raise


# ── JobBundling integration ────────────────────────────────────────


class TestJobBundlingIntegration:
    def test_bundling_dispatches_all(self) -> None:
        results: list[int] = []

        @slurp.task(gpus=1)
        def process(x: int) -> None:
            results.append(x)

        with slurp.JobBundling(max_size=3):
            for i in range(7):
                process.distribute(x=i)

        slurp.join()
        assert sorted(results) == list(range(7))

    def test_bundling_restores_dispatcher(self) -> None:
        before = slurp.get_dispatcher()

        with slurp.JobBundling(max_size=10):
            pass  # no calls

        assert slurp.get_dispatcher() is before
