"""Tests for slurp.worker — compute-node entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import pytest

from slurp.worker import _resolve_arg, _write_result, main


class TestWriteResult:
    def test_writes_pickle(self, tmp_path: Path) -> None:
        path = str(tmp_path / "result.pkl")
        _write_result(path, ("ok", {"loss": 0.01}))
        assert Path(path).exists()
        with open(path, "rb") as f:
            data = cloudpickle.load(f)
        assert data == ("ok", {"loss": 0.01})

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = str(tmp_path / "nested" / "deep" / "result.pkl")
        _write_result(path, ("ok", 42))
        assert Path(path).exists()


class TestResolveArg:
    def test_passthrough_for_plain_value(self) -> None:
        assert _resolve_arg(42) == 42
        assert _resolve_arg("hello") == "hello"

    def test_resolves_object_ref(self, tmp_path: Path) -> None:
        """ObjectRef should be resolved to the actual object."""
        obj_path = tmp_path / "obj.pkl"
        obj = {"model": "resnet", "layers": 50}
        with open(obj_path, "wb") as f:
            cloudpickle.dump(obj, f)

        # Simulate an ObjectRef
        class FakeObjectRef:
            _slurp_object_path = str(obj_path)

        result = _resolve_arg(FakeObjectRef())
        assert result == obj

    def test_none_passthrough(self) -> None:
        assert _resolve_arg(None) is None


class TestWorkerMain:
    def test_fire_and_forget_mode(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Worker without result_path should run the function and exit 0."""
        payload_path = tmp_path / "payload.pkl"
        marker_path = tmp_path / "called.txt"

        def my_func(x: int) -> int:
            marker_path.write_text(f"called:{x}")
            return x * 2

        with open(payload_path, "wb") as f:
            cloudpickle.dump((my_func, (5,), {}), f)

        with patch.object(sys, "argv", ["slurp.worker", str(payload_path)]):
            main()

        assert marker_path.read_text() == "called:5"

    def test_result_mode_writes_ok(self, tmp_path: Path) -> None:
        """Worker with result_path should write ("ok", value)."""
        payload_path = tmp_path / "payload.pkl"
        result_path = str(tmp_path / "result.pkl")

        def my_func(x: int) -> int:
            return x * 3

        with open(payload_path, "wb") as f:
            cloudpickle.dump((my_func, (7,), {}), f)

        with patch.object(sys, "argv", ["slurp.worker", str(payload_path), result_path]):
            main()

        with open(result_path, "rb") as fh:
            status, value = cloudpickle.load(fh)
        assert status == "ok"
        assert value == 21

    def test_result_mode_writes_error_on_exception(self, tmp_path: Path) -> None:
        """Worker should write ("error", exc, tb) when the function raises."""
        payload_path = tmp_path / "payload.pkl"
        result_path = str(tmp_path / "error.pkl")

        def boom(x: int) -> None:
            raise ValueError(f"bad input: {x}")

        with open(payload_path, "wb") as f:
            cloudpickle.dump((boom, (99,), {}), f)

        with (
            patch.object(sys, "argv", ["slurp.worker", str(payload_path), result_path]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        with open(result_path, "rb") as fh:
            status, exc, tb_str = cloudpickle.load(fh)
        assert status == "error"
        assert isinstance(exc, ValueError)
        assert "bad input: 99" in str(exc)
        assert "Traceback" in tb_str

    def test_resolves_object_ref_args(self, tmp_path: Path) -> None:
        """Worker should resolve ObjectRef arguments before calling the function."""
        # Write a shared object
        obj_path = tmp_path / "shared.pkl"
        shared_data = [1, 2, 3]
        with open(obj_path, "wb") as f:
            cloudpickle.dump(shared_data, f)

        # Write payload with a fake ObjectRef as argument
        payload_path = tmp_path / "payload.pkl"
        result_path = str(tmp_path / "result.pkl")

        class FakeObjectRef:
            _slurp_object_path = str(obj_path)

        def sum_list(data: list[int]) -> int:
            return sum(data)

        with open(payload_path, "wb") as f:
            cloudpickle.dump((sum_list, (FakeObjectRef(),), {}), f)

        with patch.object(sys, "argv", ["slurp.worker", str(payload_path), result_path]):
            main()

        with open(result_path, "rb") as fh:
            status, value = cloudpickle.load(fh)
        assert status == "ok"
        assert value == 6

    def test_no_args_exits_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Worker should print usage and exit 1 if no payload path given."""
        with (
            patch.object(sys, "argv", ["slurp.worker"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Usage" in captured.err
