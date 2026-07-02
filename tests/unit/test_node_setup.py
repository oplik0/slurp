"""Tests for slurp.node_setup — per-node initialization registry."""

from __future__ import annotations

from slurp.node_setup import (
    _setup_functions,
    get_setup_functions,
    node_setup,
    run_setup_functions,
)


class TestNodeSetup:
    def setup_method(self) -> None:
        """Clear registry before each test."""
        _setup_functions.clear()

    def test_registers_function(self) -> None:
        @node_setup
        def my_setup() -> None:
            pass

        funcs = get_setup_functions()
        assert "my_setup" in funcs
        assert funcs["my_setup"] is my_setup

    def test_multiple_registrations(self) -> None:
        @node_setup
        def setup_a() -> None:
            pass

        @node_setup
        def setup_b() -> None:
            pass

        funcs = get_setup_functions()
        assert len(funcs) == 2
        assert "setup_a" in funcs
        assert "setup_b" in funcs

    def test_run_setup_functions_calls_all(self) -> None:
        call_log: list[str] = []

        @node_setup
        def step1() -> None:
            call_log.append("step1")

        @node_setup
        def step2() -> None:
            call_log.append("step2")

        run_setup_functions()
        assert call_log == ["step1", "step2"]

    def test_run_setup_with_no_functions(self) -> None:
        """Should not raise when no setup functions are registered."""
        _setup_functions.clear()
        run_setup_functions()  # should be a no-op

    def test_get_setup_functions_returns_copy(self) -> None:
        """get_setup_functions should return a copy, not the internal dict."""

        @node_setup
        def s() -> None:
            pass

        funcs = get_setup_functions()
        funcs.clear()
        # Internal registry should be unaffected
        assert "s" in _setup_functions

    def test_decorated_function_still_callable(self) -> None:
        """The decorator should return the original function (callable)."""
        called: list[bool] = []

        @node_setup
        def setup() -> None:
            called.append(True)

        # Direct call should work
        setup()
        assert called == [True]
