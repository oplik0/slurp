"""Tests for slurp.guard — recursive distribution guard."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from slurp import guard


class TestIsInsideSlurmJob:
    def test_false_when_no_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_JOB_ID", None)
            assert guard.is_inside_slurm_job() is False

    def test_true_when_slurm_job_id_set(self) -> None:
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            assert guard.is_inside_slurm_job() is True


class TestCheck:
    def test_passes_outside_slurm(self) -> None:
        """check() should not raise when SLURM_JOB_ID is unset."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_JOB_ID", None)
            guard.check()  # should not raise

    def test_raises_inside_slurm(self) -> None:
        """check() should raise RuntimeError when SLURM_JOB_ID is set."""
        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "999"}),
            pytest.raises(RuntimeError, match="Recursive distribution"),
        ):
            guard.check()

    def test_raises_with_job_id_in_message(self) -> None:
        with (
            patch.dict(os.environ, {"SLURM_JOB_ID": "42"}),
            pytest.raises(RuntimeError, match="42"),
        ):
            guard.check()


class TestDisabled:
    def test_allows_distribution_inside_slurm(self) -> None:
        """disabled() context manager should suppress the guard."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}), guard.disabled():
            # Should not raise despite SLURM_JOB_ID being set
            guard.check()

    def test_restores_after_block(self) -> None:
        """Guard should be re-enabled after the context exits."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}):
            with guard.disabled():
                guard.check()  # OK inside
            # Should raise again outside
            with pytest.raises(RuntimeError):
                guard.check()

    def test_restores_on_exception(self) -> None:
        """Guard should be restored even if an exception occurs."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "777"}):
            with pytest.raises(ValueError), guard.disabled():
                raise ValueError("boom")
            # Guard should be back on
            with pytest.raises(RuntimeError):
                guard.check()

    def test_disable_alias(self) -> None:
        """disable should be the same as disabled."""
        assert guard.disable is guard.disabled
