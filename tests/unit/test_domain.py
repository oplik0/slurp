"""Unit tests for domain models."""

from __future__ import annotations

import pytest
from slurp.domain import (
    JobStatus,
    Profile,
    ResourceRequest,
    SlurmJobInfo,
    slugify_command,
)


class TestJobStatus:
    def test_is_terminal(self) -> None:
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal
        assert JobStatus.TIMEOUT.is_terminal
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.RUNNING.is_terminal
        assert not JobStatus.UNKNOWN.is_terminal


class TestResourceRequest:
    def test_defaults(self) -> None:
        r = ResourceRequest()
        assert r.gpus == 0
        assert r.nodes == 1
        assert r.time == "2:00:00"
        assert r.cpus == 8

    def test_time_validation(self) -> None:
        ResourceRequest(time="1:00:00")
        ResourceRequest(time="2:00:00")
        ResourceRequest(time="24:00:00")
        with pytest.raises(ValueError):
            ResourceRequest(time="invalid")

    def test_negative_gpus(self) -> None:
        with pytest.raises(ValueError):
            ResourceRequest(gpus=-1)

    def test_zero_nodes(self) -> None:
        with pytest.raises(ValueError):
            ResourceRequest(nodes=0)

    def test_job_name_slugify(self) -> None:
        r = ResourceRequest(job_name="my job!")
        assert r.job_name == "my-job"


class TestProfile:
    def test_format_prologue(self) -> None:
        p = Profile(
            name="test",
            hostname="host",
            account="acct123",
            prologue="jutil activate {account}",
        )
        assert p.format_prologue() == "jutil activate acct123"


class TestSlurmJobInfo:
    def test_status_mapping(self) -> None:
        info = SlurmJobInfo(job_id="1", state="RUNNING", exit_code="0:0")
        assert info.status == JobStatus.RUNNING

    def test_exit_code_parsing(self) -> None:
        info = SlurmJobInfo(job_id="1", state="COMPLETED", exit_code="1:0")
        assert info.exit_code_int == 1


class TestSlugifyCommand:
    def test_simple(self) -> None:
        assert slugify_command("python train.py") == "python"
        assert slugify_command("./train.py") == "train-py"
        assert slugify_command("") == "job"
