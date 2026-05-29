"""Tests for slurp.domain models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from slurp.domain import (
    Job,
    JobRecord,
    JobResult,
    JobStatus,
    LogOffsetRecord,
    Profile,
    ResourceRequest,
    SlurmJobInfo,
    slugify_command,
)


class TestJobStatus:
    """Tests for the JobStatus enum."""

    def test_terminal_states(self) -> None:
        assert JobStatus.COMPLETED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal
        assert JobStatus.TIMEOUT.is_terminal

    def test_non_terminal_states(self) -> None:
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.RUNNING.is_terminal
        assert not JobStatus.UNKNOWN.is_terminal

    def test_str_enum(self) -> None:
        assert str(JobStatus.RUNNING) == "RUNNING"
        assert JobStatus.PENDING == "PENDING"


class TestResourceRequest:
    """Tests for ResourceRequest validation."""

    def test_defaults(self) -> None:
        r = ResourceRequest()
        assert r.gpus == 0
        assert r.nodes == 1
        assert r.time == "2:00:00"
        assert r.cpus == 8

    def test_valid_time_formats(self) -> None:
        assert ResourceRequest(time="3600").time == "3600"
        assert ResourceRequest(time="10:00").time == "10:00"
        assert ResourceRequest(time="2:00:00").time == "2:00:00"
        assert ResourceRequest(time="1-12:00:00").time == "1-12:00:00"

    def test_invalid_time_format(self) -> None:
        with pytest.raises(ValidationError, match="Invalid SLURM time format"):
            ResourceRequest(time="not-a-time")

    def test_job_name_sanitization(self) -> None:
        assert ResourceRequest(job_name="train_123").job_name == "train_123"
        assert ResourceRequest(job_name="train model.py").job_name == "train-model-py"
        assert ResourceRequest(job_name="--bad chars--").job_name == "bad-chars"

    def test_job_name_none(self) -> None:
        assert ResourceRequest(job_name=None).job_name is None

    def test_non_negative_gpus(self) -> None:
        with pytest.raises(ValidationError, match="gpus must be non-negative"):
            ResourceRequest(gpus=-1)

    def test_positive_nodes(self) -> None:
        with pytest.raises(ValidationError, match="nodes must be positive"):
            ResourceRequest(nodes=0)

    def test_positive_cpus(self) -> None:
        with pytest.raises(ValidationError, match="cpus must be positive"):
            ResourceRequest(cpus=0)

    def test_slurm_kwargs(self) -> None:
        r = ResourceRequest(slurm_kwargs={"nodelist": "node01"})
        assert r.slurm_kwargs == {"nodelist": "node01"}


class TestProfile:
    """Tests for Profile model."""

    def test_basic_profile(self) -> None:
        p = Profile(name="test", hostname="hpc.local")
        assert p.name == "test"
        assert p.hostname == "hpc.local"
        assert p.mpi_mode == "pmi2"
        assert p.cpu_bind == "cores"
        assert p.gpu_flag_style == "gres"

    def test_sync_config(self) -> None:
        p = Profile(
            name="test",
            hostname="hpc.local",
            sync=Profile.SyncConfig(local="./src", remote="/remote/src"),
        )
        assert p.sync is not None
        assert p.sync.local == "./src"
        assert p.sync.remote == "/remote/src"

    def test_format_prologue(self) -> None:
        p = Profile(name="test", hostname="hpc.local", prologue="echo {account}")
        assert p.format_prologue() == "echo "

        p2 = Profile(
            name="test",
            hostname="hpc.local",
            account="acme",
            prologue="echo {account}",
        )
        assert p2.format_prologue() == "echo acme"


class TestJob:
    """Tests for Job model."""

    def test_job_creation(self) -> None:
        job = Job(
            job_id="12345",
            name="test",
            status=JobStatus.PENDING,
            profile="test",
        )
        assert job.job_id == "12345"
        assert job.status == JobStatus.PENDING


class TestJobRecord:
    """Tests for JobRecord model."""

    def test_job_record(self) -> None:
        record = JobRecord(
            job_id="12345",
            name="test",
            status=JobStatus.PENDING,
            profile="test",
            submitted_at=datetime.now(UTC),
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/tmp",
        )
        assert record.idempotency_hash is None


class TestJobResult:
    """Tests for JobResult model."""

    def test_defaults(self) -> None:
        result = JobResult(job_id="1", status=JobStatus.COMPLETED)
        assert result.exit_code is None
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.wall_time == 0.0


class TestLogOffsetRecord:
    """Tests for LogOffsetRecord."""

    def test_defaults(self) -> None:
        r = LogOffsetRecord()
        assert r.out == 0
        assert r.err == 0
        assert r.last_read is None


class TestSlurmJobInfo:
    """Tests for SlurmJobInfo parsing."""

    def test_status_mapping(self) -> None:
        info = SlurmJobInfo(job_id="1", state="RUNNING", exit_code="0:0")
        assert info.status == JobStatus.RUNNING

    def test_cancelled_plus(self) -> None:
        info = SlurmJobInfo(job_id="1", state="CANCELLED+", exit_code="0:0")
        assert info.status == JobStatus.CANCELLED

    def test_timeout_plus(self) -> None:
        info = SlurmJobInfo(job_id="1", state="TIMEOUT+", exit_code="0:0")
        assert info.status == JobStatus.TIMEOUT

    def test_node_fail(self) -> None:
        info = SlurmJobInfo(job_id="1", state="NODE_FAIL", exit_code="1:0")
        assert info.status == JobStatus.FAILED

    def test_out_of_memory(self) -> None:
        info = SlurmJobInfo(job_id="1", state="OUT_OF_MEMORY", exit_code="1:0")
        assert info.status == JobStatus.FAILED

    def test_unknown_state(self) -> None:
        info = SlurmJobInfo(job_id="1", state="SUSPENDED", exit_code="0:0")
        assert info.status == JobStatus.UNKNOWN

    def test_exit_code_int(self) -> None:
        info = SlurmJobInfo(job_id="1", state="COMPLETED", exit_code="0:0")
        assert info.exit_code_int == 0

        info2 = SlurmJobInfo(job_id="1", state="FAILED", exit_code="1:0")
        assert info2.exit_code_int == 1

    def test_exit_code_int_empty(self) -> None:
        info = SlurmJobInfo(job_id="1", state="FAILED", exit_code="")
        assert info.exit_code_int is None

    def test_exit_code_int_malformed(self) -> None:
        info = SlurmJobInfo(job_id="1", state="FAILED", exit_code="abc")
        assert info.exit_code_int is None


class TestSlugifyCommand:
    """Tests for slugify_command."""

    def test_basic(self) -> None:
        assert slugify_command("python train.py --lr 0.01") == "python"

    def test_path(self) -> None:
        assert slugify_command("/usr/bin/python script.py") == "python"

    def test_empty(self) -> None:
        assert slugify_command("") == "job"

    def test_special_chars(self) -> None:
        assert slugify_command("my_script@v2.py") == "my_script-v2-py"

    def test_strip_hyphens(self) -> None:
        assert slugify_command("--test--") == "test"
