"""Unit tests for SLURM script generation."""

from __future__ import annotations

import pytest

from slurp.core.slurm import (
    _gpu_directive,
    build_torchrun_command,
    generate_sbatch_script,
    generate_array_wrapper,
    log_path,
)
from slurp.domain import Profile, ResourceRequest


class TestGpuDirective:
    def test_zero_gpus(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest(gpus=0)
        assert _gpu_directive(r, p) == ""

    def test_gres(self) -> None:
        p = Profile(name="test", hostname="host", gpu_flag_style="gres")
        r = ResourceRequest(gpus=4)
        assert _gpu_directive(r, p) == "#SBATCH --gres=gpu:4"

    def test_gpus(self) -> None:
        p = Profile(name="test", hostname="host", gpu_flag_style="gpus")
        r = ResourceRequest(gpus=2)
        assert _gpu_directive(r, p) == "#SBATCH --gpus=2"


class TestGenerateSbatchScript:
    def test_basic(self) -> None:
        p = Profile(name="test", hostname="host", partition="gpu", account="acct")
        r = ResourceRequest(gpus=4, time="1:00:00", partition="gpu", account="acct")
        script = generate_sbatch_script(
            resources=r,
            profile=p,
            command="python train.py",
            working_dir="/tmp/work",
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=python" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --gres=gpu:4" in script
        assert "#SBATCH --time=1:00:00" in script
        assert "#SBATCH --account=acct" in script
        assert "python train.py" in script

    def test_array(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest()
        script = generate_sbatch_script(
            resources=r,
            profile=p,
            command="python train.py",
            working_dir="/tmp/work",
            array_size=5,
        )
        assert "#SBATCH --array=0-4" in script
        assert "slurm-python-%A_%a.out" in script

    def test_snapshot(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest()
        script = generate_sbatch_script(
            resources=r,
            profile=p,
            command="python train.py",
            working_dir="/tmp/work",
            snapshot=True,
        )
        assert "cd /tmp/work/.slurp/runs/$SLURM_JOB_ID" in script

    def test_dependencies(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest()
        script = generate_sbatch_script(
            resources=r,
            profile=p,
            command="python train.py",
            working_dir="/tmp/work",
            depends_on=["123", "456"],
            depends_on_type="afterok",
        )
        assert "#SBATCH --dependency=afterok:123,456" in script

    def test_slurm_kwargs(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest(slurm_kwargs={"exclude": "node05"})
        script = generate_sbatch_script(
            resources=r,
            profile=p,
            command="python train.py",
            working_dir="/tmp/work",
        )
        assert "#SBATCH --exclude=node05" in script


class TestGenerateArrayWrapper:
    def test_array_wrapper(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest()
        configs = [{"seed": "1"}, {"seed": "2"}]
        script = generate_array_wrapper(
            resources=r,
            profile=p,
            template="python train.py --seed {seed}",
            configs=configs,
            working_dir="/tmp/work",
        )
        assert "#SBATCH --array=0-1" in script
        assert "SEED=(\"1\" \"2\")" in script
        assert 'python train.py --seed "${SEED[$SLURM_ARRAY_TASK_ID]}"' in script

    def test_empty_configs(self) -> None:
        p = Profile(name="test", hostname="host")
        r = ResourceRequest()
        with pytest.raises(ValueError):
            generate_array_wrapper(
                resources=r,
                profile=p,
                template="python train.py",
                configs=[],
                working_dir="/tmp/work",
            )


class TestBuildTorchrunCommand:
    def test_single_node(self) -> None:
        p = Profile(name="test", hostname="host")
        cmd = build_torchrun_command("python train.py", p, nodes=1, gpus_per_node=4)
        assert cmd == "python train.py"

    def test_multi_node(self) -> None:
        p = Profile(name="test", hostname="host", mpi_mode="pmi2", cpu_bind="cores")
        cmd = build_torchrun_command("python train.py", p, nodes=2, gpus_per_node=4)
        assert "srun --mpi=pmi2 --cpu-bind=cores" in cmd
        assert "torchrun" in cmd
        assert "NCCL_DEBUG=INFO" in cmd


class TestLogPath:
    def test_single_job(self) -> None:
        out, err = log_path("123", "train", "/tmp")
        assert out == "/tmp/slurm-train-123.out"
        assert err == "/tmp/slurm-train-123.err"

    def test_array_task(self) -> None:
        out, err = log_path("123", "train", "/tmp", task_id=3)
        assert out == "/tmp/slurm-train-123_3.out"
        assert err == "/tmp/slurm-train-123_3.err"
