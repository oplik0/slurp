"""Tests for slurp.core.slurm SBATCH generation and SLURM utilities."""

import pytest

from slurp.core.slurm import (
    _gpu_directive,
    build_torchrun_command,
    generate_array_wrapper,
    generate_sbatch_script,
    log_path,
)
from slurp.domain import Profile, ResourceRequest


class TestGpuDirective:
    """Tests for _gpu_directive."""

    def test_no_gpus(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest(gpus=0)
        assert _gpu_directive(resources, profile) == ""

    def test_gres_style(self) -> None:
        profile = Profile(name="test", hostname="hpc", gpu_flag_style="gres")
        resources = ResourceRequest(gpus=4)
        assert _gpu_directive(resources, profile) == "#SBATCH --gres=gpu:4"

    def test_gpus_style(self) -> None:
        profile = Profile(name="test", hostname="hpc", gpu_flag_style="gpus")
        resources = ResourceRequest(gpus=2)
        assert _gpu_directive(resources, profile) == "#SBATCH --gpus=2"


class TestGenerateSbatchScript:
    """Tests for generate_sbatch_script."""

    def test_basic_script(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest(gpus=2, time="1:00:00")
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=python" in script
        assert "#SBATCH --time=1:00:00" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --gres=gpu:2" in script
        assert "cd /remote" in script
        assert "python train.py" in script

    def test_with_partition(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest(partition="gpu", account="lab")
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
        )
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --account=lab" in script

    def test_with_job_name(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
            job_name="my-job",
        )
        assert "#SBATCH --job-name=my-job" in script

    def test_array_job(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
            array_size=5,
            throttle=10,
        )
        assert "#SBATCH --array=0-4%10" in script
        assert "%A_%a.out" in script

    def test_array_no_throttle(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
            array_size=3,
            throttle=0,
        )
        assert "#SBATCH --array=0-2" in script

    def test_snapshot(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
            snapshot=True,
        )
        assert "cd /remote/.slurp/runs/$SLURM_JOB_ID" in script

    def test_dependency(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
            depends_on=["123", "456"],
            depends_on_type="afterok",
        )
        assert "#SBATCH --dependency=afterok:123,456" in script

    def test_slurm_kwargs(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
            slurm_kwargs={"nodelist": "node01", "verbose": ""},
        )
        assert "#SBATCH --nodelist=node01" in script
        assert "#SBATCH --verbose" in script

    def test_prologue(self) -> None:
        profile = Profile(name="test", hostname="hpc", prologue="module load cuda")
        resources = ResourceRequest()
        script = generate_sbatch_script(
            resources=resources,
            profile=profile,
            command="python train.py",
            working_dir="/remote",
        )
        assert "# Profile prologue" in script
        assert "module load cuda" in script


class TestGenerateArrayWrapper:
    """Tests for generate_array_wrapper."""

    def test_basic_array(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        configs = [
            {"lr": "0.01", "seed": "1"},
            {"lr": "0.02", "seed": "2"},
        ]
        script = generate_array_wrapper(
            resources=resources,
            profile=profile,
            template="python train.py --lr {lr} --seed {seed}",
            configs=configs,
            working_dir="/remote",
        )
        assert "# Array parameter mapping" in script
        assert 'LR=("0.01" "0.02")' in script
        assert 'SEED=("1" "2")' in script
        assert 'LR="${LR[$SLURM_ARRAY_TASK_ID]}"' in script
        assert 'SEED="${SEED[$SLURM_ARRAY_TASK_ID]}"' in script
        assert "#SBATCH --array=0-1" in script

    def test_empty_configs_raises(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        resources = ResourceRequest()
        with pytest.raises(ValueError, match="configs must not be empty"):
            generate_array_wrapper(
                resources=resources,
                profile=profile,
                template="python train.py",
                configs=[],
                working_dir="/remote",
            )


class TestLogPath:
    """Tests for log_path."""

    def test_basic(self) -> None:
        out, err = log_path("123", "job", "/remote")
        assert out == "/remote/slurm-job-123.out"
        assert err == "/remote/slurm-job-123.err"

    def test_with_task_id(self) -> None:
        out, err = log_path("123", "job", "/remote", task_id=0)
        assert out == "/remote/slurm-job-123_0.out"
        assert err == "/remote/slurm-job-123_0.err"


class TestBuildTorchrunCommand:
    """Tests for build_torchrun_command."""

    def test_single_node(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        cmd = build_torchrun_command("python train.py", profile, nodes=1, gpus_per_node=4)
        assert cmd == "python train.py"

    def test_multi_node(self) -> None:
        profile = Profile(name="test", hostname="hpc")
        cmd = build_torchrun_command("python train.py", profile, nodes=2, gpus_per_node=4)
        assert "MASTER_ADDR" in cmd
        assert "MASTER_PORT=29500" in cmd
        assert "NCCL_DEBUG=INFO" in cmd
        assert "torchrun" in cmd
        assert "--nnodes=$SLURM_JOB_NUM_NODES" in cmd

    def test_custom_mpi(self) -> None:
        profile = Profile(name="test", hostname="hpc", mpi_mode="pmi2")
        cmd = build_torchrun_command("python train.py", profile, nodes=2, gpus_per_node=4)
        assert "--mpi=pmi2" in cmd
