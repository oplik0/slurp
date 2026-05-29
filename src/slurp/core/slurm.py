"""SLURM wrappers: SBATCH generation, sbatch/squeue/sacct/scancel."""

from __future__ import annotations

import re
from typing import Any

from slurp.domain import (
    JobStatus,
    Profile,
    ResourceRequest,
    SlurmJobInfo,
    slugify_command,
)
from slurp.errors import SlurmError


def _gpu_directive(resources: ResourceRequest, profile: Profile) -> str:
    if resources.gpus <= 0:
        return ""
    if profile.gpu_flag_style == "gpus":
        return f"#SBATCH --gpus={resources.gpus}"
    return f"#SBATCH --gres=gpu:{resources.gpus}"


def generate_sbatch_script(
    resources: ResourceRequest,
    profile: Profile,
    command: str,
    *,
    working_dir: str,
    job_name: str | None = None,
    array_size: int | None = None,
    throttle: int = 20,
    snapshot: bool = False,
    depends_on: list[str] | None = None,
    depends_on_type: str = "afterok",
    slurm_kwargs: dict[str, str] | None = None,
) -> str:
    """Generate a complete SBATCH script string."""
    name = resources.job_name or job_name or slugify_command(command)
    directives: list[str] = ["#!/bin/bash", f"#SBATCH --job-name={name}"]

    if resources.partition:
        directives.append(f"#SBATCH --partition={resources.partition}")
    if resources.nodes > 1:
        directives.append(f"#SBATCH --nodes={resources.nodes}")
    gpu_line = _gpu_directive(resources, profile)
    if gpu_line:
        directives.append(gpu_line)
    directives.append(f"#SBATCH --time={resources.time}")
    if resources.mem:
        directives.append(f"#SBATCH --mem={resources.mem}")
    directives.append(f"#SBATCH --cpus-per-task={resources.cpus}")
    if resources.constraint:
        directives.append(f"#SBATCH --constraint={resources.constraint}")
    if resources.qos:
        directives.append(f"#SBATCH --qos={resources.qos}")
    if resources.account:
        directives.append(f"#SBATCH --account={resources.account}")
    if resources.mail_type:
        directives.append(f"#SBATCH --mail-type={resources.mail_type}")

    # Log paths
    if array_size is not None:
        out_path = f"{working_dir}/slurm-{name}-%A_%a.out"
        err_path = f"{working_dir}/slurm-{name}-%A_%a.err"
    else:
        out_path = f"{working_dir}/slurm-{name}-%j.out"
        err_path = f"{working_dir}/slurm-{name}-%j.err"
    directives.append(f"#SBATCH --output={out_path}")
    directives.append(f"#SBATCH --error={err_path}")

    if array_size is not None:
        throttle_str = f"%{throttle}" if throttle > 0 else ""
        directives.append(f"#SBATCH --array=0-{array_size - 1}{throttle_str}")

    if depends_on:
        dep_str = f"#SBATCH --dependency={depends_on_type}:{','.join(depends_on)}"
        directives.append(dep_str)

    # Merge slurm_kwargs last (they override)
    extra = slurm_kwargs or {}
    for k, v in extra.items():
        if v == "":
            directives.append(f"#SBATCH --{k}")
        else:
            directives.append(f"#SBATCH --{k}={v}")

    # Prologue
    prologue = profile.format_prologue().strip()

    # Working directory
    cd_line = f"cd {working_dir}"
    if snapshot:
        cd_line = f"cd {working_dir}/.slurp/runs/$SLURM_JOB_ID"

    # Assemble
    lines = directives
    if prologue:
        lines.append("")
        lines.append("# Profile prologue")
        lines.extend(prologue.splitlines())
    lines.append("")
    lines.append(cd_line)
    lines.append("")
    lines.append("# User command")
    lines.append(command)
    lines.append("")
    return "\n".join(lines) + "\n"


def generate_array_wrapper(
    resources: ResourceRequest,
    profile: Profile,
    template: str,
    configs: list[dict[str, str]],
    *,
    working_dir: str,
    job_name: str | None = None,
    **kwargs: Any,
) -> str:
    """Generate an SBATCH script for a job array with parameter mapping."""
    if not configs:
        raise ValueError("configs must not be empty")

    # Collect unique keys and build shell arrays
    keys = sorted({k for cfg in configs for k in cfg})
    arrays: list[str] = []
    for key in keys:
        values = [cfg.get(key, "") for cfg in configs]
        quoted = " ".join(f'"{v}"' for v in values)
        arrays.append(f"{key.upper()}=({quoted})")

    # Mapping lines
    mappings = []
    for key in keys:
        mappings.append(f'{key.upper()}="${{{key.upper()}[$SLURM_ARRAY_TASK_ID]}}"')

    # Substitute template
    command = template
    for key in keys:
        command = command.replace(f"{{{key}}}", f'"${{{key.upper()}}}"')

    base_script = generate_sbatch_script(
        resources=resources,
        profile=profile,
        command=command,
        working_dir=working_dir,
        job_name=job_name,
        array_size=len(configs),
        **kwargs,
    )

    # Inject array declarations after the shebang/directives, before prologue
    lines = base_script.splitlines()
    directive_end = 0
    for i, line in enumerate(lines):
        if not line.startswith("#SBATCH") and not line.startswith("#!/"):
            directive_end = i
            break

    array_section = ["# Array parameter mapping"] + arrays + [""] + mappings
    lines = lines[:directive_end] + [""] + array_section + lines[directive_end:]
    return "\n".join(lines) + "\n"


async def sbatch_submit(
    profile: Profile,
    script: str,
    *,
    working_dir: str,
    ssh_manager: Any,
) -> str:
    """Submit a script via sbatch and return the job ID."""
    from slurp.core.ssh import SSHManager

    if not isinstance(ssh_manager, SSHManager):
        ssh_manager = SSHManager()

    # Write script to remote temp file
    import random
    remote_script = f"{working_dir}/.slurm_script_{random.randint(0, 999999)}.sh"
    heredoc_cmd = (
        f"mkdir -p {working_dir} && cat > '{remote_script}' << 'EOF'\n"
        f"{script}\n"
        "EOF"
    )
    await ssh_manager.run(
        profile,
        heredoc_cmd,
        timeout=15.0,
    )

    # Run sbatch
    exit_code, stdout, stderr = await ssh_manager.run(
        profile,
        f"cd {working_dir} && sbatch '{remote_script}'",
        timeout=30.0,
    )

    # Clean up temp script
    await ssh_manager.run(profile, f"rm -f '{remote_script}'", timeout=5.0, check=False)

    if exit_code != 0:
        raise SlurmError(
            f"sbatch failed: {stderr.strip()}",
            stderr_fragment=stderr[-2048:] if stderr else None,
            hint="Check SLURM account, partition, and resource limits.",
        )

    match = re.search(r"Submitted batch job (\d+)", stdout)
    if not match:
        raise SlurmError(
            f"Could not parse job ID from sbatch output: {stdout.strip()}",
            hint="Unexpected sbatch output format.",
        )
    return match.group(1)


async def sacct_query(
    profile: Profile,
    job_ids: list[str],
    *,
    ssh_manager: Any,
) -> dict[str, SlurmJobInfo]:
    """Query sacct for job states and return a mapping."""
    from slurp.core.ssh import SSHManager

    if not isinstance(ssh_manager, SSHManager):
        ssh_manager = SSHManager()

    if not job_ids:
        return {}

    id_str = ",".join(job_ids)
    cmd = (
        f"sacct -j {id_str} "
        "--format=JobID,State,ExitCode,MaxRSS,Elapsed "
        "--noheader --parsable2"
    )
    try:
        _, stdout, stderr = await ssh_manager.run(profile, cmd, timeout=30.0)
    except Exception:
        return {}

    results: dict[str, SlurmJobInfo] = {}
    for line in stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id = parts[0].strip()
        state = parts[1].strip()
        exit_code = parts[2].strip()
        max_rss = parts[3].strip() if len(parts) > 3 else None
        elapsed = parts[4].strip() if len(parts) > 4 else None
        results[job_id] = SlurmJobInfo(
            job_id=job_id,
            state=state,
            exit_code=exit_code,
            max_rss=max_rss,
            elapsed=elapsed,
        )
    return results


async def squeue_query(
    profile: Profile,
    *,
    ssh_manager: Any,
    user: str | None = None,
) -> dict[str, JobStatus]:
    """Query squeue for live job states."""
    from slurp.core.ssh import SSHManager

    if not isinstance(ssh_manager, SSHManager):
        ssh_manager = SSHManager()

    user_filter = f" -u {user}" if user else ""
    cmd = f"squeue{user_filter} --format='%i %T' --noheader"
    try:
        _, stdout, _ = await ssh_manager.run(profile, cmd, timeout=15.0)
    except Exception:
        return {}

    results: dict[str, JobStatus] = {}
    for line in stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            jid = parts[0]
            state = parts[1].upper()
            mapping = {
                "PENDING": JobStatus.PENDING,
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELLED": JobStatus.CANCELLED,
                "TIMEOUT": JobStatus.TIMEOUT,
            }
            results[jid] = mapping.get(state, JobStatus.UNKNOWN)
    return results


async def scancel(
    profile: Profile,
    job_id: str,
    *,
    ssh_manager: Any,
) -> None:
    """Cancel a job."""
    from slurp.core.ssh import SSHManager

    if not isinstance(ssh_manager, SSHManager):
        ssh_manager = SSHManager()

    try:
        exit_code, _, stderr = await ssh_manager.run(
            profile, f"scancel {job_id}", timeout=15.0, check=False
        )
        if exit_code != 0 and "Invalid job id" not in stderr:
            raise SlurmError(
                f"scancel failed: {stderr.strip()}",
                hint="Job may already be completed or you lack permission.",
            )
    except Exception as exc:
        if isinstance(exc, SlurmError):
            raise
        raise SlurmError(
            f"scancel failed: {exc}",
            hint="Check SSH connection and job ID.",
        )


def log_path(job_id: str, job_name: str, working_dir: str, task_id: int | None = None) -> tuple[str, str]:
    """Return (out_path, err_path) for a job."""
    suffix = f"_{task_id}" if task_id is not None else ""
    jid = f"{job_id}{suffix}"
    out = f"{working_dir}/slurm-{job_name}-{jid}.out"
    err = f"{working_dir}/slurm-{job_name}-{jid}.err"
    return out, err


def build_torchrun_command(
    user_command: str,
    profile: Profile,
    *,
    nodes: int,
    gpus_per_node: int,
) -> str:
    """Generate multi-node PyTorch launch command."""
    if nodes <= 1:
        return user_command

    mpi = profile.mpi_mode or "pmi2"
    cpu_bind = profile.cpu_bind or "cores"

    env_lines = [
        'export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)',
        "export MASTER_PORT=29500",
        "export OMP_NUM_THREADS=1",
    ]
    if nodes > 1:
        env_lines.append("export NCCL_DEBUG=INFO")
        env_lines.append(
            "export NCCL_DEBUG_FILE=$PROJECT/.slurp/nccl_logs/$SLURM_JOB_ID-%h.log"
        )

    srun_line = (
        f"srun --mpi={mpi} --cpu-bind={cpu_bind} --distribution=block:cyclic "
        f"torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc-per-node=$SLURM_GPUS_PER_NODE "
        f"--rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT "
        f"--rdzv-id=$SLURM_JOB_ID {user_command}"
    )

    return "\n".join(env_lines) + "\n" + srun_line + "\n"
