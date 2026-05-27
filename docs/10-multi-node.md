# 10 — Multi-Node and GPU Specification

This document specifies how slurp handles distributed multi-node PyTorch jobs, GPU resource requests, launcher generation, cluster-specific defaults, and failure modes unique to multi-node execution.

---

## 1. TorchrunLauncher Auto-Generation

For jobs requesting more than one node, slurp automatically generates a wrapper command that sets up PyTorch Distributed environment variables and invokes `torchrun` under `srun`.

### Generated command (multi-node)

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1

# Environment setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

# Multi-node launch via srun + torchrun
srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic \
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-id=$SLURM_JOB_ID \
    train.py --lr 0.01
```

### Variable resolution

| Variable | Source | Example value |
|----------|--------|---------------|
| `MASTER_ADDR` | First host in `scontrol show hostnames $SLURM_JOB_NODELIST` | `jrc0771` |
| `MASTER_PORT` | Hard-coded default; user can override via `slurm_kwargs` | `29500` |
| `OMP_NUM_THREADS` | Hard-coded to 1 to prevent CPU/GPU contention | `1` |
| `SLURM_JOB_NUM_NODES` | SLURM runtime variable | `2` |
| `SLURM_GPUS_PER_NODE` | SLURM runtime variable (from `--gres=gpu:N`) | `4` |
| `SLURM_JOB_ID` | SLURM runtime variable | `12345` |

### `srun` vs direct execution

| Node count | Execution model | Rationale |
|------------|-----------------|-----------|
| 1 | Direct user command (no `srun`, no `torchrun`) | Single-node jobs do not need distributed setup. Avoids `srun` overhead. |
| > 1 | `srun --mpi=pmi2 --cpu-bind=cores torchrun ...` | `srun` is required to spawn processes across nodes. `torchrun` manages intra-node GPU mapping. |

The `build_torchrun_command()` function in `core/launcher.py` returns the user command unchanged for `nodes == 1`, and the full `srun` wrapper for `nodes > 1`.

---

## 2. Auto-Behaviors for Multi-Node

### `NCCL_DEBUG`

For all multi-node jobs (`nodes > 1`), slurp automatically injects:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$PROJECT/.slurp/nccl_logs/$SLURM_JOB_ID-%h.log
```

- `NCCL_DEBUG=INFO` enables verbose NCCL logging, essential for diagnosing interconnect and ring-topology issues.
- `NCCL_DEBUG_FILE` redirects the voluminous logs to a dedicated directory rather than the main job stdout, keeping `slurm-<name>-<job_id>.out` readable.

**Override:** The user can suppress this by setting `NCCL_DEBUG=WARN` in their command or via `slurm_kwargs`, which takes precedence because slurp's exports are placed before the profile prologue.

### `NCCL_IB_HCA`

On InfiniBand clusters (e.g., JURECA), the correct IB device must be selected. slurp does **not** hard-code device names. Instead, the JURECA profile template sets:

```toml
[profiles.jureca]
prologue = """
export NCCL_IB_HCA=mlx5_0,mlx5_1
# ... rest of prologue ...
"""
```

This keeps device selection in the profile, not in `core/launcher.py`, preserving cluster-agnostic core code.

---

## 3. Profile Fields for MPI

The profile TOML supports two fields that influence multi-node launcher generation:

| Field | Default | Purpose |
|-------|---------|---------|
| `mpi_mode` | `pmi2` | Passed to `srun --mpi=<value>` |
| `cpu_bind` | `cores` | Passed to `srun --cpu-bind=<value>` |

**JURECA profile example:**

```toml
[profiles.jureca]
mpi_mode = "pmi2"
cpu_bind = "cores"
```

**Fallback behavior:** If `nodes > 1` but the profile does not define `mpi_mode`, slurp falls back to `pmi2` and logs a warning:

```
⚠ Profile 'jureca' missing mpi_mode; defaulting to pmi2.
```

Other clusters may use `none`, `pmix`, or `pmi2`. slurp does not validate the value; it passes it directly to `srun` and surfaces SLURM's error if the mode is unsupported.

---

## 4. GPU Monitoring (v0.2)

Live GPU utilization is deferred to v0.2. The planned implementation is:

```bash
# On the compute node, inside the job
nvidia-smi dmon -s pucm -d 1000 > gpu_stats.jsonl
```

- `dmon` (device monitoring) prints one line per GPU per interval.
- `-s pucm` selects power, utilization, clocks, and memory metrics.
- `-d 1000` samples every 1000 ms.

Parsing produces:

```python
@dataclass
class GPUStat:
    timestamp: datetime
    gpu_id: int
    power_w: float
    utilization_pct: float
    mem_used_mb: int
    mem_total_mb: int
```

The `--monitor-gpus` flag will be accepted in v0.1 but will print:

```
⚠ --monitor-gpus is not yet implemented. It will be available in slurp v0.2.
```

---

## 5. JURECA-Specific Multi-Node Defaults

All JURECA-specific behavior lives in the **profile template**, not in core code.

**JURECA multi-node profile template:**

```toml
[profiles.jureca]
hostname = "jrlogin"
partition = "dc-gpu"
account = "training2615"

prologue = """
jutil env activate -p {account}
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate
export NCCL_IB_HCA=mlx5_0,mlx5_1
"""

mpi_mode = "pmi2"
cpu_bind = "cores"
```

**What this provides:**
- `jutil env activate` sets up the Jülich project environment.
- `module load` sequence brings in CUDA and PyTorch.
- `NCCL_IB_HCA` selects the correct InfiniBand devices.
- `mpi_mode` and `cpu_bind` configure `srun` for JURECA's Slurm + ParaStation MPI setup.

**Core invariant:** `core/launcher.py` does not contain the string `jureca`, `fz-juelich.de`, or `jutil`. It reads `mpi_mode`, `cpu_bind`, and `prologue` from the profile and generates commands generically.

---

## 6. Error Handling for Multi-Node Failures

Multi-node jobs fail in ways that single-node jobs do not. slurp handles three common failure categories:

### 6.1 NCCL Errors

NCCL failures manifest in the `.err` log as:

```
RuntimeError: NCCL operation failed: unhandled system error
```

or

```
NCCL WARN NET/IB : Got completion with error 12, opcode 0, len 0, vendor err 129
```

**slurp behavior:**
- These errors are text in the `.err` log. `slurp logs` and `slurp watch` surface them normally.
- `slurm.py` does not attempt to parse NCCL error strings. The user inspects the log.
- If `--snapshot` was used, the `.err` log is preserved in `~/.slurp/runs/<job_id>/` for post-mortem analysis.

### 6.2 Node Failure

If a compute node dies during a multi-node job, SLURM marks the job as `FAILED` or `NODE_FAIL`. The `sacct` exit code is non-zero, but the specific failing node may not be obvious.

**slurp behavior:**
- `job.wait()` returns a `JobResult` with `status=FAILED` and `exit_code` from `sacct`.
- If `NCCL_DEBUG_FILE` was set, the per-node NCCL logs may contain the failing hostname.
- slurp does not automatically retry node-failed jobs. The user must resubmit.

### 6.3 `srun` / `torchrun` Mismatch

If `torchrun` arguments do not match the SLURM allocation (e.g., `--nproc-per-node` > GPUs per node, or `--nnodes` > `--nodes`), `torchrun` exits immediately with a clear error:

```
ValueError: The number of GPUs per node must be <= the number of GPUs allocated
```

**slurp behavior:**
- slurp populates `--nnodes` and `--nproc-per-node` from SLURM runtime variables, so this mismatch should not occur unless the user overrides `torchrun` arguments via `slurm_kwargs`.
- If it does occur, the error appears in `.err` and `slurp logs` surfaces it.

---

## 7. Command Generation Matrix

| User input | Nodes | GPUs | Generated command |
|------------|-------|------|-------------------|
| `python train.py` | 1 | 0 | `python train.py` (no wrapper) |
| `python train.py` | 1 | 4 | `python train.py` (no wrapper) |
| `python train.py` | 2 | 4 | `srun --mpi=pmi2 --cpu-bind=cores torchrun --nnodes=2 --nproc-per-node=4 ... train.py` |
| `train.sh` (existing script) | any | any | Submitted as-is; slurp does not wrap existing scripts |

**Note on existing scripts:** If the user submits an existing `.sh` file with `#SBATCH` lines, slurp passes it to `sbatch` directly. It does **not** inject `torchrun` or `srun`. The user is responsible for distributed setup inside their own script.

---

## Summary

| Concern | Implementation | Failure handling |
|---------|----------------|------------------|
| Launcher generation | `build_torchrun_command()` in `core/launcher.py` | Falls back to `pmi2` with warning if `mpi_mode` missing |
| Single-node vs multi-node | Direct execution (1 node); `srun + torchrun` (>1 node) | `torchrun` mismatch error in `.err` log |
| NCCL debugging | Auto-export `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE` for `nodes > 1` | User inspects `.err` or NCCL debug file |
| IB device selection | Profile `prologue` sets `NCCL_IB_HCA` | Cluster-agnostic core; no hard-coded devices |
| MPI mode | Profile `mpi_mode` field (`pmi2`, `pmix`, `none`) | Pass through to `srun`; surface SLURM error |
| GPU monitoring | `nvidia-smi dmon` parsing (v0.2) | Flag accepted but warns "not yet implemented" in v0.1 |
| Node failure | `sacct` status `FAILED` / `NODE_FAIL` | User resubmits; no automatic retry |
| JURECA defaults | Profile template, not core code | Core remains cluster-agnostic |

