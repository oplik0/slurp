# Job Submission Internals

This document describes how slurp transforms a user command into a submitted SLURM job. It covers SBATCH script generation, script detection, cluster-specific defaults, job arrays, dependencies, idempotency, and deterministic log paths.

---

## SBATCH Script Generation

For a typical command invocation:

```python
job = slurp.submit("python train.py --lr 0.01", gpus=4, time="2:00:00")
```

slurp generates a temporary wrapper script on the remote login node, submits it with `sbatch`, and then deletes the temp file. The generated script has three sections:

1. **SBATCH directives**
2. **Profile prologue**
3. **User command**

**Example generated script:**

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --output=/p/project1/training2615/user/projects/slurm-train-12345.out
#SBATCH --error=/p/project1/training2615/user/projects/slurm-train-12345.err
#SBATCH --account=training2615

# Profile prologue (injected from profile.toml)
jutil env activate -p training2615
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate

# User command
python train.py --lr 0.01
```

**Directive generation rules:**

- Named parameters (`gpus`, `nodes`, `time`, etc.) map to directives with known SLURM names. `gpus` is translated to `--gres=gpu:N` on JURECA and `--gpus=N` on generic clusters.
- `slurm_kwargs` is merged last. Any key in `slurm_kwargs` overrides a named parameter with the same directive name.
- `--output` and `--error` are always set explicitly by slurp to deterministic paths (see Log File Path Determinism below). The user can override them via `slurm_kwargs`, but doing so breaks `job.logs()` and `job.wait(follow_logs=True)`.
- The `--job-name` directive defaults to a slug derived from the command (e.g., `python-train-py`). The user can override it with the `name` parameter.

**Failure modes:**
- `SlurmError` — `sbatch` rejects the script because of an invalid directive (bad partition, oversubscribed account, malformed time string). The stderr from `sbatch` is surfaced directly in the exception message.
- `SSHError` — The temp file cannot be written to the remote working directory (permissions, full disk, disconnected SSH).

---

## Script Detection: Command vs. Existing Script vs. `--wrap`

slurp accepts three kinds of `command` input and detects which one it is:

### 1. Inline command (default)

If the string does not name an existing file on the remote side, slurp treats it as a shell command and wraps it in a generated SBATCH script.

```python
job = slurp.submit("python train.py --lr 0.01", gpus=4)
```

This is the common case for ML workflows. The generated script includes directives, prologue, and the command.

### 2. Existing SBATCH script

If the string names a file that exists on the remote side and contains `#!/bin/bash` and at least one `#SBATCH` line, slurp submits it directly with `sbatch`. Any missing directives (e.g., `--account` or `--partition`) are prepended before the existing `#SBATCH` lines so that slurp-provided values take precedence.

```bash
# Remote file: train.sh
#!/bin/bash
#SBATCH --job-name=custom_train
#SBATCH --time=4:00:00

echo "Running custom script"
python train.py
```

```python
job = slurp.submit("train.sh", gpus=4, partition="dc-gpu")
```

The generated submission script will have slurp's `--partition=dc-gpu` and `--gres=gpu:4` prepended above the user's `#SBATCH` lines.

### 3. `--wrap` (trivial one-liners)

If the user passes `wrap=True`, slurp uses `sbatch --wrap="command"` instead of generating a temp file. This avoids file I/O on the remote node and is suitable for truly trivial commands.

```python
job = slurp.submit("echo hello", wrap=True)
```

**Tradeoffs:**
- `wrap=True` does not support profile prologue injection (the command runs in the default shell environment).
- `wrap=True` disables deterministic log paths; SLURM uses its default `slurm-%j.out` naming, which breaks `job.logs()`.
- Use `wrap=True` only for debugging or smoke tests, not for production training jobs.

**Failure mode:** If the user passes an existing file path but the file lacks `#SBATCH` lines, slurp treats it as an inline command and the job fails at runtime because the shell tries to execute a path that exists but is not a script. slurp does not introspect file contents beyond the `#!/bin/bash` + `#SBATCH` heuristic.

---

## JURECA-Aware Defaults (Profile-Specific, Not Built-In)

All cluster-specific setup lives in the profile's `prologue` and per-profile defaults. The core `slurm.py` module is cluster-agnostic.

**JURECA profile template:**

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
"""

mpi_mode = "pmi2"
cpu_bind = "cores"
```

**What the JURECA profile provides:**
- `jutil env activate` for project environment setup.
- `module load` sequence for CUDA, PyTorch, and stage.
- `mpi_mode` and `cpu_bind` values for multi-node launcher generation.
- Default `partition` and `account` so the user can omit them.

**These defaults are NOT hard-coded in slurp core.** A new cluster (e.g., Helmholtz AI, LRZ) gets a new profile with a different `prologue` and different per-profile defaults. The core codebase does not branch on `hostname == "jureca"`.

**Failure mode:** If a profile's `prologue` contains a malformed shell command, the job fails at runtime. The error appears in the job's `.err` log, which `slurp logs` surfaces immediately. slurp does not validate shell syntax at submission time.

---

## Job Arrays

`slurp.submit_array()` generates a native SLURM job array (`--array=0-N%M`) and a wrapper script that maps `SLURM_ARRAY_TASK_ID` to parameter values.

**Generated wrapper script:**

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=0-4%20
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --output=/p/project1/training2615/user/projects/slurm-sweep-%A_%a.out
#SBATCH --error=/p/project1/training2615/user/projects/slurm-sweep-%A_%a.err
#SBATCH --account=training2615

# Profile prologue
jutil env activate -p training2615
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate

# Array parameter mapping
SEEDS=(1 2 3 4 5)
LR=(0.01 0.001 0.0001 0.01 0.001)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
LR=${LR[$SLURM_ARRAY_TASK_ID]}

# User command
python train.py --seed "$SEED" --lr "$LR"
```

**Key mechanics:**
- `configs` is a list of dictionaries. The wrapper script declares one shell array per unique key and indexes into it with `$SLURM_ARRAY_TASK_ID`.
- The `--array` directive uses `%M` throttling. Default `throttle=20`. Set `throttle=0` for unlimited concurrency.
- Log files use the `%A_%a` pattern (`%A` = base job ID, `%a` = task ID), producing deterministic paths: `slurm-<name>-<array_job_id>_<task_id>.out`.
- Task IDs are zero-indexed internally. The CLI generates `configs` automatically when the user passes comma-separated values (e.g., `--seed 1,2,3,4,5`).

**Failure modes:**
- `ValueError` — A config dictionary is missing a key referenced in the template, or `configs` is empty.
- `SlurmError` — Array size exceeds cluster limit (e.g., JURECA caps arrays at 10,000 tasks).
- `SlurmError` — Throttle value `%M` exceeds partition limit.

---

## Job Dependencies

slurp exposes SLURM dependency chains via `depends_on` and `depends_on_type`.

```python
pre = slurp.submit("python preprocess.py", gpus=0, time="1:00:00")
train = slurp.submit("python train.py", gpus=4, depends_on=[pre])
```

This generates:

```bash
#SBATCH --dependency=afterok:12345
```

**Valid dependency types:**

| Type | SLURM behavior |
|------|----------------|
| `afterok` | Start only if all dependencies exit with code 0 (default) |
| `afterany` | Start when all dependencies finish, regardless of exit code |
| `after` | Start when all dependencies begin running |
| `afternotok` | Start only if all dependencies fail |

**Multiple dependencies:**

```python
c = slurp.submit("python c.py", gpus=4, depends_on=[a, b], depends_on_type="afterany")
```

This produces `afterany:12345,12346`.

**Validation:** Before any network call, `submit()` checks that every `Job` in `depends_on` has a non-empty `job_id`. If a dependency was constructed locally but never submitted, `submit()` raises `ValueError` immediately.

**Failure mode:** If a dependency job is cancelled or fails before the dependent job starts, SLURM leaves the dependent job in `PENDING` with a `DependencyNeverSatisfied` reason. `slurp watch` shows this reason in the status table. The user must cancel the hanging job manually.

---

## Idempotency

A dropped SSH connection during `sbatch` creates ambiguous state: the user does not know whether the job was submitted. The natural response is to retry, which can produce duplicate jobs. slurp mitigates this with client-side idempotency.

### Hash computation

The idempotency key is a SHA-256 hash of the canonical JSON representation of:

```json
{
  "command": "python train.py --lr 0.01",
  "resources": {"gpus": 4, "time": "2:00:00"},
  "working_dir": "/p/project1/training2615/user/projects",
  "profile": "jureca"
}
```

Keys are sorted alphabetically. Values are normalized (e.g., paths resolved to absolute, integers as numbers, not strings).

### 30-second window

If the user submits the exact same spec within 30 seconds and the previous job is still `PENDING`, slurp warns:

```
⚠ Job 12345 with identical spec submitted 5s ago. Submit again? [y/N]
```

A negative response cancels the new submission without error. A positive response proceeds normally, creating a duplicate.

### Ctrl+C fallback

If the user interrupts `submit()` with Ctrl+C during the SSH round-trip, slurp cannot know whether `sbatch` succeeded. On catching `KeyboardInterrupt`, slurp queries `sacct -S now-5minutes --format=JobID,SubmitLine --noheader` for a job matching the command hash within the last 5 minutes. If found, it returns the existing `Job` handle. If not found, it raises `SlurpError` with the message `Submission interrupted and no matching job found in SLURM accounting.`

### Storage

The hash is stored in the local job store (`~/.local/share/slurp/jobs.json`) in two places: inside the job record and in a top-level `idempotency` mapping. Entries older than 30 seconds are pruned on every store write.

**Failure mode:** If the local store is corrupted and rebuilt (see State Management), idempotency hashes are lost. A duplicate submission within the window may go undetected until the store is repopulated.

---

## Log File Path Determinism

slurp explicitly sets `--output` and `--error` on every generated script to deterministic paths in the remote working directory. This ensures the client knows exactly where to tail, regardless of SLURM defaults or cluster configuration.

**Path templates:**

| Job type | stdout path | stderr path |
|----------|-------------|-------------|
| Single job | `slurm-<name>-<job_id>.out` | `slurm-<name>-<job_id>.err` |
| Array job | `slurm-<name>-<array_job_id>_<task_id>.out` | `slurm-<name>-<array_job_id>_<task_id>.err` |

**Examples:**
- Single job: `slurm-train-12345.out`
- Array task 3: `slurm-sweep-12345_3.out`

**Why determinism matters:**
- `job.logs()` and `job.wait(follow_logs=True)` rely on knowing the exact remote path.
- Log offsets are keyed by `job_id` in the local store. If the path changed between submission and read, offsets would point to the wrong file.
- Multiple concurrent `watch` or `logs` calls on the same job all reference the same path without coordination.

**Override behavior:**
Advanced users can override paths via `slurm_kwargs={"output": "/custom/path.out"}`. slurp respects the override and passes it directly to SBATCH. However, `job.logs()` will fail with `FileNotFoundError` unless the custom path is also readable from the remote working directory and the user manages their own log streaming.

**Failure mode:** If the remote working directory does not exist or is not writable, `sbatch` fails at submission time with a clear SLURM error: `sbatch: error: Unable to open file ...`. This is caught and raised as `SlurmError` before the job is queued.

---

## Summary

| Concern | Implementation | Failure handling |
|---------|----------------|------------------|
| SBATCH generation | Temp wrapper script with directives + prologue + command | `sbatch` stderr surfaced as `SlurmError` |
| Script detection | File existence + `#SBATCH` heuristic | Misclassified scripts fail at runtime |
| Cluster defaults | Profile `prologue` and per-profile defaults | Malformed prologue appears in `.err` log |
| Job arrays | `--array=0-N%M` + shell array mapping | `ValueError` for missing keys; `SlurmError` for size limits |
| Dependencies | `--dependency=afterok:job_id` | Hanging jobs if dependency fails; visible in `watch` |
| Idempotency | SHA-256 hash + 30 s window + `sacct` fallback | Lost if store is corrupted and rebuilt |
| Log paths | Deterministic `slurm-<name>-<job_id>.out` | `FileNotFoundError` if user overrides path manually |
