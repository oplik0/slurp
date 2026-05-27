# Job Submission Internals

This document describes the internal machinery of job submission: SBATCH script generation, script detection, cluster-specific defaults, array handling, dependency chains, idempotency, the `slurp run` loop, and multi-node launchers.

---

## SBATCH Script Generation

For every command that is not an existing script file, slurp generates a temporary SBATCH script. The script is deterministic: given the same inputs, it produces the same directives, the same prologue, and the same log paths.

### Full Template

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --partition=<partition>
#SBATCH --nodes=<nodes>
#SBATCH --cpus-per-task=<cpus>
#SBATCH --mem=<mem>
#SBATCH --time=<time>
#SBATCH --gres=gpu:<gpus>   # or --gpus=<gpus> on non-JURECA clusters
#SBATCH --account=<account>
#SBATCH --output=<working_dir>/slurm-<name>-<job_id>.out
#SBATCH --error=<working_dir>/slurm-<name>-<job_id>.err
#SBATCH --dependency=<type>:<dependency_job_ids>
#SBATCH --array=0-<N>%<throttle>
# <any slurm_kwargs passthrough directives>

# Profile prologue (injected from profiles.toml)
<profile.prologue>

# User command
<command>
```

### Deterministic Log Paths

slurp explicitly sets `--output` and `--error` to paths in the remote working directory:

```
<working_dir>/slurm-<name>-<job_id>.out
<working_dir>/slurm-<name>-<job_id>.err
```

For job arrays:

```
<working_dir>/slurm-<name>-<array_job_id>_<task_id>.out
<working_dir>/slurm-<name>-<array_job_id>_<task_id>.err
```

**Why deterministic paths matter:**
- The client knows exactly where to tail, regardless of SLURM's `SlurmSpec` defaults.
- Log files survive job completion and are easy to locate manually.
- The `job.logs()` and `job.wait(follow_logs=True)` methods rely on these paths.
- If the user overrides `output` or `error` via `slurm_kwargs`, log streaming breaks. This is documented as an advanced-user caveat.

**Failure mode:** If the working directory does not exist on the remote side, `sbatch` rejects the script with `No such file or directory`. slurp validates the working directory existence during the sync phase (if `sync=True`) and raises `SlurmError` with a clear hint before submission.

---

## Script Detection: Commands vs Existing Scripts vs `--wrap`

slurp inspects the `command` argument to determine how to submit it.

### 1. Commands (default case)

If the argument is a shell command (e.g., `python train.py --lr 0.01`), slurp generates a temporary SBATCH script with injected directives and profile prologue.

**Detection rule:** The argument does not name an existing file on the remote filesystem, or it names a file that does not have `#!/bin/bash` as its first line.

**Wrapper script lifecycle:**
1. Generate the script content locally.
2. Upload it to a temporary path on the remote side: `/tmp/slurp-<uuid>.sh`.
3. Run `sbatch /tmp/slurp-<uuid>.sh`.
4. Delete the temp file immediately after submission.

**Failure mode:** If the temp file cannot be written (e.g., `/tmp` is full), `sbatch` fails. slurp retries once with a temp file in the working directory, then raises `SlurmError`.

### 2. Existing scripts

If the argument names an existing file with `#!/bin/bash` and at least one `#SBATCH` line, slurp submits it directly.

**Detection rule:**
1. Check if the file exists on the remote filesystem (after sync, if `sync=True`).
2. Read the first 1024 bytes.
3. If the first line starts with `#!` and the file contains `#SBATCH`, treat it as an existing script.

**Directive merging:** If the script is missing directives that slurp would normally generate (e.g., `--partition`, `--gres`), slurp prepends them. User directives in the script take precedence over slurp-generated defaults, but CLI/Python arguments override both.

**Precedence:**
1. `slurm_kwargs` (highest)
2. Named parameters (`gpus`, `time`, etc.)
3. Script directives
4. Profile defaults
5. Built-in defaults (lowest)

**Failure mode:** If the script has a malformed `#SBATCH` line (e.g., `#SBATCH --gpus=abc`), `sbatch` fails and slurp surfaces the stderr. slurp does not validate SBATCH syntax locally.

### 3. `--wrap`

For truly trivial one-liners, slurp can use `sbatch --wrap` instead of generating a file. This is triggered when the command is short (< 200 characters) and contains no shell special characters that would require a script.

**When `--wrap` is used:**
- The command is passed directly: `sbatch --wrap "python -c 'print(1)'"`.
- The profile prologue is **not** injected (because `--wrap` does not support a preamble). Use a full script if you need prologue.
- Log paths are still set explicitly via `--output` and `--error`.

**Failure mode:** `--wrap` is limited by the shell's command-line length limit (~128 KB on most systems). For long commands, slurp falls back to a temp script.

---

## JURECA-Aware Defaults (Profile-Specific, Not Built-In)

All cluster-specific defaults live in the profile, not in `core/slurm.py`. The core is cluster-agnostic.

### JURECA Profile Template

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

**Defaults applied when `profile` matches JURECA:**

- `gpus` maps to `--gres=gpu:N` (not `--gpus=N`).
- The `prologue` is injected into every SBATCH script.
- For multi-node jobs (`nodes > 1`), the launcher wraps the command with `srun --mpi=pmi2 --cpu-bind=cores`.
- `jutil env activate -p <account>` is executed before module loads.

**These are NOT hard-coded.** If a user creates a profile with `hostname = "mycluster"`, none of the JURECA defaults apply. The user can set their own `prologue`, `mpi_mode`, and `cpu_bind` in their profile.

**Failure mode:** If the JURECA profile is missing `account`, `jutil env activate` fails with a generic error. slurp validates required profile fields (hostname, account) on first use and prompts the user interactively if they are missing.

---

## Job Arrays: Native SLURM `--array=0-N%M`

`slurp.submit_array()` generates a native SLURM job array. The array is a single `sbatch` call that produces `N+1` tasks.

### Wrapper Script Generation

For a template like `python train.py --seed {seed}` and configs `[{"seed": "1"}, {"seed": "2"}]`, slurp generates:

```bash
#!/bin/bash
#SBATCH --job-name=train_array
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH --output=/p/project1/.../slurm-train_array-12345_%a.out
#SBATCH --error=/p/project1/.../slurm-train_array-12345_%a.err
#SBATCH --array=0-1%20
#SBATCH --account=training2615

# Profile prologue
jutil env activate -p training2615
module load Stages/2024
module load CUDA/12
module load PyTorch
source $PROJECT/.venv/bin/activate

# Array mapping
SEEDS=(1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# User command
python train.py --seed $SEED
```

**Key points:**
- The wrapper script defines shell arrays for each template variable. The array index is `$SLURM_ARRAY_TASK_ID`.
- All tasks share the same SBATCH directives (resources, time, partition). Per-task resource variation is not supported in v0.1.
- The throttle (`%20`) limits concurrent tasks. Default is 20. Set to 0 for unlimited.
- Log files use `%a` (task ID) in the path: `slurm-<name>-<array_job_id>_<task_id>.out`.

**Failure mode:** If the array size exceeds the cluster's `MaxArraySize` (e.g., 10,000 on JURECA), `sbatch` fails with `Job array size exceeds limit`. slurp catches this error and suggests splitting the array into chunks.

---

## Job Dependencies: `afterok`, `afterany`, Passthrough

Dependencies are expressed as `--dependency=<type>:<job_id1>:<job_id2>...` in the SBATCH script.

**Generated directive:**
```bash
#SBATCH --dependency=afterok:12345:12346
```

**Supported types:**

| Type | SLURM Behavior |
|------|----------------|
| `afterok` | Job starts only if all dependencies exit 0 (default) |
| `afterany` | Job starts when all dependencies finish, regardless of exit code |
| `after` | Job starts when all dependencies begin running |
| `afternotok` | Job starts only if all dependencies fail |

**Passthrough:** If the user needs a complex dependency expression (e.g., `afterok:12345+afterany:12346`), they can pass it via `slurm_kwargs={"dependency": "afterok:12345+afterany:12346"}`. The `depends_on` parameter is bypassed in this case.

**Validation:** Before generating the directive, slurp validates that every `Job` in `depends_on` has a `job_id`. If a job was constructed locally but never submitted, `submit()` raises `ValueError` before any network call.

**Failure mode:** If a dependency job does not exist in SLURM (e.g., it was cancelled before the dependent job was submitted), `sbatch` rejects the script with `Dependency condition invalid`. slurp surfaces this error and suggests removing the dependency.

---

## Idempotency: Hash of `(command + resources + working_dir)`

### Hash Computation

The idempotency hash is SHA-256 of a canonical JSON object:

```json
{
  "command": "python train.py --lr 0.01",
  "resources": {"gpus": 4, "time": "2:00:00", "nodes": 1},
  "working_dir": "/p/project1/training2615/user/projects",
  "profile": "jureca"
}
```

The JSON is serialized with sorted keys, no whitespace, and `ensure_ascii=True`. This guarantees the same inputs produce the same hash across platforms and Python versions.

**What is NOT included in the hash:**
- `job_id` (not known at submission time)
- `submitted_at` (time-varying)
- `experiment` (metadata, not part of the job spec)
- `name` (derived from command, not a distinguishing factor)
- `depends_on` (dependencies change the DAG, not the job itself)

### 30-Second Window

After computing the hash, slurp checks the local store:

```python
if hash in store.idempotency:
    previous = store.idempotency[hash]
    age = now - previous.submitted_at
    if age < 30 seconds:
        warn("Job {previous.job_id} with identical spec submitted {age}s ago. Submit again? [y/N]")
        if not user_confirms:
            return previous_job
```

**Why 30 seconds?** This is long enough to catch the common "SSH dropped, did it submit?" retry, but short enough to not block legitimate rapid resubmissions (e.g., hyperparameter sweeps where the same command is submitted with different `experiment` tags).

**What happens after 30 seconds?** The hash entry is removed from the `idempotency` mapping. The job record remains in the store, but duplicate detection no longer triggers. The user can submit the same spec again without warning.

### `sacct` Fallback on Ctrl+C

If the user interrupts `submit()` with Ctrl+C during the SSH round-trip, the job may or may not have been submitted. The state is ambiguous.

**Recovery:**
1. Catch `KeyboardInterrupt` during the `sbatch` call.
2. Query `sacct -S now-5minutes -u <user> --format=JobID,JobName,State --noheader`.
3. Look for a job with the same name submitted in the last 5 minutes.
4. If found, return the existing `Job` with a warning: `Job 12345 was already submitted. Returning existing job.`
5. If not found, re-raise the `KeyboardInterrupt`.

**Limitation:** This heuristic relies on job name matching. If the user submitted a different job with the same name in the last 5 minutes, the fallback may return the wrong job. The user can override with `submit(..., name="unique_name")`.

---

## `slurp run` Loop Internals

`slurp run` is a blocking CLI command that submits a job and then waits for it to complete, streaming logs and exiting with the job's exit code.

### Loop Steps

1. **Submit** — Call `slurp.submit()` with `sync=True`.
2. **Tail logs** — Start two background tasks:
   - Incremental tail of `.out` every 2 seconds.
   - Incremental tail of `.err` every 2 seconds.
   - Both use `tail -c +<offset>` and update `log_offsets.json` after each read.
3. **Poll `sacct`** — Every 5 seconds, query `sacct` for the job's state.
4. **Exit with job code** — When `sacct` reports a terminal state:
   - If `COMPLETED` with exit code 0, exit 0.
   - If `FAILED`, `TIMEOUT`, or `CANCELLED`, exit with the job's exit code (or 1 if exit code is unavailable).
   - Print final status and a summary of `stdout`/`stderr` tail.
5. **Reconnect on SSH drop** — If the SSH connection drops:
   - Detect via `asyncssh` connection error.
   - Trigger control master respawn: `ssh -MNf <profile>`.
   - Resume log tailing from the last stored offset.
   - Resume `sacct` polling.
   - If reconnect fails after 30 seconds of exponential backoff, raise `SSHError` and exit 1.

**Log streaming details:**
- The tail commands run over the existing `asyncssh` connection (via the control master socket). No new SSH connection is opened per poll.
- `asyncssh` multiplexes multiple concurrent `tail` processes over one socket.
- If the control master dies, all tail streams pause. After respawn, they resume from the last offset.

**Failure modes:**
- `SSHError` during reconnect — Exit 1 with hint: `SSH connection lost. Check network and try slurp run again.`
- `TimeoutError` — If `time` limit is reached, the job is killed by SLURM and `slurp run` exits with the job's exit code.
- `KeyboardInterrupt` — If the user presses Ctrl+C during the loop, `scancel` is sent and the program exits 130.

---

## Multi-Node Launcher: `TorchrunLauncher` Auto-Generation

When `nodes > 1` or `gpus > 1` per node, slurp auto-generates a multi-node launcher command. The default launcher is `TorchrunLauncher`.

### Generated Command

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=1

srun --mpi=pmi2 --cpu-bind=cores --distribution=block:cyclic \
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-id=$SLURM_JOB_ID \
    train.py
```

**Auto-injected environment variables:**
- `MASTER_ADDR` — First node in the allocation.
- `MASTER_PORT` — Fixed at 29500 (configurable per profile).
- `OMP_NUM_THREADS` — Set to 1 to prevent oversubscription.

**Profile-specific launcher extras:**
- `mpi_mode` — Passed to `srun --mpi=<mode>` (default `pmi2`).
- `cpu_bind` — Passed to `srun --cpu-bind=<mode>` (default `cores`).
- `distribution` — Passed to `srun --distribution=<mode>` (default `block:cyclic`).

**When the launcher is triggered:**
- `nodes > 1` — Always.
- `nodes == 1` and `gpus > 1` — Only if `launcher="auto"` (default). The user can disable with `launcher="none"`.

**Failure mode:** If `torchrun` is not in the remote `PATH`, the job fails at runtime. The profile `prologue` should include `module load PyTorch` or `source .venv/bin/activate` to ensure `torchrun` is available. slurp does not validate `torchrun` existence before submission.

---

## `srun` vs Direct Execution for Single-Node

For single-node, single-GPU jobs, slurp does **not** wrap the command with `srun`. The command runs directly in the SBATCH script:

```bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

python train.py
```

For single-node, multi-GPU jobs (`nodes=1`, `gpus=4`), the behavior depends on the `launcher` parameter:

- `launcher="auto"` (default) — Wrap with `srun` and `torchrun` for distributed training.
- `launcher="none"` — Run directly: `python train.py`. The user is responsible for spawning processes (e.g., `torchrun` in the command string).

**Rationale:** `srun` adds overhead and may break scripts that are not MPI-aware. For single-node, single-GPU jobs, direct execution is simpler and faster. For multi-GPU, `srun` is necessary to bind processes to GPUs correctly on SLURM clusters.

**Failure mode:** If a user submits a multi-GPU script that expects `torchrun` but forgets to set `launcher="auto"`, the script may run on GPU 0 only and leave GPUs 1–3 idle. The CLI warns when `gpus > 1` and `launcher="none"` is used.
