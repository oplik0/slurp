# 03 — CLI Specification

## 1. Global Options

Every `slurp` command accepts the following global flags:

```
--profile NAME      Use profile NAME from ~/.config/slurp/profiles.toml
--experiment EXP    Tag all jobs in this invocation with experiment name EXP
--verbose, -v       Increase log verbosity (can be repeated: -vv)
--dry-run           Show generated SBATCH script and exit without submitting
--help              Show help for the current command
```

**Default profile resolution:** If `--profile` is omitted, `slurp` tries `default` profile, then the first profile in the file, then interactive fallback.

**Experiment tag:** The `--experiment` string is stored in the local job cache and passed to SLURM via `--comment`. It is used by `watch`, `list`, and `cancel-all` as a filter key. It has no semantic meaning to SLURM.

---

## 2. Argument Parsing Rules

`slurp` distinguishes between **slurp flags** and **user command arguments** using two mechanisms:

### 2.1 Explicit separator: `--`

Everything after `--` is treated as the user's command verbatim. Slurp flags must appear before `--`.

```bash
slurp submit --gpus 4 --time 2:00:00 -- python train.py --lr 0.01 --time 10
# slurp sees:   gpus=4, time="2:00:00"
# user command: python train.py --lr 0.01 --time 10
```

**Edge case:** If `--` appears multiple times, only the first `--` is the separator; subsequent `--` tokens are part of the user command.

### 2.2 Trailing positional arguments (no `--`)

If no `--` is present, all positional arguments after the last recognized slurp flag are treated as the user command.

```bash
slurp submit --gpus 4 python train.py --lr 0.01
# slurp sees:   gpus=4
# user command: python train.py --lr 0.01
```

**Edge case:** If a positional argument looks like a flag but is meant for the user command (e.g. a script that takes `--config`), the user must use `--` to avoid ambiguity.

```bash
slurp submit --gpus 4 python train.py --config cfg.yaml   # DANGEROUS: --config may be parsed by slurp
slurp submit --gpus 4 -- python train.py --config cfg.yaml  # SAFE
```

**Failure mode:** Typer (the CLI framework) raises a `BadParameter` error before slurp code runs. The error message indicates which flag is unknown and suggests using `--`.

---

## 3. Resource Flags (Flat Design)

All resource flags are top-level options. There are no "Simple / Standard / Advanced" tiers.

| Flag | SLURM Mapping | Default | Validation |
|------|---------------|---------|------------|
| `--gpus N` | `--gres=gpu:N` (JURECA) or `--gpus=N` | `0` | `N >= 0` |
| `--nodes N` | `--nodes=N` | `1` | `N >= 1` |
| `--time T` | `--time=T` | `"2:00:00"` | SLURM duration format |
| `--mem M` | `--mem=M` | profile default or `0` (unlimited) | e.g. `16G`, `16384` |
| `--cpus N` | `--cpus-per-task=N` | `8` | `N >= 1` |
| `--partition P` | `--partition=P` | profile default | non-empty string |
| `--constraint C` | `--constraint=C` | none | string |
| `--qos Q` | `--qos=Q` | none | string |
| `--account A` | `--account=A` | profile default | non-empty string |
| `--mail-type T` | `--mail-type=T` | none | `BEGIN`, `END`, `FAIL`, `ALL` |
| `--job-name NAME` | `--job-name=NAME` | derived from command | slugified to `[a-zA-Z0-9_-]+` |
| `--slurm-kwargs K=V` | passthrough | none | can be repeated |

**`--slurm-kwargs` passthrough:** Any SBATCH directive not covered by the flags above can be injected directly. This is the escape hatch for cluster-specific or rarely-used directives.

```bash
slurp submit python train.py --slurm-kwargs exclude=node05 --slurm-kwargs ntasks-per-node=2
```

**Failure mode:** If a `--slurm-kwargs` key conflicts with an explicit slurp flag (e.g. `--gpus 4` and `--slurm-kwargs gres=gpu:2`), the behavior is undefined and a warning is emitted. The explicit flag wins in practice, but users should not rely on this.

---

## 4. Command Reference

### 4.1 `slurp submit`

**Synopsis:**
```
slurp submit [OPTIONS] [--] <command>...
```

**Description:** Fire-and-forget job submission. Syncs code via rsync, generates an SBATCH script, submits it to SLURM, and returns immediately with the job ID.

**Workflow:**
1. Resolve profile and working directory.
2. Sync local code to remote (`rsync -avz --delete --filter=':- .gitignore'`).
3. Generate SBATCH script (directives + profile prologue + user command).
4. Run `sbatch` on the login node.
5. Store job metadata locally.
6. Print `Job <id> submitted.`

**Examples:**
```bash
# Basic submit
slurp submit python train.py --lr 0.01 --gpus 4

# With explicit time and partition override
slurp submit --time 4:00:00 --partition dc-gpu-debug python train.py --gpus 4

# Snapshot for reproducibility
slurp submit --snapshot python train.py --gpus 4

# Dry-run: preview script without submitting
slurp submit --dry-run --gpus 4 python train.py
```

**Options:** All resource flags (see §3), plus `--snapshot` and `--dry-run`.

**Exit codes:**
- `0` — Job submitted successfully.
- `1` — `SlurmError` (e.g. invalid account, partition not found).
- `2` — `SSHError` (login node unreachable).
- `3` — `SyncError` (rsync failed, e.g. disk quota exceeded on remote).
- `4` — Idempotency warning (duplicate spec within 30s; user declined resubmit).
- `10` — Profile missing and interactive fallback failed (non-TTY).

**Edge cases:**
- **Dirty git repo:** A warning is printed but submission proceeds. The user is trusted to decide whether uncommitted changes are intentional.
- **No GPUs requested (`--gpus 0`):** The job runs on CPU nodes. The `--gres=gpu:0` directive is omitted from the generated script.
- **Existing `.sbatch` script submitted:** If the user command is a file ending in `.sh` or `.sbatch` containing `#SBATCH` lines, slurp detects it and prepends missing directives rather than generating a wrapper script.

---

### 4.2 `slurp run`

**Synopsis:**
```
slurp run [OPTIONS] [--] <command>...
```

**Description:** Blocking submit with live log streaming. Internally: `slurp submit` + `job.wait(follow_logs=True)`. The command does not return until the job reaches a terminal state.

**Workflow:**
1. Submit the job (same as `slurp submit`).
2. Start incremental log tailing (`tail -c +offset`, 2-second interval).
3. Poll `sacct` every 5 seconds for terminal state.
4. On terminal state: print final status and exit with the job's exit code.
5. On SSH drop: reconnect via control master, resume from last log offset.

**Examples:**
```bash
# Run and stream logs until completion
slurp run python train.py --gpus 4

# With timeout (raises error if job exceeds duration)
slurp run --time 2:00:00 python train.py --gpus 4
```

**Exit codes:**
- `0` — Job completed successfully (SLURM exit code 0).
- `N` — Job failed with SLURM exit code N (1–255).
- `1` — `SlurmError` during submission.
- `2` — `SSHError` during streaming (reconnect exhausted).
- `124` — `slurp run` itself timed out waiting (if a local `--timeout` flag is added).

**Edge cases:**
- **Ctrl+C during `slurp run`:** The local `tail` stream stops, but the remote job continues running. A message is printed: `Detached from log stream. Job 12345 is still RUNNING. Use "slurp logs 12345 --follow" to reattach.`
- **SSH drops during long training:** The control master auto-respawns. If the control master cannot be respawned within the retry budget (4 attempts, max 30s backoff), `SSHError` is raised and `slurp run` exits. The remote job is unaffected.

---

### 4.3 `slurp submit-array`

**Synopsis:**
```
slurp submit-array [OPTIONS] [--] <command>...
slurp submit-array [OPTIONS] "<command with {placeholder}>" --<key> v1,v2,v3
```

**Description:** Submit a SLURM job array where each task runs the same command with a different parameter value. Native SLURM arrays (`--array=0-N%M`) are used internally.

**Parameter sweep syntax:**

```bash
# Implicit: flags ending in comma-separated lists become array parameters
slurp submit-array python train.py --seed 1,2,3,4,5 --gpus 4
# Generates --array=0-4%20, wrapper script sets seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Explicit template syntax
slurp submit-array "python train.py --seed {seed}" --seed 1,2,3,4,5 --gpus 4
```

**Options:** All resource flags from `submit`, plus:
- `--throttle N` — Max concurrent tasks (default: `20`). Maps to `%N` in `--array`.

**Exit codes:** Same as `submit`, plus:
- `5` — Invalid sweep specification (e.g. mismatched list lengths, placeholder not found in command).

**Edge cases:**
- **Multiple sweep parameters:** If two or more flags receive comma-separated lists, they must be the same length. A Cartesian product is not supported in v0.1.
- **Zero tasks:** If all parameter lists are empty, the command raises `SlurmError` before calling `sbatch`.
- **Log files:** Per-task logs are named `slurm-<job_name>-<array_job_id>_<task_id>.out` and `.err`.

---

### 4.4 `slurp watch`

**Synopsis:**
```
slurp watch [OPTIONS]
```

**Description:** Live table of all tracked jobs. Uses `rich.live.Live` to refresh every 2–5 seconds. Shows job name, status, experiment, elapsed time, and latest progress metrics (if `progress.jsonl` exists).

**Options:**
- `--experiment EXP` — Filter to jobs tagged with experiment `EXP`.
- `--refresh SEC` — Poll interval in seconds (default: `5`).

**Examples:**
```bash
slurp watch
slurp watch --experiment exp_v1
slurp watch --refresh 2
```

**Exit codes:**
- `0` — User quit watch (Ctrl+C or `q`).
- `2` — `SSHError` (sacct query failed; table shows cached data with stale indicator).

**Edge cases:**
- **No jobs tracked:** Table is empty with a message: `No jobs in local cache. Submit a job with "slurp submit" or run "slurp list" to reconcile.`
- **Stale data indicator:** If `sacct` fails for >2 consecutive polls, the status column shows `(stale)` in yellow.

---

### 4.5 `slurp logs`

**Synopsis:**
```
slurp logs [OPTIONS] <job_id>
```

**Description:** Print stdout/stderr of a job. Supports one-shot tail (`--tail N`) or live follow (`--follow`).

**Options:**
- `--follow, -f` — Block and stream new log lines as they are written (uses `tail -F` over SSH).
- `--tail N` — Print last N lines (default: `100`). Ignored if `--follow` is set.
- `--stderr` — Print stderr instead of stdout.

**Examples:**
```bash
slurp logs 12345
slurp logs 12345 --follow
slurp logs 12345 --tail 20 --stderr
```

**Exit codes:**
- `0` — Log printed successfully.
- `1` — Job ID not found in local cache or SLURM.
- `2` — `SSHError` during file read.

**Edge cases:**
- **Log file does not exist yet:** If the job is still PENDING, the log file may not have been created. The command prints `Log file not yet created. Job 12345 is PENDING.` and exits with code `0`.
- **Follow + disconnect:** If SSH drops during `--follow`, the offset is persisted to `~/.local/share/slurp/log_offsets.json`. The next `slurp logs --follow` resumes from the offset.
- **Array job logs:** For array jobs, `slurp logs 12345` prints the base job log (usually empty). To read a task log, use `slurp logs 12345_2` (task ID appended with underscore).

---

### 4.6 `slurp status`

**Synopsis:**
```
slurp status <job_id>
```

**Description:** Print detailed status of a single job: SLURM state, exit code (if terminal), wall time, max RSS, and node list.

**Examples:**
```bash
slurp status 12345
```

**Exit codes:**
- `0` — Status retrieved.
- `1` — Job ID not found.
- `2` — `SSHError`.

**Edge cases:**
- **Job not in local cache but known to SLURM:** The command queries `sacct` directly and prints the status. It does not add the job to the local cache (only `list` and `watch` reconcile).

---

### 4.7 `slurp list`

**Synopsis:**
```
slurp list [OPTIONS]
```

**Description:** Print a table of all tracked jobs. Reconciles local cache against `sacct` before rendering. Supports filtering by experiment and status.

**Options:**
- `--experiment EXP` — Filter by experiment tag.
- `--status S` — Filter by status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`).
- `--limit N` — Max rows to show (default: `50`).

**Examples:**
```bash
slurp list
slurp list --experiment exp_v1 --status RUNNING
slurp list --limit 10
```

**Exit codes:**
- `0` — Table printed (may be empty).
- `2` — `SSHError` during reconciliation; falls back to cached data with a warning.

---

### 4.8 `slurp cancel`

**Synopsis:**
```
slurp cancel <job_id> [<job_id> ...]
```

**Description:** Cancel one or more jobs via `scancel`. Accepts multiple job IDs in a single invocation.

**Examples:**
```bash
slurp cancel 12345
slurp cancel 12345 12346 12347
```

**Exit codes:**
- `0` — All jobs cancelled (or already terminal).
- `1` — `SlurmError` (e.g. job not found, permission denied).
- `2` — `SSHError`.

**Edge cases:**
- **Already terminal job:** `scancel` returns non-zero for completed jobs. `slurp cancel` treats this as success and prints `Job 12345 is already COMPLETED.`
- **Array job:** Cancelling the base job ID cancels all tasks. To cancel a single task, append the task ID: `slurp cancel 12345_3`.

---

### 4.9 `slurp sync`

**Synopsis:**
```
slurp sync [OPTIONS]
```

**Description:** Sync local code to the remote working directory without submitting a job. Useful for verifying that rsync works or for pre-staging code before a manual SSH session.

**Options:**
- `--profile NAME` — Target profile (global).

**Examples:**
```bash
slurp sync
slurp sync --profile jureca
```

**Exit codes:**
- `0` — Sync completed.
- `3` — `SyncError` (rsync exited non-zero).
- `2` — `SSHError`.

**Edge cases:**
- **No changes:** `rsync` exits `0` even if nothing changed. The command prints `Sync complete. 0 files transferred.`
- **Large transfers:** For trees >1 GB, the command may take minutes. There is no progress bar in v0.1; the user sees a spinner.

---

### 4.10 `slurp pull`

**Synopsis:**
```
slurp pull <job_id> [--local DIR]
```

**Description:** Download job results (stdout, stderr, and any files in the job's remote working directory) to a local directory.

**Options:**
- `--local DIR` — Destination directory (default: `./outputs/<job_id>/`).

**Examples:**
```bash
slurp pull 12345
slurp pull 12345 --local ./results/
```

**Exit codes:**
- `0` — Pull completed.
- `1` — Job ID not found.
- `3` — `SyncError` (rsync or `scp` failure).
- `2` — `SSHError`.

**Edge cases:**
- **Job still running:** Pull proceeds, but files may be incomplete. A warning is printed: `Job 12345 is RUNNING. Output may be incomplete.`
- **Missing results:** If the remote working directory is empty, the local directory is created but contains only `.out` and `.err` files.

---

### 4.11 `slurp config`

**Synopsis:**
```
slurp config add-profile NAME [OPTIONS]
slurp config list-profiles
slurp config show-profile NAME
slurp config edit-profile NAME
```

**Description:** Manage connection profiles stored in `~/.config/slurp/profiles.toml`.

**Subcommands:**

| Subcommand | Purpose |
|------------|---------|
| `add-profile` | Create a new profile interactively or non-interactively |
| `list-profiles` | Print all profiles as a table |
| `show-profile` | Print one profile as TOML |
| `edit-profile` | Open `$EDITOR` on the profile file |

**`add-profile` options:**
- `--hostname HOST` — Login node hostname.
- `--user USER` — SSH username.
- `--key-file PATH` — SSH private key path.
- `--proxy-jump HOST` — Jump host (optional).
- `--partition P` — Default SLURM partition.
- `--account A` — Default SLURM account.

**Examples:**
```bash
# Interactive first-run setup
slurp config add-profile jureca

# Non-interactive (CI/automation)
slurp config add-profile jureca --hostname jrlogin --user alice --partition dc-gpu --account training2615

slurp config list-profiles
slurp config show-profile jureca
```

**Exit codes:**
- `0` — Operation completed.
- `10` — Profile name already exists (add-profile with `--force` to overwrite).
- `11` — `~/.config/slurp/` not writable.

**Edge cases:**
- **Missing `~/.ssh/config`:** `add-profile` prompts for hostname and user. If the user has an SSH config Host entry, the profile auto-populates from it.
- **Key file not found:** A warning is printed but the profile is saved. The error will surface on the first SSH connection attempt.

---

## 5. Exit Code Summary

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | SLURM error (`SlurmError`, `JobFailedError`) |
| `2` | SSH transport error (`SSHError`) |
| `3` | Sync / file transfer error (`SyncError`) |
| `4` | Idempotency duplicate rejected by user |
| `5` | Invalid array sweep specification |
| `10` | Profile missing or not writable |
| `11` | Config directory not writable |
| `124` | Local timeout (`slurp run` waited too long) |
| `125` | Internal / unexpected error |
| `126` | CLI argument parsing error (`typer.BadParameter`) |
| `127` | Command not found (e.g. `rsync` not installed locally) |
