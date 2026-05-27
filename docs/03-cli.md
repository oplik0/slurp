# 03 — CLI Specification (v0.1)

## 1. Argument Parsing & Command Separation

`slurp` uses a hybrid argument parser (Typer + custom post-processing) designed to feel natural for researchers who pass flags to both `slurp` and their training scripts.

### The `--` Separator

Arguments after `--` are **always** treated as the user's command, even if they look like `slurp` flags:

```bash
# Explicit separation (recommended for clarity)
slurp submit --gpus 4 -- python train.py --lr 0.01 --time 10
#                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                              user's command; --time is NOT a slurp flag

# Implicit separation (works when the command starts with a non-flag token)
slurp submit --gpus 4 python train.py --lr 0.01
#                           ^^^^^^^^^^^^^^^^^^^^
#                           trailing positional args become the command
```

### Parsing Rules

1. **Flags before `--`** are consumed by `slurp`. Unknown flags raise `typer.BadParameter` immediately.
2. **Flags after `--`** are untouched and passed verbatim to the remote shell.
3. **If no `--` is present**, all tokens after the last recognized `slurp` flag are treated as the command.
4. **Ambiguity resolution:** If a trailing positional arg matches a `slurp` flag name (e.g., `python train.py --time 10` without `--`), the parser uses the *first* occurrence of a known flag as the split point. To avoid ambiguity, always use `--` when the user's command contains tokens that look like slurp flags.

### Edge Case: No Command Provided

```bash
$ slurp submit --gpus 4
Error: Missing command. Provide a command after `--` or as trailing arguments.
Hint: slurp submit --gpus 4 -- python train.py
Exit code: 2
```

---

## 2. Resource Flags & SLURM Mapping

All resource flags are flat — there is no "Simple/Standard/Advanced" tier. Every flag maps to exactly one SBATCH directive or CLI passthrough.

| `slurp` Flag | SLURM Directive | Default | Notes |
|--------------|-----------------|---------|-------|
| `--gpus N` | `--gres=gpu:N` (JURECA) or `--gpus=N` (generic) | `0` | Cluster detected via profile hostname pattern |
| `--nodes N` | `--nodes=N` | `1` | Must be ≥ 1 |
| `--time T` | `--time=T` | `2:00:00` | Format: `[[HH:]MM:]SS` or `D-HH:MM:SS` |
| `--mem M` | `--mem=M` | `32G` | Accepts `G`, `M`, `T` suffixes |
| `--cpus N` | `--cpus-per-task=N` | `8` | Per-task CPUs |
| `--partition P` | `--partition=P` | profile default | Prompted on first run if missing |
| `--constraint C` | `--constraint=C` | none | Hardware constraint string |
| `--qos Q` | `--qos=Q` | none | Quality-of-service class |
| `--account A` | `--account=A` | profile default | Required on most clusters |
| `--mail-type T` | `--mail-type=T` | none | `BEGIN`, `END`, `FAIL`, `ALL` |
| `--slurm-kwargs K=V` | `--K=V` | none | Passthrough; can be repeated |
| `--name N` | `--job-name=N` | auto-generated | Slugified to `[a-zA-Z0-9_-]+` |
| `--experiment E` | injected into job metadata | none | Filter tag for `watch` / `list` |
| `--snapshot` | triggers `rsync` + remote copy | `false` | Copies tree to `$PROJECT/.slurp/runs/<job_id>/` |

### `--slurm-kwargs` Examples

```bash
# Single passthrough
slurp submit --slurm-kwargs exclude=node05 python train.py

# Multiple passthroughs
slurp submit --slurm-kwargs constraint=a100 --slurm-kwargs ntasks-per-node=8 python train.py
```

---

## 3. Command Reference

### 3.1 `slurp submit`

**Syntax:**
```
slurp submit [OPTIONS] [--] <command...>
```

**Purpose:** Fire-and-forget job submission. Returns immediately after `sbatch` succeeds.

**Flags:** All resource flags from §2, plus:
- `--wait` — block until job reaches terminal state (internal flag; prefer `slurp run`)
- `--follow-logs` — stream logs while waiting (internal flag; prefer `slurp run`)
- `--profile P` — use profile `P` instead of default
- `--working-dir PATH` — remote working directory (default: profile `sync.remote` or current directory name)

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Job submitted successfully |
| 1 | `sbatch` failed (SLURM rejected the job) |
| 2 | Argument parsing error (missing command, invalid flag) |
| 3 | `rsync` failed (code not synced; job not submitted) |
| 10 | SSH connection failed (retryable) |
| 130 | User interrupted (Ctrl+C) |

**Examples:**

```bash
# Basic usage
slurp submit python train.py --lr 0.01 --gpus 4

# Explicit separator with training flags that look like slurp flags
slurp submit --gpus 4 --nodes 2 -- python train.py --time 10 --lr 0.01

# Using a non-default profile
slurp submit --profile helga --gpus 8 python train.py

# Snapshot mode for reproducibility
slurp submit --snapshot --experiment exp_v1 python train.py

# Full resource override
slurp submit --gpus 4 --nodes 2 --time 4:00:00 --mem 128G --partition dc-gpu \
  -- python train.py --batch-size 64
```

**Error handling:**
- If `rsync` fails (e.g., disk quota exceeded), the job is **not** submitted. The CLI prints the `rsync` stderr and exits with code 3.
- If `sbatch` returns a non-zero exit code, `slurp` parses the stderr for known SLURM errors (invalid account, invalid partition, QoS violation) and raises the corresponding `SlurmError` with a actionable hint.
- If the SSH control master is dead, `slurp` auto-respawns it with exponential backoff (1s, 2s, 4s, max 30s). If all retries fail, exits with code 10.

---

### 3.2 `slurp run`

**Syntax:**
```
slurp run [OPTIONS] [--] <command...>
```

**Purpose:** Blocking submit with live log streaming. Equivalent to `slurp submit --wait --follow-logs`.

**Behavior:**
1. Submit the job via `sbatch`
2. Start incremental log tailing (`tail -c +offset` every 2s)
3. Poll `sacct` every 5s for terminal state
4. On terminal state: print final status and exit with the job's exit code
5. On SSH disconnect: reconnect via control master, resume from last log offset

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Job completed successfully |
| 1 | Job failed (non-zero exit code from user script) |
| 2 | Argument parsing error |
| 3 | `rsync` failed |
| 10 | SSH connection failed |
| 124 | Timeout (`--timeout` exceeded) |
| 130 | User interrupted (Ctrl+C) — cancels the job unless `--no-cancel-on-sigint` |

**Examples:**

```bash
# Standard blocking run
slurp run python train.py --gpus 4

# With timeout (fails if job runs longer than 2 hours)
slurp run --timeout 2h python train.py

# Do not cancel on Ctrl+C (allow job to continue)
slurp run --no-cancel-on-sigint python train.py
```

**Error handling:**
- If the user's script exits non-zero, `slurp run` exits with the same code and prints the last 50 lines of stderr.
- If the job is killed by SLURM (OOM, timeout, node failure), `slurp run` exits with code 1 and prints the `sacct`-derived reason (`OUT_OF_MEMORY`, `TIMEOUT`, `NODE_FAIL`).

---

### 3.3 `slurp submit-array`

**Syntax:**
```
slurp submit-array [OPTIONS] [--] <command_template...>
slurp submit-array [OPTIONS] [--] <command> --<param> v1,v2,v3
```

**Purpose:** Submit a SLURM job array where each task runs the command with a different parameter value.

**Flags:** All resource flags from §2, plus:
- `--throttle N` — max concurrent tasks (default: 20, maps to `%20`)

**Array generation rules:**
- If the command contains `{param}` placeholders, they are expanded per task.
- If no placeholders are found, `slurp` generates a wrapper script that exports environment variables `SEED`, `LR`, etc. based on the comma-separated values.
- Log files: `slurm-<name>-<array_job_id>_<task_id>.out`

**Examples:**

```bash
# Template syntax
slurp submit-array "python train.py --seed {seed}" --seed 1,2,3,4,5 --gpus 4

# Implicit wrapper generation
slurp submit-array python train.py --seed 1,2,3,4,5 --gpus 4
# Internally generates wrapper that sets seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Throttle concurrent tasks
slurp submit-array --throttle 10 python train.py --lr 0.001,0.01,0.1 --gpus 4
```

**Exit codes:** Same as `slurp submit`, plus:
| Code | Meaning |
|------|---------|
| 4 | Invalid array parameter (empty list, mismatched lengths) |

**Error handling:**
- Mismatched parameter list lengths raise a parser error before any remote call.
- If one array task fails, `slurp` still submits the full array. Use `slurp watch` or `slurp logs` to inspect individual tasks.

---

### 3.4 `slurp watch`

**Syntax:**
```
slurp watch [OPTIONS]
```

**Purpose:** Live table of all tracked jobs, updated every 2–5s.

**Flags:**
- `--experiment E` — filter to jobs tagged with experiment `E`
- `--refresh N` — poll interval in seconds (default: 5)
- `--no-progress` — hide `progress.jsonl` columns

**Display columns:**
| Column | Source |
|--------|--------|
| Job ID | `sacct` |
| Name | Job metadata |
| Status | `sacct` (`PENDING`, `RUNNING`, `COMPLETED`, etc.) |
| Partition | `sacct` |
| GPUs | Resource request |
| Time elapsed | `sacct` |
| Step / Total | `progress.jsonl` (if present) |
| Latest metric | `progress.jsonl` (last `metrics` dict) |

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | User quit (`q` or Ctrl+C) |
| 10 | SSH connection lost and could not reconnect |

**Error handling:**
- If SSH drops, `watch` attempts to respawn the control master once. If that fails, it prints a stale snapshot and exits with code 10.
- Jobs not found in `sacct` are shown as `UNKNOWN` with a `?` indicator.

---

### 3.5 `slurp logs`

**Syntax:**
```
slurp logs [OPTIONS] <job_id>
```

**Purpose:** Print stdout/stderr of a job.

**Flags:**
- `--follow, -f` — blocking tail (like `tail -f`)
- `--tail N` — print last N lines (default: 100)
- `--stderr` — show stderr instead of stdout
- `--task-id N` — for array jobs, show logs of task `N`

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Log output printed successfully |
| 5 | Job ID not found in local store or SLURM |
| 10 | SSH connection failed |

**Examples:**

```bash
# Last 100 lines of stdout
slurp logs 12345

# Follow live output
slurp logs 12345 --follow

# Array task stderr
slurp logs 12345 --task-id 3 --stderr --follow
```

**Error handling:**
- If the log file does not exist yet (job is `PENDING`), `slurp` prints a warning and waits up to 30s for the file to appear (when following).
- On disconnect during `--follow`, `slurp` persists the byte offset to `~/.local/share/slurp/log_offsets.json` and resumes on reconnect.

---

### 3.6 `slurp status`

**Syntax:**
```
slurp status <job_id>
```

**Purpose:** One-shot status query for a single job. Prints a Rich panel with all metadata.

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Query succeeded |
| 5 | Job ID not found |
| 10 | SSH connection failed |

**Output format:**
```
┌──────── Job 12345 ─────────┐
│ Name:       train          │
│ Status:     RUNNING        │
│ Partition:  dc-gpu         │
│ Nodes:      2              │
│ GPUs:       8              │
│ Time:       01:23:45       │
│ Exit code:  —              │
└────────────────────────────┘
```

---

### 3.7 `slurp list`

**Syntax:**
```
slurp list [OPTIONS]
```

**Purpose:** List all tracked jobs. Non-interactive; suitable for scripts.

**Flags:**
- `--experiment E` — filter by experiment
- `--status S` — filter by status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, etc.)
- `--json` — output as JSON (one object per line)
- `--limit N` — max rows (default: 50)

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | List printed |
| 10 | SSH connection failed |

**Examples:**

```bash
# Human-readable table
slurp list

# JSON for downstream processing
slurp list --experiment exp_v1 --status RUNNING --json | jq '.job_id'

# Recently failed jobs
slurp list --status FAILED --limit 10
```

---

### 3.8 `slurp cancel`

**Syntax:**
```
slurp cancel <job_id> [<job_id> ...]
slurp cancel-all --experiment <experiment>
```

**Purpose:** Cancel one or more SLURM jobs.

**Flags:**
- `--experiment E` — cancel all jobs tagged with experiment `E`
- `--force` — skip confirmation prompt

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Cancellation succeeded (or job was already terminal) |
| 6 | Permission denied (cannot cancel another user's job) |
| 5 | Job ID not found |
| 10 | SSH connection failed |

**Examples:**

```bash
# Cancel single job
slurp cancel 12345

# Cancel multiple jobs
slurp cancel 12345 12346 12347

# Cancel all jobs in an experiment
slurp cancel-all --experiment exp_v1 --force
```

**Error handling:**
- Cancelling a job that is already `COMPLETED` or `FAILED` is a no-op (exit code 0).
- Cancelling a job you do not own raises `SlurmError` with exit code 6.

---

### 3.9 `slurp sync`

**Syntax:**
```
slurp sync [OPTIONS]
```

**Purpose:** Sync local code to the remote working directory without submitting a job.

**Flags:**
- `--profile P` — target profile
- `--dry-run` — show what would be synced without transferring

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Sync completed |
| 3 | `rsync` failed |
| 10 | SSH connection failed |

**Example:**
```bash
# Verify sync before submitting
slurp sync --dry-run
slurp submit python train.py --gpus 4
```

---

### 3.10 `slurp pull`

**Syntax:**
```
slurp pull <job_id> --local <path>
```

**Purpose:** Download results from a completed job's remote working directory to a local path.

**Flags:**
- `--local PATH` — destination directory (required)
- `--include PATTERN` — rsync include filter (repeatable)
- `--exclude PATTERN` — rsync exclude filter (repeatable)

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Pull completed |
| 5 | Job ID not found |
| 7 | Job not in terminal state (cannot pull from RUNNING job) |
| 10 | SSH connection failed |

**Example:**
```bash
slurp pull 12345 --local ./outputs/run-12345
```

**Error handling:**
- Pulling from a `RUNNING` job raises a warning and exits with code 7. Use `--force` to override.
- If the remote directory was deleted (e.g., scratch cleanup), exits with code 3 and prints the remote path.

---

### 3.11 `slurp config`

**Syntax:**
```
slurp config add-profile <name> [OPTIONS]
slurp config list-profiles
slurp config show-profile <name>
slurp config remove-profile <name>
```

**Purpose:** Manage connection profiles stored in `~/.config/slurp/profiles.toml`.

**Flags for `add-profile`:**
- `--hostname HOST` — login node hostname
- `--user USER` — SSH username
- `--key-file PATH` — SSH private key
- `--proxy-jump HOST` — jump host (optional)
- `--partition P` — default SLURM partition
- `--account A` — default SLURM account
- `--default` — set as default profile

**Exit codes:**
| Code | Meaning |
|------|---------|
| 0 | Profile operation succeeded |
| 8 | Profile already exists (add) or not found (show/remove) |
| 10 | SSH connection failed (when testing new profile) |

**Examples:**

```bash
# Add a new profile
slurp config add-profile jureca \
  --hostname jrlogin \
  --user alice \
  --key-file ~/.ssh/id_ed25519 \
  --partition dc-gpu \
  --account training2615 \
  --default

# List all profiles
slurp config list-profiles

# Inspect a profile
slurp config show-profile jureca

# Remove a profile
slurp config remove-profile jureca
```

**Error handling:**
- Adding a profile with the same name prompts for overwrite confirmation unless `--force` is passed.
- `add-profile` attempts an SSH connection to verify host reachability. If the connection fails, the profile is saved but a warning is printed.

---

## 4. Global Flags

The following flags are valid for **all** commands:

| Flag | Effect |
|------|--------|
| `--profile P` | Use profile `P` instead of default |
| `--verbose, -v` | Increase log verbosity (repeatable: `-vv`) |
| `--quiet, -q` | Suppress non-error output |
| `--json` | Output machine-readable JSON (where applicable) |
| `--dry-run` | Show what would happen without executing |
| `--help` | Show command-specific help |

---

## 5. Exit Code Summary

| Code | Meaning | Commands |
|------|---------|----------|
| 0 | Success | All |
| 1 | SLURM / job failure | `submit`, `run`, `submit-array` |
| 2 | Argument parsing error | All |
| 3 | Sync / rsync failure | `submit`, `run`, `sync` |
| 4 | Invalid array parameters | `submit-array` |
| 5 | Job ID not found | `logs`, `status`, `cancel`, `pull` |
| 6 | Permission denied | `cancel` |
| 7 | Job not terminal (pull) | `pull` |
| 8 | Profile error | `config` |
| 10 | SSH transport failure | All |
| 124 | Timeout | `run` |
| 130 | User interrupt (Ctrl+C) | All interactive commands |

---

## 6. Idempotency & Duplicate Submission Guard

`slurp submit` stores a hash of `(command + resources + working_dir)` in the local job store. If the user submits an identical spec within 30 seconds and the previous job is still `PENDING`, the CLI warns:

```
Warning: Job 12345 with identical spec submitted 5s ago. Submit again? [y/N]
```

If the user pressed Ctrl+C during a previous `sbatch` call (creating ambiguous state), `slurp` queries `sacct -S now-5minutes` to check whether the job actually landed before offering the duplicate guard.

This guard is **client-side only** — there is no distributed token or remote state machine. It protects against the most common accidental duplicate: the impatient retry.
