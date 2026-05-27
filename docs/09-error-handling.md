# 09 — Error Handling and Diagnostics

## 1. Exception Hierarchy

All `slurp` exceptions inherit from a single base class. The hierarchy is intentionally flat: four classes cover every error domain in v0.1.

```python
class SlurpError(Exception):
    """Base exception with .message, .hint, .retryable"""
    message: str
    hint: str | None
    retryable: bool

class SSHError(SlurpError):
    """Control master or asyncssh transport failure."""
    pass

class SlurmError(SlurpError):
    """SLURM binary returned non-zero or malformed output."""
    stderr_fragment: str | None

class JobFailedError(SlurmError):
    """Job reached a terminal state with non-zero exit code."""
    job_id: str
    exit_code: int
    stdout_tail: str
    stderr_tail: str
```

### 1.1 `SlurpError` (base)

**Fields:**
- `message` — Human-readable description of what failed.
- `hint` — Actionable suggestion for the most likely fix. May be `None` if the cause is genuinely unknown.
- `retryable` — Boolean. If `True`, retrying the same operation (with exponential backoff) may succeed. If `False`, retrying without changing inputs is expected to fail again.

**Example:**
```python
try:
    job = slurp.submit("python train.py", gpus=4)
except slurp.SlurmError as e:
    print(e.message)   # "sbatch: error: Invalid account or account/partition combination"
    print(e.hint)      # "Verify account name in ~/.config/slurp/profiles.toml"
    print(e.retryable) # False
```

### 1.2 `SSHError`

Raised when the hybrid SSH transport (control master + asyncssh) cannot execute a command. This includes connection timeouts, refused connections, dead control master sockets, and jump host failures.

**Fields (inherited from `SlurpError`):**
- `message` — e.g. "SSH connection to jrlogin timed out after 30s"
- `hint` — e.g. "Check VPN and try again."
- `retryable` — Always `True` for transient network issues; `False` for permanent auth failures (e.g. wrong key passphrase).

**Edge case — SSH drops during submit:** If the control master dies between `sbatch` invocation and response, `asyncssh` raises `SSHError`. The CLI auto-reconnects (see §3). In the Python API, the exception propagates unless caught by the caller.

### 1.3 `SlurmError`

Raised when a SLURM binary (`sbatch`, `sacct`, `scancel`, `squeue`) exits non-zero or returns output that cannot be parsed.

**Additional fields:**
- `stderr_fragment` — Last 2 KB of stderr from the SLURM command, sanitized (control characters stripped). This is the primary diagnostic for SLURM-side failures.

**Examples of common `SlurmError` scenarios:**

| Scenario | stderr_fragment | hint |
|----------|----------------|------|
| Invalid account | `sbatch: error: Batch job submission failed: Invalid account...` | Verify account in profiles.toml |
| Partition not found | `sbatch: error: Invalid partition name: dc-gpu-debug` | Check available partitions with `sinfo` |
| Time limit too long | `sbatch: error: Time limit exceeds partition max of 24:00:00` | Reduce --time or choose different partition |
| Disk quota exceeded | `sbatch: error: unable to create file` | Clean up remote home directory |

**Edge case — Job ID not found in `sacct`:** If a tracked job ID disappears from `sacct` (e.g. due to log rotation or the job being purged), `sacct_query` returns a `Job` with status `UNKNOWN` rather than raising. This avoids crashing `watch` and `list` when old jobs expire. The status column shows `(unknown to sacct)`.

### 1.4 `JobFailedError`

Subclass of `SlurmError` raised specifically when a job reaches `FAILED`, `TIMEOUT`, or `CANCELLED` state, or `COMPLETED` with a non-zero exit code. This is the exception you catch to handle training crashes, OOM kills, or time-limit breaches.

**Additional fields:**
- `job_id` — The SLURM job ID.
- `exit_code` — The job's exit code (may be `None` if SLURM did not record one).
- `stdout_tail` — Last 1 KB of stdout log.
- `stderr_tail` — Last 1 KB of stderr log.

**Example:**
```python
try:
    result = job.wait(follow_logs=True)
except slurp.JobFailedError as e:
    print(f"Job {e.job_id} failed with exit code {e.exit_code}")
    print("Last stderr:")
    print(e.stderr_tail)
```

**Edge case — `CANCELLED` by user vs. by system:** Both raise `JobFailedError`. The `stderr_tail` usually contains `slurmstepd: error: *** JOB CANCELLED ***` for system kills (e.g. preemption). User-initiated `scancel` typically leaves no stderr trace.

---

## 2. Error Message Format

### 2.1 CLI Rendering (Rich Panels)

When an exception propagates to the CLI top-level handler (`cli/main.py`), it is rendered as a structured Rich panel rather than a raw Python traceback.

**Layout:**
```
┌─ SlurmError ──────────────────────────────────────────┐
│ sbatch: error: Invalid account or account/partition   │
│ combination                                           │
├─ Hint ────────────────────────────────────────────────┤
│ Verify account name in ~/.config/slurp/profiles.toml  │
├─ stderr ──────────────────────────────────────────────┤
│ sbatch: error: Batch job submission failed: Invalid   │
│ account or account/partition combination              │
├─ Context ─────────────────────────────────────────────┤
│ Profile: jureca  Partition: dc-gpu  Account: ???      │
└───────────────────────────────────────────────────────┘
```

**Rules:**
- The exception class name is the panel title.
- `message` is printed in bold red.
- `hint` is printed in yellow, prefixed with a lightbulb emoji only if the terminal supports Unicode.
- `stderr_fragment` (if present) is printed in a separate `Syntax` block with `bash` highlighting.
- Raw Python tracebacks are suppressed unless `--verbose` (`-v`) is passed at least twice (`-vv`).
- `retryable=True` adds a line: `This error may resolve if you retry the command.`

### 2.2 Python API Error Behavior

In the Python API, exceptions are raised identically to the CLI — the same classes, same fields, same messages. There is no JSON error envelope or secondary error channel.

**Notebook / Jupyter:** The Rich panel renderer is not used in notebooks. Instead, the exception prints as plain text with ANSI color codes. If `IPython.display` is available, a simplified HTML representation is shown.

**Idempotency in notebooks:** Calling `job.result()` a second time returns the cached `JobResult`; it does not re-query SLURM. If the first `wait()` raised `JobFailedError`, the second `result()` call returns the `JobResult` object (not an exception), because the terminal state is now known and cached.

---

## 3. Retry Logic

### 3.1 Auto-Reconnect for SSH

The hybrid transport has a built-in retry loop for transient SSH failures:

1. **Detect failure:** `asyncssh` raises `SSHError` (or underlying `ConnectionLost`).
2. **Check control master:** Run `ssh -O check <profile>` with 0.5s timeout.
3. **Respawn if dead:** `ssh -MNf <profile>` in background.
4. **Retry command:** With exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (cap).
5. **Max attempts:** 4 retries (5 total attempts including the original).
6. **Final failure:** Raise `SSHError` with message and hint.

**Commands that trigger auto-reconnect:** `sbatch`, `sacct`, `scancel`, `squeue`, `tail`, `rsync` (via SSH socket).

**Commands that do NOT trigger auto-reconnect:** Interactive `questionary` prompts (local only), local file I/O (job store reads/writes).

### 3.2 SLURM Retry Policy

SLURM errors are **not** retried automatically, because SLURM failures are almost always deterministic (invalid account, bad partition, syntax error). The one exception is `sacct` queries during `watch`, where a single transient `sacct` failure is retried once on the next poll cycle (2–5s later) without raising.

### 3.3 Sync Retry Policy

`rsync` failures are retried once after a 2-second delay. If the second attempt fails, `SyncError` is raised immediately. This handles transient NFS flushes on the login node.

---

## 4. Exit Codes

The CLI maps exceptions to process exit codes deterministically:

| Exit Code | Source Exception | Typical Cause |
|-----------|-----------------|---------------|
| `0` | — | Success |
| `1` | `SlurmError` | Invalid SLURM directive, permission denied |
| `1` | `JobFailedError` | Job failed or timed out (non-interactive CLI) |
| `2` | `SSHError` | Network down, login node unreachable, key auth failed |
| `3` | `SyncError` | rsync/scp failure, disk quota, permission denied on remote FS |
| `4` | `IdempotencyError` | User declined duplicate submission |
| `5` | `SlurmError` (array) | Mismatched sweep parameter lengths |
| `10` | `ProfileError` | Profile missing, TTY unavailable for interactive fallback |
| `11` | `ConfigError` | Config directory not writable |
| `124` | `TimeoutError` | `slurp run` local wait timeout exceeded |
| `125` | `SlurpError` (unexpected) | Bug or unhandled edge case |
| `126` | `typer.BadParameter` | Unknown CLI flag or invalid argument format |
| `127` | `FileNotFoundError` | Required binary (`rsync`, `ssh`) not found in `$PATH` |

**Special case — `slurp run` exit code:** `slurp run` exits with the job's own exit code if the job reaches `COMPLETED` or `FAILED`. This mirrors `srun` behavior. If the job itself exits `0`, `slurp run` exits `0`. If the job exits `42`, `slurp run` exits `42`. If the transport fails before the job finishes, `slurp run` exits `2` (`SSHError`).

---

## 5. Edge Cases and Failure Modes

### 5.1 SSH Drop During `sbatch`

**Problem:** The control master dies after `sbatch` is sent but before the job ID is returned. The user does not know whether the job was actually submitted.

**Behavior:**
1. `SSHError` is raised to the CLI.
2. The CLI auto-reconnects (§3.1) and retries `sbatch`.
3. If the retry succeeds, the job ID is returned normally.
4. If the job *was* submitted during the first attempt but the response was lost, the second `sbatch` creates a duplicate job. The idempotency check (hash of command + resources + working dir) detects the duplicate within 30s and warns: `Job 12345 with identical spec submitted 5s ago. Submit again? [y/N]`
5. If the user answers `N`, the command exits with code `4`.

**Python API behavior:** The exception propagates after the retry budget is exhausted. The caller must handle ambiguous state manually (or query `sacct` for recent submissions).

### 5.2 Job Not Found in `sacct`

**Problem:** A job ID stored in the local cache no longer appears in `sacct` (purged after retention window, or job ID typo).

**Behavior:**
- `sacct_query` returns `Job` with status `UNKNOWN`.
- `slurp status` prints the `UNKNOWN` state with a hint: `Job not found in sacct. It may have been purged or the ID may be incorrect.`
- `slurp watch` and `slurp list` show the job in grey with status `UNKNOWN`.
- No exception is raised.

### 5.3 Duplicate Job Idempotency

**Problem:** A user retries `slurp submit` because they are unsure whether the first attempt worked (e.g. Ctrl+C, network blip).

**Behavior:**
- A hash of `(command + sorted(resources) + working_dir)` is stored in the local job cache at submission time.
- If a new submission with the identical hash occurs within 30 seconds and the previous job is still `PENDING`, an interactive prompt warns the user.
- If the previous job is already `RUNNING` or terminal, no warning is shown. The user may legitimately want to run the same command again.
- In non-interactive mode (no TTY), the warning is printed to stderr and the duplicate is blocked unless `--force` is passed.

**Edge case — 30-second window:** If the user retries after 31 seconds, the duplicate is allowed silently. The short window is intentional: it catches rapid accidental retries without burdening legitimate repeated experiments.

### 5.4 Dirty Git Repository

**Problem:** The user submits code with uncommitted changes, then later cannot reproduce the result.

**Behavior:**
- `slurp submit` detects a dirty git repo via `git status --porcelain`.
- A warning is printed: `Warning: Git workspace has uncommitted changes. Use --snapshot if reproducibility matters.`
- Submission proceeds. This is not a failure mode, but a diagnostic.

### 5.5 Log File Race Condition

**Problem:** `slurp logs --follow` starts before SLURM creates the log file.

**Behavior:**
- The `tail` command is deferred until the file exists. The CLI polls for file existence every 2 seconds (up to a 60-second timeout).
- If the file does not appear within 60 seconds (e.g. job stuck in `PENDING` due to resource exhaustion), the command prints `Timed out waiting for log file. Job is still PENDING.` and exits with code `1`.

### 5.6 Control Master Respawn Storm

**Problem:** The login node is under maintenance. Every `slurp` command triggers a control master respawn, potentially creating many background `ssh -MNf` processes.

**Behavior:**
- The respawn logic holds a file lock on `~/.local/share/slurp/.control_master.lock` while spawning.
- Only one respawn attempt per profile runs concurrently. Subsequent commands wait for the lock and reuse the new socket if successful.
- If respawn fails, all waiting commands receive the same `SSHError` after their respective retry budgets.
