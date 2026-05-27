# 09 — Error Handling and Exception Hierarchy

## 1. Design Philosophy

Every failure in `slurp` surfaces as an exception that carries three fields:

- **message** — what failed, in plain English, without stack-trace noise
- **hint** — the most likely fix, based on the error context and profile configuration
- **retryable** — whether repeating the same command might succeed without user intervention

The CLI catches all exceptions in `cli/main.py`, renders them as a Rich error panel, and exits with a deterministic code. The Python API raises the same classes, so notebook and script users get identical diagnostics.

**Rule:** If an error is retryable, `slurp` retries it internally up to a configured limit before surfacing it. If it is not retryable, it fails fast with a precise hint.

---

## 2. Exception Class Hierarchy

```python
class SlurpError(Exception):
    """Base exception for all slurp failures.

    Attributes:
        message (str): Human-readable description of what failed.
        hint (str): Actionable suggestion for the user.
        retryable (bool): Whether retrying the same operation may succeed.
        exit_code (int): The CLI exit code that corresponds to this error.
    """

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        retryable: bool = False,
        exit_code: int = 1,
    ) -> None:
        self.message = message
        self.hint = hint
        self.retryable = retryable
        self.exit_code = exit_code
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if self.retryable:
            parts.append("This error may resolve on retry.")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"hint={self.hint!r}, "
            f"retryable={self.retryable}, "
            f"exit_code={self.exit_code}"
            f")"
        )


class SSHError(SlurpError):
    """Failure in the SSH transport layer (control master, asyncssh, or network)."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        retryable: bool = True,
        exit_code: int = 10,
        profile: str | None = None,
    ) -> None:
        super().__init__(
            message,
            hint=hint or self._default_hint(profile),
            retryable=retryable,
            exit_code=exit_code,
        )

    @staticmethod
    def _default_hint(profile: str | None) -> str:
        base = "Check your network connection and VPN status."
        if profile:
            base += f" Verify the profile '{profile}' hostname is reachable."
        return base


class SlurmError(SlurpError):
    """Failure returned by a SLURM binary (sbatch, squeue, sacct, scancel)."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        retryable: bool = False,
        exit_code: int = 1,
        slurm_command: str | None = None,
        stderr_fragment: str | None = None,
    ) -> None:
        super().__init__(
            message,
            hint=hint or self._parse_hint(stderr_fragment, slurm_command),
            retryable=retryable,
            exit_code=exit_code,
        )
        self.slurm_command = slurm_command
        self.stderr_fragment = stderr_fragment

    @staticmethod
    def _parse_hint(stderr: str | None, command: str | None) -> str:
        if not stderr:
            return "Check SLURM daemon status with 'sinfo' on the login node."
        stderr = stderr.lower()
        if "invalid account" in stderr:
            return "Verify the account name in ~/.config/slurp/profiles.toml or pass --account."
        if "invalid partition" in stderr:
            return "Verify the partition name. Run 'sinfo' to list available partitions."
        if "qos" in stderr:
            return "Check your QOS limits with 'sacctmgr show qos format=name,maxwall,maxtres'."
        if "bank limit" in stderr or "association" in stderr:
            return "Your account may have exceeded its allocation. Contact cluster support."
        return f"Review the stderr from '{command}' and correct the requested resources."


class JobFailedError(SlurmError):
    """A job reached a terminal state other than COMPLETED (FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY)."""

    def __init__(
        self,
        message: str,
        *,
        job_id: str,
        status: str,
        exit_code: int | None = None,
        slurm_reason: str | None = None,
        max_rss_mb: float | None = None,
        hint: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            hint=hint or self._derive_hint(status, slurm_reason, max_rss_mb),
            retryable=retryable,
            exit_code=exit_code or 1,
        )
        self.job_id = job_id
        self.status = status
        self.slurm_reason = slurm_reason
        self.max_rss_mb = max_rss_mb

    @staticmethod
    def _derive_hint(status: str, reason: str | None, max_rss: float | None) -> str:
        if status == "TIMEOUT":
            return "Increase --time or implement checkpointing in your script."
        if status == "OUT_OF_MEMORY" or (reason and "oom" in reason.lower()):
            return f"Increase --mem (requested {max_rss:.0f} MB max RSS observed)."
        if status == "CANCELLED":
            return "Job was cancelled by user or administrator. Review scancel history."
        if status == "NODE_FAIL":
            return "A compute node failed. Resubmit; SLURM will schedule on healthy nodes."
        return "Review job logs with 'slurp logs <job_id>' to diagnose the failure."


class SyncError(SlurpError):
    """Failure during rsync code synchronization."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        retryable: bool = True,
        exit_code: int = 3,
        rsync_exit_code: int | None = None,
    ) -> None:
        super().__init__(
            message,
            hint=hint or self._rsync_hint(rsync_exit_code),
            retryable=retryable,
            exit_code=exit_code,
        )
        self.rsync_exit_code = rsync_exit_code

    @staticmethod
    def _rsync_hint(code: int | None) -> str:
        if code == 11:
            return "rsync: File system error on remote. Check disk quota with 'quota'."
        if code == 12:
            return "rsync: Protocol error. Ensure rsync is installed on the login node."
        if code == 30:
            return "rsync: Timeout. The connection may be unstable; retry."
        return "Check local path exists and remote working directory is writable."


class ConfigError(SlurpError):
    """Failure in profile loading, parsing, or validation."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        retryable: bool = False,
        exit_code: int = 8,
        config_path: str | None = None,
    ) -> None:
        super().__init__(
            message,
            hint=hint or (f"Check {config_path}" if config_path else ""),
            retryable=retryable,
            exit_code=exit_code,
        )


class ArgumentError(SlurpError):
    """Failure in CLI argument parsing or validation."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        retryable: bool = False,
        exit_code: int = 2,
    ) -> None:
        super().__init__(message, hint=hint, retryable=retryable, exit_code=exit_code)
```

---

## 3. Error Message Format (CLI Rendering)

When an exception propagates to the CLI boundary, `cli/main.py` renders it as:

```
┌─────────────────────────────────────────────────────────────┐
│  Error: SSHError                                              │
│  Message: Connection to jrlogin timed out after 30s           │
│  Hint: Check your network connection and VPN status.          │
│        Verify the profile 'jureca' hostname is reachable.     │
│  Retryable: Yes                                               │
│  Exit code: 10                                                │
└─────────────────────────────────────────────────────────────┘
```

**Rules:**
- If `--quiet` is set, only the message line is printed to stderr.
- If `--json` is set, the exception is serialized as:
  ```json
  {"error": "SSHError", "message": "...", "hint": "...", "retryable": true, "exit_code": 10}
  ```
- If `SLURP_DEBUG=1` is set, the full Python traceback is appended after the panel.

---

## 4. Retry Logic

### 4.1 Retryable Operations

The following operations are retried with exponential backoff:

| Operation | Base Delay | Max Delay | Max Attempts | Exception |
|-----------|------------|-----------|--------------|-----------|
| SSH command (control master alive) | 1s | 30s | 5 | `SSHError` |
| `sbatch` submission | 2s | 30s | 3 | `SlurmError` |
| `sacct` / `squeue` poll | 1s | 10s | 3 | `SlurmError` |
| `rsync` | 2s | 30s | 3 | `SyncError` |
| Log tail during `--follow` | 1s | 8s | ∞ (until user quits) | `SSHError` |

**Backoff formula:** `delay = min(base * 2^(attempt-1), max_delay)`

### 4.2 Non-Retryable Operations

The following fail immediately:

- Argument parsing errors (`ArgumentError`)
- Profile validation errors (`ConfigError`)
- Job-not-found errors (`SlurmError` with exit code 5)
- Permission-denied cancel errors (`SlurmError` with exit code 6)
- Terminal job failures (`JobFailedError`) — the job is already dead; retrying the cancel or status query is meaningless

### 4.3 Retry Decorator (Internal)

```python
from functools import wraps
import asyncio


def with_retry(max_attempts: int, base_delay: float, max_delay: float, on: type[Exception]):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except on as exc:
                    if not exc.retryable or attempt == max_attempts:
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    await asyncio.sleep(delay)
            raise RuntimeError("unreachable")
        return wrapper
    return decorator
```

---

## 5. Troubleshooting Matrix

| Symptom | Exception | Root Cause | Hint | Fix |
|---------|-----------|------------|------|-----|
| `slurp submit` hangs for 30s then fails | `SSHError` | VPN down or login node unreachable | Check network and VPN | Connect to VPN; verify `ssh jrlogin` works |
| `sbatch: error: Invalid account or account/partition combination` | `SlurmError` | Wrong account or partition in profile | Verify account name in profiles.toml | Run `sacctmgr show associations` on login node; update profile |
| `sbatch: error: QOSMinGRES` | `SlurmError` | Requested GPUs exceed QOS limit | Check QOS limits with sacctmgr | Reduce `--gpus` or request higher QOS from admins |
| Job status shows `OUT_OF_MEMORY` | `JobFailedError` | RAM request too low | Increase --mem | Resubmit with `--mem 128G` or larger |
| Job status shows `TIMEOUT` | `JobFailedError` | Wall time exceeded | Increase --time or add checkpointing | Resubmit with `--time 4:00:00`; save checkpoints every epoch |
| `slurp logs` shows no file | `SlurmError` (code 5) | Job not yet started or log path mismatch | Wait for job to start | Job is PENDING; logs appear once RUNNING |
| `rsync: write failed on ... (No space left on device)` | `SyncError` (code 11) | Remote disk quota exceeded | Check disk quota with 'quota' | Clean remote scratch; request quota increase |
| `slurp cancel` says "Permission denied" | `SlurmError` (code 6) | Cancelling another user's job | Cancel only your own jobs | Verify job ID with `slurp list` |
| `slurp watch` shows stale data | `SSHError` (code 10) | Control master died during watch | Reconnecting... | Press `r` to force reconnect; or restart `slurp watch` |
| `slurm_pmi_init failed` in job stderr | `JobFailedError` (code 1) | Wrong MPI mode for multi-node | Verify mpi_mode in profile | Set `mpi_mode = "pmi2"` in profile for JURECA |
| `module: command not found` in job stderr | `JobFailedError` (code 1) | Missing profile prologue or wrong shell | Check profile prologue | Ensure `prologue` includes `module load` lines for target cluster |
| `slurp submit` warns about duplicate job | `SlurpError` (warning only) | Identical spec submitted within 30s | Job 12345 with identical spec submitted 5s ago | Press `n` to avoid duplicate; press `y` if previous submit was lost |
| `sacct` returns empty for recent job | `SlurmError` (code 5) | SLURM database lag or job purged | Query with `sacct -S now-1hour` | Wait 10s for DB sync; or use `squeue` for active jobs |
| `asyncssh.connect() refused` | `SSHError` (code 10) | Control master socket missing | SSH control master not running | Run `ssh -MNf -S ~/.ssh/cm-%r@%h:%p jrlogin` manually |

---

## 6. Error Code Mapping

| Exit Code | Exception Class | Typical Trigger |
|-----------|-----------------|-----------------|
| 0 | — | Success |
| 1 | `SlurmError`, `JobFailedError` | SLURM binary error or job terminated abnormally |
| 2 | `ArgumentError` | Invalid CLI arguments or missing required flags |
| 3 | `SyncError` | `rsync` failed (disk full, network timeout, path missing) |
| 4 | `ArgumentError` | Invalid array parameters (mismatched lengths, empty list) |
| 5 | `SlurmError` | Job ID not found in local store or SLURM |
| 6 | `SlurmError` | Permission denied (cancel other user's job) |
| 7 | `SlurmError` | Pull attempted on non-terminal job |
| 8 | `ConfigError` | Profile missing, malformed TOML, or invalid field |
| 10 | `SSHError` | SSH control master dead, network unreachable, or host key mismatch |
| 124 | `SlurmError` | `slurp run` timeout (`--timeout` exceeded) |
| 130 | — | User interrupt (Ctrl+C); mapped from `KeyboardInterrupt` |

---

## 7. Examples by Exception Type

### 7.1 SSHError

```python
# Python API
import slurp

try:
    job = slurp.submit("python train.py", gpus=4)
except slurp.SSHError as exc:
    print(exc.message)   # "Connection to jrlogin timed out after 30s"
    print(exc.hint)      # "Check your network connection and VPN status..."
    print(exc.retryable) # True
```

**CLI output:**
```bash
$ slurp submit python train.py --gpus 4
Error: SSHError
Message: Connection to jrlogin timed out after 30s
Hint: Check your network connection and VPN status. Verify the profile 'jureca' hostname is reachable.
Retryable: Yes
Exit code: 10
```

### 7.2 SlurmError

```python
# Python API
try:
    job = slurp.submit("python train.py", gpus=4, account="wrong_account")
except slurm.SlurmError as exc:
    print(exc.message)         # "sbatch: error: Batch job submission failed..."
    print(exc.stderr_fragment) # "sbatch: error: Invalid account or account/partition..."
    print(exc.hint)            # "Verify the account name in ~/.config/slurp/profiles.toml..."
    print(exc.retryable)       # False
```

**CLI output:**
```bash
$ slurp submit --account wrong_account python train.py
Error: SlurmError
Message: sbatch failed with exit code 1
Hint: Verify the account name in ~/.config/slurp/profiles.toml or pass --account.
SLURM stderr: sbatch: error: Invalid account or account/partition combination
Retryable: No
Exit code: 1
```

### 7.3 JobFailedError

```python
# Python API
job = slurp.submit("python train.py", gpus=4, time="0:05:00")
result = job.wait(follow_logs=True)
# If job times out:
# JobFailedError: Job 12345 reached TIMEOUT
# Hint: Increase --time or implement checkpointing in your script.
```

**CLI output (from `slurp run`):**
```bash
$ slurp run --time 0:05:00 python train.py
... training output ...
Error: JobFailedError
Message: Job 12345 reached terminal state TIMEOUT (wall time 05:00 exceeded)
Hint: Increase --time or implement checkpointing in your script.
Status: TIMEOUT
Max RSS: 8192 MB
Retryable: No
Exit code: 1
```

### 7.4 SyncError

```python
# Python API
try:
    slurp.submit("python train.py", gpus=4)
except slurp.SyncError as exc:
    print(exc.message)          # "rsync exited with code 11"
    print(exc.rsync_exit_code)  # 11
    print(exc.hint)             # "rsync: File system error on remote..."
```

**CLI output:**
```bash
$ slurp submit python train.py
Error: SyncError
Message: rsync exited with code 11
Hint: rsync: File system error on remote. Check disk quota with 'quota'.
Rsync stderr: rsync: write failed on "/p/project1/..." (No space left on device)
Retryable: Yes
Exit code: 3
```

---

## 8. Failure Mode Considerations

### 8.1 Ambiguous Submit State

If the user's laptop loses network during `sbatch`, the client cannot know whether the job was accepted. `slurp` handles this by:

1. Recording a "pending submit" token in `jobs.json` before `sbatch`
2. On reconnect, querying `sacct -S now-5minutes` for jobs matching the command hash
3. If found: returning the existing `Job` object
4. If not found: offering re-submission with the duplicate guard

### 8.2 Concurrent Submit Corruption

Two simultaneous `slurp submit` processes could corrupt `jobs.json`. Prevention:

- File locking (`fcntl.flock`) around read-modify-write cycles
- Atomic writes (temp file + `os.rename()`)
- Lock held for < 10 ms; never across the SSH round-trip

### 8.3 Log File Rotation

SLURM does not rotate logs mid-job, but a very long-running job could produce multi-gigabyte stdout. `slurp logs` caps internal buffering at 1 MB and streams line-by-line to keep memory usage bounded.

### 8.4 Control Master Death During Long Operations

If the control master dies while `slurp watch` or `slurp logs --follow` is running:

1. The next `tail` or `sacct` command detects the dead socket (`asyncssh` raises `ConnectionLost`)
2. `core/ssh.py` triggers respawn: `ssh -MNf <profile>`
3. The operation retries with backoff
4. Log streaming resumes from the persisted byte offset
5. If respawn fails after max attempts, `SSHError` is raised with hint to check VPN
