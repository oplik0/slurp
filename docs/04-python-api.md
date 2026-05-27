# Python API Specification

The Python API is a secondary surface — the CLI is primary — but it is fully functional and idiomatic. It is designed for scripts, Jupyter notebooks, and programmatic orchestration. The guiding principle is explicitness over convenience: one call to submit a job, then explicit calls to wait, stream, or cancel.

## No `slurp.run()` in Python

There is no `slurp.run()` function in the Python API. The CLI verb `slurp run` is sugar for `submit` + `wait(follow_logs=True)`, but in Python that sugar is noise. The model is: `submit()` returns a `Job`, then you call `wait()` or `result()` on it. This teaches the user the object lifecycle once and keeps the namespace small.

---

## `slurp.submit()`

```python
import slurp

job = slurp.submit(
    command: str,
    *,
    profile: str | None = None,
    experiment: str | None = None,
    name: str | None = None,
    gpus: int | None = None,
    nodes: int = 1,
    cpus: int | None = None,
    mem: str | None = None,
    time: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    constraint: str | None = None,
    qos: str | None = None,
    depends_on: list[Job] | None = None,
    depends_on_type: str = "afterok",
    slurm_kwargs: dict[str, str] | None = None,
    working_dir: str | None = None,
    sync: bool = True,
    snapshot: bool = False,
) -> Job
```

**Return type:** `Job` — a frozen handle with the SLURM `job_id`, initial status `PENDING`, and metadata. The handle is not a live object; it must be `refresh()`-ed to observe state changes.

**Parameters:**

- `command` — The user command string. If the string names an existing file with `#!/bin/bash` and `#SBATCH` lines, slurp submits it directly. Otherwise, it generates a temporary wrapper script.
- `profile` — Profile name from `~/.config/slurp/profiles.toml`. If omitted, the default profile is used (the first one, or the one named `default`).
- `experiment` — Optional tag for grouping. Equivalent to `--experiment` on the CLI.
- `name` — Job name. Defaults to a slug derived from the command.
- `gpus`, `nodes`, `cpus`, `mem`, `time`, `partition`, `account`, `constraint`, `qos` — Map directly to SBATCH directives. `gpus` is translated to `--gres=gpu:N` on JURECA and `--gpus=N` on other clusters.
- `depends_on` — List of `Job` objects. A dependency chain is submitted to SLURM via `--dependency=<type>:<job_id>,...`.
- `depends_on_type` — Dependency type. Valid values: `afterok`, `afterany`, `after`, `afternotok`. Default is `afterok`.
- `slurm_kwargs` — Passthrough dictionary for any SBATCH directive not covered by the named parameters. Keys are directive names (without `--`), values are strings. Example: `{"exclude": "node05", "mail-type": "FAIL"}`.
- `working_dir` — Remote working directory. Defaults to the current working directory (after sync, if any).
- `sync` — Run `rsync` before submission. Default `True`. Set to `False` only if you manage synchronization manually.
- `snapshot` — If `True`, copy the working directory into `$PROJECT/.slurp/runs/<job_id>/` on the remote side before execution. Default `False`.

**Example — fire-and-forget:**
```python
job = slurp.submit("python train.py --lr 0.01", gpus=4, time="2:00:00")
print(job.job_id)  # 12345
```

**Example — passthrough directives:**
```python
job = slurp.submit(
    "python train.py",
    gpus=4,
    slurm_kwargs={"constraint": "a100", "exclude": "node05", "mail-type": "ALL"}
)
```

**Example — dependency chain:**
```python
pre = slurp.submit("python preprocess.py", gpus=0, time="1:00:00")
train = slurp.submit("python train.py", gpus=4, depends_on=[pre])
```

**Failure modes:**
- `SSHError` — Control master unreachable, profile misconfigured, or jump host down.
- `SlurmError` — `sbatch` rejected the script (invalid partition, account lacks quota, malformed directive).
- `ValueError` — `depends_on` contains a `Job` without a `job_id` (not yet submitted) or `depends_on_type` is invalid.

---

## `slurp.submit_array()`

```python
array = slurp.submit_array(
    template: str,
    *,
    configs: list[dict],
    throttle: int = 20,
    # ... all parameters from submit() except command
) -> ArrayJob
```

`submit_array` generates a native SLURM job array (`--array=0-N%M`) and a wrapper script that maps `SLURM_ARRAY_TASK_ID` to the values in `configs`. The `template` string uses Python brace-style formatting: `{seed}`, `{lr}`, etc.

**Parameters:**

- `template` — Command template with `{key}` placeholders.
- `configs` — List of dictionaries. Each dictionary maps placeholder names to string values. The length of the list determines the array size.
- `throttle` — Maximum concurrent tasks (`%M` in `--array`). Default `20`. Set to `0` for unlimited.

**Return type:** `ArrayJob` — a handle with `array_job_id`, `task_count`, and `throttle`.

**Example — explicit template:**
```python
array = slurp.submit_array(
    "python train.py --seed {seed} --lr {lr}",
    configs=[{"seed": "1", "lr": "0.01"}, {"seed": "2", "lr": "0.001"}],
    gpus=4,
    time="2:00:00",
)
print(array.array_job_id)  # 12345
```

**Example — CLI-style shorthand:**
```python
# This is what the CLI generates internally when you pass --seed 1,2,3,4,5
array = slurp.submit_array(
    "python train.py --seed {seed}",
    configs=[{"seed": str(i)} for i in range(1, 6)],
    gpus=4,
    throttle=10,
)
```

**Failure modes:**
- `ValueError` — A config dictionary is missing a key referenced in the template, or `configs` is empty.
- `SlurmError` — `sbatch` rejected the array (e.g., array size exceeds cluster limit).

---

## `Job` class

```python
@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    name: str
    status: JobStatus  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT
    profile: str
    experiment: Optional[str]
    submitted_at: datetime
```

Because the class is frozen and slotted, it is hashable and thread-safe. All state-changing methods return new objects rather than mutating in place.

### `Job.refresh() -> Job`

Query `sacct` for the current state of this job and return a new `Job` with updated `status`. The original object is unchanged.

```python
job = slurp.submit("python train.py", gpus=4)
# ... later ...
job = job.refresh()
print(job.status)  # RUNNING
```

**Failure mode:** `SSHError` if the query fails. The original `Job` is returned unchanged if the query fails and `raise_on_error=False` is passed (not the default).

### `Job.wait() -> JobResult`

Block until the job reaches a terminal state, then return a `JobResult`.

```python
def wait(
    self,
    *,
    timeout: str | float | None = None,
    follow_logs: bool = False,
    poll_interval: float = 5.0,
) -> JobResult
```

- `timeout` — Maximum time to wait. Accepts `"2h"`, `"30m"`, or seconds as a float. `None` means wait forever.
- `follow_logs` — If `True`, incrementally tail the job's `.out` and `.err` logs to the local terminal while waiting.
- `poll_interval` — Seconds between `sacct` polls. Minimum `1.0`.

The result is cached on the `Job` object (via a weakly-referenced internal cache). Subsequent calls to `result()` return the same `JobResult` instance.

```python
result = job.wait(follow_logs=True, timeout="2h")
print(result.exit_code)   # 0
print(result.stdout[:500])  # Capped at 1 MB
```

**Failure modes:**
- `TimeoutError` — Job did not reach a terminal state within `timeout`.
- `JobFailedError` — Job reached terminal state with non-zero exit code or `FAILED`/`TIMEOUT`/`CANCELLED` status.
- `SSHError` — Connection lost during wait. `wait()` does not auto-reconnect; the caller must retry.

### `Job.logs() -> Iterator[str]`

Yield lines from the job's `.out` and `.err` log files.

```python
def logs(
    self,
    *,
    follow: bool = False,
    tail: int = 100,
) -> Iterator[str]
```

- `follow` — If `True`, block and yield new lines as they appear (using `tail -F`). This is a single-job, blocking stream. The caller is responsible for interrupting it.
- `tail` — Number of lines to read from the end of each file. Default `100`. If `follow=True`, this is the initial window before blocking.

```python
for line in job.logs(follow=True, tail=10):
    print(line, end="")
```

**Failure modes:** `SSHError` if the log file is unreachable. `FileNotFoundError` if the log file does not exist (e.g., job was cancelled before output was written).

### `Job.cancel() -> Job`

Call `scancel` on this job and return a new `Job` with `status=CANCELLED`.

```python
job = job.cancel()
assert job.status == JobStatus.CANCELLED
```

**Failure mode:** `SlurmError` if `scancel` fails (e.g., job already completed). The original `Job` is not modified.

### `Job.result() -> JobResult`

Idempotent accessor for the job's result.

```python
result = job.result()
```

- If `wait()` has already been called, returns the cached `JobResult`.
- If `wait()` has not been called, calls `wait()` with default arguments.
- Calling `result()` twice always returns the same object: `job.result() is job.result()`.

This is safe to call in loops, error handlers, or Jupyter cells without worrying about duplicate polls or redundant log reads.

---

## `ArrayJob` class

```python
@dataclass(frozen=True, slots=True)
class ArrayJob:
    array_job_id: str       # Base ID, e.g. "12345"
    name: str
    profile: str
    experiment: Optional[str]
    submitted_at: datetime
    task_count: int
    throttle: int
```

### `ArrayJob.watch() -> None`

Display a live `rich.live.Live` table of all tasks. Blocks until the array completes or the user interrupts.

```python
array.watch()
```

### `ArrayJob.logs() -> Iterator[str]`

```python
def logs(
    self,
    *,
    task_id: int | None = None,
    follow: bool = False,
    tail: int = 100,
) -> Iterator[str]
```

- `task_id` — If `None`, yield lines from all tasks interleaved with a `[task_id]` prefix. If an integer, yield only that task's logs.

```python
# Stream all tasks
for line in array.logs(follow=True):
    print(line, end="")

# Stream just task 3
for line in array.logs(task_id=3, follow=True):
    print(line, end="")
```

### `ArrayJob.cancel() -> ArrayJob`

Cancel the entire array (`scancel <array_job_id>`).

### `ArrayJob.cancel_task(task_id: int) -> ArrayJob`

Cancel a single task (`scancel <array_job_id>_<task_id>`).

### `ArrayJob.results() -> list[JobResult]`

Block until all tasks reach terminal states, then return a list of `JobResult` objects ordered by `task_id`.

```python
results = array.results(timeout="4h", poll_interval=5.0)
for i, r in enumerate(results):
    print(f"Task {i}: exit={r.exit_code}")
```

**Failure mode:** `TimeoutError` if the array does not complete within `timeout`. Partial results are not returned; the caller should use `tasks()` to inspect individual states.

### `ArrayJob.tasks() -> list[Job]`

Return a list of `Job` handles, one per task. Each `Job` has a `job_id` of the form `<array_job_id>_<task_id>`.

```python
for task in array.tasks():
    print(task.job_id, task.status)
```

---

## `JobResult` class

```python
@dataclass(frozen=True, slots=True)
class JobResult:
    job_id: str
    status: JobStatus
    exit_code: Optional[int]
    stdout: str           # Capped at 1 MB
    stderr: str           # Capped at 1 MB
    max_rss_mb: Optional[float]
    wall_time: float      # seconds
```

**Caching behavior:** `JobResult` is cached internally on first production. The `Job` object holds a weak reference to the result. If the `Job` object is garbage collected, the cache entry is evicted. This prevents memory leaks in long-running scripts that submit many jobs.

**Idempotency:** `job.result()` always returns the same `JobResult` instance for the lifetime of the `Job` object. If you need a fresh read (e.g., because the log file was appended externally), create a new `Job` via `refresh()`.

---

## `Experiment` class

```python
exp = slurp.Experiment("exp_v1")
```

`Experiment` is a convenience wrapper that sets a default `experiment` tag on every job it submits. It does **not** manage state, orchestrate execution, enforce quotas, or provide a DAG scheduler. It is purely a namespace helper.

```python
exp = slurp.Experiment("exp_v1")

job1 = exp.submit("python preprocess.py", gpus=0)
job2 = exp.submit("python train.py", gpus=4, depends_on=[job1])

# These are equivalent to:
# job1 = slurp.submit("python preprocess.py", gpus=0, experiment="exp_v1")
# job2 = slurp.submit("python train.py", gpus=4, depends_on=[job1], experiment="exp_v1")
```

### What `Experiment` does NOT do

- It does not retry failed jobs.
- It does not enforce resource limits.
- It does not create a SLURM reservation.
- It does not start a background thread or daemon.
- It does not store state beyond the `experiment` string.

If you need orchestration, build it yourself on top of `Job` and `ArrayJob`.

---

## `AsyncClient` (deferred to v0.2)

An async-native API (`AsyncClient`) is planned for v0.2. It will expose the same surface as `slurp.submit()` but with `await` semantics and concurrent log streaming for multiple jobs. The sync API (`slurp.submit()`) will remain the default and will be implemented as a thin wrapper around `AsyncClient` using `asyncio.run()`.

In v0.1, if you need concurrency, submit jobs in a loop and then call `wait()` on each. `asyncio` is not required for basic usage, and `nest_asyncio` is never injected.

---

## Jupyter / IPython Integration

The sync API is designed to work inside Jupyter notebooks without event-loop conflicts.

- `slurp.submit()` returns immediately — no blocking.
- `job.wait()` blocks the cell, which is the expected behavior in a notebook.
- `job.logs(follow=True)` blocks the cell and streams output, analogous to `!tail -f`.
- Rich tables (`array.watch()`) render correctly in terminals and in Jupyter via `rich.jupyter`.

No special `%autoreload` or `nest_asyncio` setup is required. The internal event loop is scoped to the `submit()` / `wait()` call and does not pollute the global IPython loop.

**Caveat:** In a long-running notebook kernel, `Job` objects may accumulate in user namespace variables, holding references to `JobResult` caches. If you submit thousands of jobs from a single notebook, periodically delete old `Job` variables to allow garbage collection.

---

## Dependencies (`depends_on`, `depends_on_type`)

SLURM dependency chains are expressed as lists of `Job` objects and a type string.

```python
a = slurp.submit("python a.py", gpus=4)
b = slurp.submit("python b.py", gpus=4, depends_on=[a])           # afterok

c = slurp.submit("python c.py", gpus=4, depends_on=[a, b], depends_on_type="afterany")
```

**Valid `depends_on_type` values:**

| Type | SLURM meaning |
|------|---------------|
| `afterok` | Job starts only if all dependency jobs exit with code 0 (default) |
| `afterany` | Job starts when all dependency jobs finish, regardless of exit code |
| `after` | Job starts when all dependency jobs begin running |
| `afternotok` | Job starts only if all dependency jobs fail |

**Passthrough:** `slurm_kwargs` is merged last, so you can pass raw dependency strings if you need complex logic (`"dependency=afterok:12345+afterany:12346"`). However, `depends_on` and `depends_on_type` are preferred because they validate `job_id` existence.

**Failure mode:** If a dependency `Job` has no `job_id` (e.g., it was constructed locally but never submitted), `submit()` raises `ValueError` before any network call.

---

## `slurm_kwargs` Passthrough

`slurm_kwargs` is the escape hatch for directives not covered by named parameters.

```python
job = slurp.submit(
    "python train.py",
    gpus=4,
    slurm_kwargs={
        "constraint": "a100",
        "exclude": "node05,node06",
        "mail-type": "FAIL,TIME_LIMIT",
        "mail-user": "user@example.com",
        "begin": "now+1hour",
        "requeue": "",
    }
)
```

**Rules:**
- Keys are directive names without leading `--`.
- Values are strings. An empty string (`""`) means the directive is passed as a flag without a value (`--requeue`).
- `slurm_kwargs` override named parameters with the same SLURM directive. If you pass both `partition="dc-gpu"` and `slurm_kwargs={"partition": "booster"}`, the `slurm_kwargs` value wins.
- Validation is minimal: slurp passes the key-value pair directly to SBATCH. If the directive is invalid, `sbatch` will fail and `slurp` will surface the stderr.

**Failure mode:** If `slurm_kwargs` contains a key that conflicts with an internally-generated directive (e.g., `output` or `error`), the user value wins. This is intentional — it allows advanced users to override log paths. However, overriding log paths breaks `job.logs()` and `job.wait(follow_logs=True)`, which rely on deterministic paths.
