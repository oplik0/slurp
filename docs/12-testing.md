# 12 — Testing Strategy Specification

This document defines the testing pyramid for slurp: unit tests (80%), integration tests (15%), and end-to-end smoke tests (5%). It covers fixtures, mocking strategy, CI/CD integration, and failure mode testing.

---

## 1. Philosophy

Tests must be **fast, deterministic, and hermetic**. A developer should be able to run the full unit suite in under 10 seconds without Docker, network access, or a SLURM cluster. Integration tests verify cross-module behavior with a fake SLURM environment. E2E tests validate the tool against a real cluster, but run only in scheduled CI or on demand.

---

## 2. Unit Tests (80%)

Unit tests cover all modules in `src/slurp/` except CLI argument parsing (which is tested implicitly through integration tests). Every test runs in isolation with mocked external dependencies.

### Mocking asyncssh

`asyncssh` is replaced with a `MockSSHConnection` class that implements the same interface (`run()`, `create_process()`, `close()`) but returns canned stdout/stderr strings.

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
def mock_ssh():
    conn = AsyncMock()
    conn.run = AsyncMock(return_value=AsyncMock(
        stdout="12345\n",
        stderr="",
        exit_status=0,
    ))
    return conn
```

Tests for `core/ssh.py` verify that:
- Control master setup calls `subprocess.run(["ssh", "-MNf", ...])` with correct arguments.
- Auto-reconnect retries up to 3 times with exponential backoff.
- A dead socket triggers respawn before the next command.

### Mocking SLURM Commands

A `FakeSlurm` fixture maps command strings to captured outputs. This allows testing `core/slurm.py` without executing real `sbatch` or `squeue`.

```python
@pytest.fixture
def fake_slurm():
    return {
        "sbatch": "Submitted batch job 12345\n",
        "squeue -u user -o '%i %T %j'": "12345 RUNNING train\n",
        "sacct -j 12345 -o JobID,State,ExitCode": "12345 COMPLETED 0:0\n",
        "scancel 12345": "",
    }
```

Tests assert:
- `sbatch` wrapper generates the correct directive lines.
- `squeue` parser handles missing jobs (empty output → `NOT_FOUND`).
- `sacct` parser handles array jobs (`12345_0`, `12345_1`).
- Exit codes like `0:0` and `1:0` are mapped correctly to `JobStatus`.

### tmp_path Fixtures

All filesystem tests use `pytest.tmp_path` to avoid polluting the real `~/.local/share/slurp/` directory.

```python
def test_store_atomic_write(tmp_path):
    store = JobStore(path=tmp_path / "jobs.json")
    store.write(Job(job_id="999", status="PENDING", ...))
    # Assert temp file was created and renamed
    assert (tmp_path / "jobs.json").exists()
    # Assert no partial writes remain
    assert len(list(tmp_path.glob("*.tmp"))) == 0
```

Key unit test modules:

| Module | Coverage Target | Key Behaviors Tested |
|--------|---------------|----------------------|
| `core/slurm.py` | 95% | SBATCH generation, status parsing, error handling |
| `core/ssh.py` | 90% | Control master lifecycle, auto-reconnect, multiplexing |
| `core/store.py` | 95% | Atomic writes, locking, reconciliation, corruption recovery |
| `core/sync.py` | 90% | Rsync command construction, `.gitignore` filtering, snapshot mode |
| `client.py` | 90% | `submit()`, `wait()`, `logs()`, idempotency, experiment grouping |
| `domain.py` | 100% | Pydantic validation, `Job.refresh()`, `JobResult` caching |
| `cli/*.py` | 80% | Command delegation, exit codes, error message formatting |

---

## 3. Integration Tests (15%)

Integration tests run inside a Docker container that provides a fake SLURM environment (`giovtorres/slurm-docker-cluster`) and an OpenSSH server. This validates the full path from CLI invocation to remote job execution.

### Docker Environment

```yaml
# tests/docker/docker-compose.yml
services:
  slurm:
    image: giovtorres/slurm-docker-cluster:latest
    hostname: slurm-test
    volumes:
      - ./test_keys:/test_keys:ro
    ports:
      - "2222:22"
```

The container exposes:
- A functional `slurmctld` and `slurmd`.
- A pre-created test user (`slurmuser`) with passwordless SSH key auth.
- A shared filesystem where job scripts and logs are written.

### Integration Test Cases

```python
@pytest.mark.integration
class TestSubmitToCancel:
    def test_submit_poll_cancel(self, slurm_container):
        # Uses the real SSH + SLURM stack inside Docker
        result = slurm_container.run("slurp submit -- /bin/hostname")
        job_id = parse_job_id(result.stdout)

        assert slurm_container.run(f"slurp status {job_id}").stdout == "RUNNING"

        slurm_container.run(f"slurp cancel {job_id}")
        assert slurm_container.run(f"slurp status {job_id}").stdout == "CANCELLED"
```

Key integration scenarios:

1. **Submit → Watch → Cancel** — Full lifecycle with log streaming.
2. **Array job** — `submit-array` with 5 tasks; verify all task IDs and log files.
3. **Sync + Submit** — Rsync code to container, submit script referencing synced files.
4. **Profile management** — `config add-profile`, submit with profile, verify SSH host.
5. **Reconnection** — Kill control master mid-stream; verify resume behavior.
6. **Concurrent submits** — Two parallel `slurp submit` calls; verify store locking.

Integration tests run in CI on every pull request but are skipped locally unless `SLURP_TEST_INTEGRATION=1` is set.

---

## 4. E2E / Smoke Tests (5%)

E2E tests run against the real JURECA cluster. They are marked with `@pytest.mark.e2e` and are **never** run in standard CI. They execute in a scheduled nightly job or on manual trigger.

### E2E Test Cases

```python
@pytest.mark.e2e
@pytest.mark.timeout(600)
def test_submit_gpu_job_jureca():
    job = slurp.submit("python -c 'import torch; print(torch.cuda.device_count())'", gpus=1)
    result = job.wait(timeout="5m")
    assert result.exit_code == 0
    assert "1" in result.stdout
```

```python
@pytest.mark.e2e
def test_cancel_all_by_experiment():
    exp = slurp.Experiment("smoke_test_cancel")
    j1 = exp.submit("sleep 300", gpus=0)
    j2 = exp.submit("sleep 300", gpus=0)
    time.sleep(5)  # let them reach RUNNING or PENDING
    exp.cancel_all()
    assert j1.refresh().status == "CANCELLED"
    assert j2.refresh().status == "CANCELLED"
```

### E2E Requirements

- A valid JURECA profile must be present in `~/.config/slurp/profiles.toml`.
- The test runner must be on a machine with SSH access to `jrlogin` (or use a jump host).
- Tests must not submit large or long-running jobs. Maximum wall time: 10 minutes.
- Tests must clean up after themselves (cancel jobs, remove experiment tags) even on failure.

### Failure Modes

If an E2E test fails, the output must include:
- The SLURM job ID.
- The `sacct` output for that job.
- The contents of `slurm-<name>-<id>.out` and `.err` (fetched via SSH).

This allows debugging without reproducing the failure manually.

---

## 5. Testing Helpers and Fixtures

### Mock Profiles

```python
@pytest.fixture
def mock_profile(tmp_path):
    return Profile(
        name="test",
        hostname="localhost",
        username="testuser",
        key_file=str(tmp_path / "id_ed25519"),
        partition="test-part",
        account="test-acct",
    )
```

### Mock SSH with Programmable Responses

```python
@pytest.fixture
def mock_ssh_conn():
    class FakeSSH:
        def __init__(self, responses: dict[str, str]):
            self._responses = responses
        async def run(self, cmd: str) -> FakeResult:
            for pattern, stdout in self._responses.items():
                if pattern in cmd:
                    return FakeResult(stdout=stdout, stderr="", exit_status=0)
            raise RuntimeError(f"Unexpected command: {cmd}")
    return FakeSSH
```

### Fake SLURM stdout Fixtures

Captured outputs from real JURECA commands are stored in `tests/fixtures/slurm_outputs/`:

- `squeue_running.txt`
- `sacct_completed.txt`
- `sacct_array.txt`
- `sbatch_success.txt`
- `sbatch_fail_qos.txt`

These fixtures ensure tests break if SLURM changes its output format, but do not require live cluster access.

### Store Isolation

```python
@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "slurp.core.store.DEFAULT_STORE_PATH",
        tmp_path / "slurp_jobs.json",
    )
```

---

## 6. CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: test
on: [push, pull_request]
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -vv --cov=slurp --cov-report=xml
      - uses: codecov/codecov-action@v3

  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker compose -f tests/docker/docker-compose.yml up -d
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -vv -m integration
      - run: docker compose -f tests/docker/docker-compose.yml down

  e2e:
    runs-on: self-hosted  # Jülich runner with VPN/SSH access
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[e2e]')
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/e2e/ -vv -m e2e --timeout=600
```

### Coverage Gates

- Unit test line coverage must remain ≥ 80%.
- Any PR that drops coverage by > 2% fails CI.
- Integration tests do not contribute to coverage gates (they exercise external systems).

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest-unit
        entry: pytest tests/unit/ -q
        language: system
        types: [python]
        pass_filenames: false
        always_run: true
```

---

## 7. Failure Mode Testing

Every risk identified in `14-risks.md` must have at least one automated test covering its failure path:

| Risk | Test |
|------|------|
| State corruption from concurrent writes | `test_store_concurrent_writes` spawns 10 processes, all appending jobs |
| SSH disconnect mid-submit | `test_reconnect_after_disconnect` kills socket after `sbatch` stdout is received |
| Duplicate job submit | `test_idempotency_warning` submits same spec twice within 30s |
| Dirty git repo | `test_submit_warns_dirty_git` mocks `git status` and asserts warning in stderr |
| Corrupted local store | `test_store_corrupted_json` writes invalid JSON, asserts graceful re-initialization |

These tests are unit tests (fast, no Docker) that simulate the failure condition through mocks or filesystem manipulation.
