# slurp

> A Python library and CLI for running ML jobs on SLURM clusters.
> **Simpler than `sbatch` for people who don't know `sbatch`.**

## Philosophy

`slurp` is designed for researchers who understand PyTorch but do not want to learn SLURM internals, SBATCH directives, or multi-node launcher incantations. It provides a zero-config first run, interactive profile learning, and a tiny CLI surface that covers 80% of daily cluster workflows.

## Installation

```bash
# Using uv (recommended)
uv pip install slurp

# Using pip
pip install slurp

# With web UI extras
pip install slurp[web]
```

## Quick Start

### 1. Configure your cluster profile

```bash
# Interactive first-run setup
slurp config add-profile jureca
# ? Hostname: jrlogin
# ? Username: alice
# ? Default partition: dc-gpu
# ? Default account: training2615
# ? Save profile? [Y/n]: Y
```

Profiles are stored in `~/.config/slurp/profiles.toml`.

### 2. Submit a job

```bash
# Fire-and-forget
slurp submit python train.py --lr 0.01 --gpus 4 --time 2:00:00

# Blocking run with live log streaming
slurp run python train.py --lr 0.01 --gpus 4

# Dry-run to preview the SBATCH script
slurp submit --dry-run python train.py --gpus 4
```

### 3. Monitor jobs

```bash
# Live watch table
slurp watch

# Job logs
slurp logs 12345
slurp logs 12345 --follow

# Status and listing
slurp status 12345
slurp list --experiment exp_v1
```

### 4. Manage jobs

```bash
# Cancel jobs
slurp cancel 12345

# Sync code without submitting
slurp sync

# Pull results
slurp pull 12345 --local ./outputs
```

## Python API

```python
import slurp

# Fire-and-forget
job = slurp.submit("python train.py --lr 0.01", gpus=4, time="2:00:00")
print(job.job_id)  # 12345

# Block until completion
result = job.wait(follow_logs=True)
print(result.exit_code, result.stdout)

# Job arrays
array = slurp.submit_array(
    "python train.py --seed {seed}",
    configs=[{"seed": str(i)} for i in range(5)],
    gpus=4,
)

# Experiment grouping
exp = slurp.Experiment("exp_v1")
job = exp.submit("python train.py", gpus=4)
exp.watch()
```

## Profile Configuration

Example `~/.config/slurp/profiles.toml`:

```toml
[profiles.jureca]
hostname = "jrlogin"
username = "alice"
key_file = "~/.ssh/id_ed25519"
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

[profiles.jureca.sync]
local = "/home/alice/projects"
remote = "/p/project1/training2615/alice/projects"
```

## Development

```bash
# Clone and setup
uv sync

# Run tests
uv run pytest tests/unit/ -v

# Lint and type check
uv run ruff check src/slurp/
uv run mypy src/slurp/

# Install pre-commit hooks
uv run pre-commit install
```

## Architecture

```
src/slurp/
├── __init__.py          # Public API
├── domain.py            # Pydantic models (Job, Profile, ResourceRequest)
├── client.py            # SyncClient (public API)
├── core/
│   ├── ssh.py           # asyncssh transport with auto-reconnect
│   ├── slurm.py         # SBATCH generation and SLURM wrappers
│   ├── sync.py          # rsync code sync
│   ├── launcher.py      # TorchrunLauncher for multi-node
│   └── store.py         # Atomic JSON job store
├── cli/
│   ├── main.py          # Typer entry point
│   ├── submit.py        # submit, run, submit-array
│   ├── watch.py         # Live watch table
│   ├── logs.py          # Log streaming
│   ├── status.py        # Status and list
│   ├── cancel.py        # Cancel jobs
│   ├── sync.py          # Sync code
│   ├── pull.py          # Pull results
│   └── config.py        # Profile management
└── helpers/
    └── debug.py         # debugpy helper
```

## License

MIT
