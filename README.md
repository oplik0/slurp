# slurp

[![CI](https://github.com/slurp/slurp/actions/workflows/test.yml/badge.svg)](https://github.com/slurp/slurp/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **A Python library and CLI for running ML jobs on SLURM clusters.**

`slurp` makes it frictionless to submit, monitor, and manage ML experiments on SLURM. It treats your local workstation as the control plane: you write code locally, run `slurp submit`, and your code is synced to the cluster, scheduled, and tracked. When the job finishes, pull results back with a single command.

## Philosophy

- **Local-first**: Keep your editor, debugging, and git workflow on your laptop. The cluster is just compute.
- **Zero-config first run**: `slurp submit` works even without a pre-written profile — it interactively asks for the connection details.
- **Idempotency**: Accidentally run the same command twice? `slurp` warns you instead of launching a duplicate job.
- **Pythonic API**: Use the CLI for quick experiments, or embed the Python API into your training pipelines for programmatic control.
- **Batteries included**: Multi-node PyTorch Distributed, job arrays, live log streaming, and snapshot isolation out of the box.

## Installation

Requires Python **3.11+**.

```bash
# With uv (recommended)
uv tool install slurp

# Or add to a project
uv add slurp

# Or with pip
pip install slurp
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/slurp/slurp.git
cd slurp
uv sync --extra dev
```

## Quick Start

### 1. Configure a profile

```bash
slurp config add-profile my-cluster \
  --hostname gpu-cluster.university.edu \
  --user $USER \
  --partition gpu \
  --account my-lab
```

Or let `slurp` ask you interactively:

```bash
slurp config add-profile my-cluster
```

### 2. Submit a job

```bash
slurp submit python train.py --lr 0.001 --epochs 100
```

`slurp` will:
1. Sync your local directory to the cluster.
2. Generate an `sbatch` script.
3. Submit the job.
4. Store the job ID locally so you can track it later.

### 3. Watch and pull

```bash
slurp watch                        # Live dashboard of all jobs
slurp logs 12345 --follow          # Tail logs
slurp pull 12345                   # Download outputs to ./outputs/12345/
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `slurp submit <command>` | Fire-and-forget job submission |
| `slurp run <command>` | Blocking submit with live log streaming |
| `slurp submit-array <template> --key v1,v2,...` | Submit a SLURM job array |
| `slurp status <job_id>` | Show current job status |
| `slurp list` | List tracked jobs |
| `slurp watch` | Live table of jobs |
| `slurp logs <job_id>` | Show stdout/stderr |
| `slurp cancel <job_id>` | Cancel a job |
| `slurp sync` | Sync code to remote without submitting |
| `slurp pull <job_id>` | Download results from remote |
| `slurp config add-profile <name>` | Add a cluster profile |
| `slurp config list-profiles` | Show configured profiles |
| `slurp config show-profile <name>` | Show a single profile |
| `slurp config edit-profile` | Open `profiles.toml` in `$EDITOR` |

### Common flags

- `--profile <name>` — Use a specific cluster profile.
- `--gpus <n>` — Request GPUs.
- `--nodes <n>` — Request multiple nodes (auto-inserts `torchrun`).
- `--time <HH:MM:SS>` — Wall-clock limit.
- `--partition <name>` — SLURM partition.
- `--account <name>` — SLURM account.
- `--experiment <tag>` — Tag the job for filtering.
- `--snapshot` — Snapshot the remote working directory before the job runs.
- `--dry-run` — Print the generated `sbatch` script without submitting.
- `--slurm-kwargs key=value` — Pass arbitrary `#SBATCH` directives.

## Python API Example

```python
from slurp import SyncClient

client = SyncClient(profile="my-cluster")

# Submit a single job
job = client.submit(
    "python train.py --lr 0.001",
    gpus=2,
    time="4:00:00",
    partition="gpu",
    experiment="transformer-v2",
)

# Wait for completion with live logs
result = job.wait(follow_logs=True)
print(result.stdout)

# Submit a hyper-parameter sweep as a job array
array = client.submit_array(
    "python train.py --seed {seed}",
    configs=[{"seed": str(s)} for s in range(5)],
    gpus=1,
    time="2:00:00",
)

# Wait for all tasks
results = array.results()
```

## Profile Configuration

Profiles are stored in `~/.config/slurp/profiles.toml`:

```toml
[profiles.default]
hostname = "gpu-cluster.university.edu"
username = "alice"
partition = "gpu"
account = "lab-123"
key_file = "~/.ssh/id_ed25519"

[profiles.default.sync]
local = "."
remote = "/home/alice/projects/my-project"

[profiles.jureca]
hostname = "jureca.fz-juelich.de"
username = "alice"
partition = "dc-gpu"
account = "my-project"
prologue = "module load Python\nmodule load CUDA"
mpi_mode = "pmi2"
gpu_flag_style = "gres"
```

### Profile fields

| Field | Description |
|-------|-------------|
| `hostname` | SSH target host |
| `username` | SSH username |
| `key_file` | SSH private key path |
| `proxy_jump` | Bastion host for multi-hop SSH |
| `partition` | Default SLURM partition |
| `account` | Default SLURM account |
| `prologue` | Shell commands injected before the user command |
| `mpi_mode` | MPI mode for multi-node (`pmi2`, `pmix`, etc.) |
| `cpu_bind` | CPU binding strategy (`cores`, `threads`, etc.) |
| `gpu_flag_style` | `gres` or `gpus` depending on cluster SLURM version |
| `sync.local` | Local path to sync from |
| `sync.remote` | Remote path to sync to |

## Development Setup

```bash
# Clone and enter the repo
git clone https://github.com/slurp/slurp.git
cd slurp

# Install dependencies with uv
uv sync --extra dev

# Run the test suite
uv run pytest

# Run linting
uv run ruff check src tests
uv run mypy src

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Project Structure

```
slurp/
├── src/slurp/
│   ├── cli/          # Typer CLI commands
│   ├── core/         # SSH, SLURM, sync, store
│   ├── domain.py     # Pydantic models
│   ├── client.py     # SyncClient public API
│   └── errors.py     # Exception hierarchy
├── tests/
│   ├── unit/         # Fast, isolated tests
│   ├── integration/  # Tests with Dockerized SLURM
│   └── e2e/          # Tests against real clusters
├── docs/             # Design documents and RFCs
└── templates/        # Example cluster profiles
```

## Contributing

Contributions are welcome! Please open an issue or pull request. Make sure to run `ruff` and `pytest` before submitting.

## License

MIT License — see [LICENSE](LICENSE) for details.
