# 15 — Dependencies and Rationale

## 1. Dependency Policy

`slurp` follows a conservative dependency policy: **depend on libraries that solve a hard problem correctly, not on libraries that save ten lines of code.** Every direct dependency must justify its presence with a specific capability that is either non-trivial to replicate or already a community standard. Indirect dependencies are accepted only if they come from well-maintained packages with stable release cadences.

Python version baseline: **3.11+**. This enables `asyncio.TaskGroup`, `tomllib` (stdlib), `typing.Self`, and `str.removeprefix`, all of which reduce code volume and eliminate backport dependencies.

---

## 2. Core Dependencies

These are installed unconditionally with `pip install slurp`.

### `asyncssh >= 2.14`

**Role:** SSH transport for all remote command execution, file transfer coordination, and log streaming.

**Rationale:** `asyncssh` is the only mature, high-performance async SSH library for Python. It supports multiplexing hundreds of concurrent channels over a single connection, which is essential for `slurp watch` tailing 20+ job logs simultaneously. It also handles Ed25519 keys, jump hosts, and OpenSSH agent forwarding correctly — edge cases where pure-subprocess or Paramiko approaches break.

**Why not Paramiko?** Paramiko is single-threaded and GIL-bound. Benchmarks show ~15× lower throughput when multiplexing multiple channels. It is also sysadmin-oriented (interactive shell emulation) rather than library-oriented (structured command execution).

**Why not Fabric?** Fabric builds on Paramiko and inherits its performance limitations. It is designed for deployment scripts, not for long-lived multiplexed sessions.

**Why not pure `subprocess ssh`?** Raw subprocess SSH is brittle for programmatic use: unstructured stdout/stderr, manual pipe management, and no built-in retry or reconnect logic. It is used only for the one-shot control-master setup (`ssh -MNf`), not for the interactive streaming path.

**Version constraint `>= 2.14`:** 2.14 introduced improved connection pooling and fixes for edge cases with jump-host reconnect. Older versions occasionally deadlock when the control master socket is recreated.

### `pydantic >= 2.0`

**Role:** Data validation, serialization, and schema documentation for all domain objects (`Job`, `Profile`, `ResourceRequest`).

**Rationale:** Pydantic v2 provides runtime validation with minimal boilerplate and generates JSON Schema automatically. This is used for the job store (JSON serialization), profile parsing (TOML-to-model coercion), and CLI argument validation (rich error messages on invalid `--time` formats). Re-implementing validation manually would require ~500 lines of fragile regex and custom error formatting.

**Why not `attrs` + `cattrs`?** `attrs` is excellent for class definitions but lacks built-in validation and JSON Schema generation. The combination of `attrs`, `validators`, and `cattrs` would exceed Pydantic's dependency footprint while providing less functionality.

**Why not manual `dataclasses`?** Manual dataclasses require handwritten `__post_init__` validation, custom JSON encoders, and no schema documentation. The maintenance cost is higher than the Pydantic dependency cost.

**Version constraint `>= 2.0`:** Pydantic v2 is a full rewrite with a Rust core; it is significantly faster and has a cleaner API than v1. There is no reason to support v1.

### `typer >= 0.9`

**Role:** CLI framework: command routing, argument parsing, help generation, and shell completion.

**Rationale:** `typer` leverages Python type hints to generate CLI interfaces, eliminating the need for separate argparse definitions. It supports nested subcommands (`slurp config add-profile`), option groups, and automatic `--help` generation with Rich formatting. This keeps CLI code declarative and maintainable.

**Why not `click` directly?** `typer` is a thin, opinionated layer over `click`. It adds type-hint inference and better help-text defaults. The alternative is writing `click.option` decorators manually, which is more verbose and error-prone for complex multi-command apps.

**Why not `argparse`?** `argparse` from the standard library is sufficient for simple scripts but becomes unwieldy for nested subcommands (e.g., `slurp debug tunnel 12345`). The code-to-feature ratio is poor, and generated help text is plain-text only.

**Version constraint `>= 0.9`:** 0.9 introduced `typer.Typer` chaining and improved `Annotated` support, which `slurp` uses for resource-flag grouping.

### `rich >= 13.0`

**Role:** Terminal formatting: tables, progress bars, live dashboards, syntax highlighting, and structured error panels.

**Rationale:** `rich` is the de facto standard for modern terminal UIs in Python. It provides `rich.live.Live` for the `slurp watch` table, `rich.panel.Panel` for structured error output, and `rich.syntax.Syntax` for SBATCH script preview. Re-implementing ANSI escape sequences, terminal width detection, and live refresh logic would be a multi-week project.

**Why not `blessed` or `colorama`?** `blessed` is lower-level and requires manual layout math. `colorama` is Windows-only and does not provide tables or live refresh. Neither matches `rich`'s feature set.

**Why not a TUI framework (Textual, urwid)?** Full TUI frameworks add heavy dependencies (textual requires `markdown-it-py`, `pygments`, `linkify-it-py`) and consume the entire terminal. `slurp` needs inline tables and progress bars, not a full-screen application. Textual is deferred to v0.4+ for an optional dashboard mode.

**Version constraint `>= 13.0`:** 13.0 added `rich.table.Table.add_row` performance improvements and better async compatibility with `live.Live`.

### `structlog >= 24.0`

**Role:** Structured logging: JSON-formatted logs in production, pretty-printed logs in development.

**Rationale:** `structlog` separates log *event* generation from log *rendering*. In development, events are rendered as human-readable key=value lines. In production (or when piped to a file), they are rendered as JSON for ingestion by log aggregators. This dual-mode output is non-trivial to achieve with the standard `logging` module.

**Why not standard `logging`?** Python's `logging` module requires custom `Formatter` classes and `Filter` chains to achieve structured output. The configuration is verbose and error-prone. `structlog` provides the same capability with a single `configure()` call.

**Version constraint `>= 24.0`:** 24.0 stabilized the `typing` annotations and improved async context variable propagation.

### `questionary >= 2.0`

**Role:** Interactive CLI prompts for zero-config first-run profile setup.

**Rationale:** `questionary` provides validated input prompts, autocomplete, and confirmation dialogs with cross-platform terminal compatibility. It is used in exactly one flow: the first time a user runs `slurp submit` without a profile.

**Why not `rich.prompt`?** `rich` has basic prompt support (`Console.input`), but lacks validation, autocomplete, and select-box widgets. Re-implementing these on top of `rich` would approach the complexity of `questionary` itself.

**Why not `inquirer`?** `inquirer` is heavier and less maintained. It also has rendering issues on narrow terminals.

**Version constraint `>= 2.0`:** 2.0 dropped the `prompt_toolkit` version pin that caused conflicts with other CLI tools.

---

## 3. Optional Dependencies

### Web UI group: `pip install slurp[web]`

| Package | Version | Rationale |
|---------|---------|-----------|
| `fastapi >= 0.100` | HTTP API framework. Chosen over Flask for native async support (SSE endpoints) and automatic OpenAPI documentation. Starlette is too low-level for full app assembly. |
| `uvicorn >= 0.23` | ASGI server. Required to run FastAPI. Hypercorn is more complex; Daphne is Django-centric. |
| `jinja2 >= 3.0` | HTML template engine. Used only for the dashboard index page. FastAPI recommends it; no compelling alternative exists. |

**Exclusion rationale:** The web UI is v0.2. Including these packages in the core install would bloat the footprint for CLI-only users who will never run a local dashboard.

### Debug group: `pip install slurp[debug]`

| Package | Version | Rationale |
|---------|---------|-----------|
| `debugpy` | Remote Python debugger. Required for `slurp debug tunnel` (v0.2). No alternative exists for VSCode-compatible remote debugging. |

---

## 4. Development Dependencies

Installed only in development environments (`pip install slurp[dev]`).

| Package | Version | Rationale |
|---------|---------|-----------|
| `pytest` | Test runner and assertion library. Standard choice; no viable alternative in the Python ecosystem. |
| `pytest-asyncio` | Async test support for `core/ssh.py` and `core/slurm.py`. Required because the standard `pytest` does not handle `async def` test functions. |
| `mypy` | Static type checker. Enforces the type contracts between `domain.py`, `client.py`, and `core/`. Prevents runtime type errors in SSH and SLURM wrappers. |
| `ruff` | Fast Python linter and formatter (replaces `flake8`, `black`, `isort`). Selected for speed (~100× faster than `black` + `flake8`) and unified configuration. |
| `pre-commit` | Git hook manager. Runs `ruff` and `mypy` before every commit. Prevents style and type regressions from entering the codebase. |

**Docs group:** `pip install slurp[docs]`

| Package | Rationale |
|---------|-----------|
| `mkdocs-material` | Static site generator with Material Design theme. Chosen over Sphinx for simpler Markdown-native authoring and better built-in search. |
| `mkdocstrings[python]` | Auto-generated API documentation from docstrings. Integrates with `mkdocs-material` and supports Google-style docstrings. |

---

## 5. Dependency Graph

### Direct Dependencies

```
                    slurp
                      |
      +-------+-------+-------+-------+-------+
      |       |       |       |       |       |
  asyncssh pydantic  typer   rich  structlog questionary
      |       |       |       |       |       |
      |       |       |       |       |       |
   (ssh)   (data)  (cli)  (display) (logs)  (prompts)
```

### Full Graph (including optional groups)

```
slurp
├── asyncssh >= 2.14
│   ├── cryptography
│   └── typing_extensions (py<3.13)
├── pydantic >= 2.0
│   ├── annotated_types
│   └── pydantic_core (Rust extension)
├── typer >= 0.9
│   └── click >= 8.0
├── rich >= 13.0
│   ├── pygments (optional, for syntax highlighting)
│   └── markdown-it-py (optional, for markdown)
├── structlog >= 24.0
└── questionary >= 2.0
    └── prompt_toolkit >= 3.0
        └── wcwidth

[web]
├── fastapi >= 0.100
│   ├── starlette
│   └── pydantic (already core)
├── uvicorn >= 0.23
│   ├── click (already via typer)
│   └── h11
└── jinja2 >= 3.0
    └── MarkupSafe

[debug]
└── debugpy

[dev]
├── pytest
│   ├── pluggy
│   └── iniconfig
├── pytest-asyncio
├── mypy
│   ├── typing_extensions (already indirect)
│   └── mypy_extensions
├── ruff
└── pre-commit
    ├── pyyaml
    └── virtualenv

[docs]
├── mkdocs-material
│   ├── mkdocs
│   ├── pymdownx
│   └── material
└── mkdocstrings[python]
    └── mkdocstrings
```

**Size impact:**
- Core install: ~8 MB, ~15 transitive packages
- With `[web]`: ~22 MB, ~35 transitive packages
- With `[dev]`: ~45 MB, ~80 transitive packages (includes type stubs and test frameworks)

---

## 6. Rejected Dependencies

| Package | Why Rejected |
|---------|-------------|
| `paramiko` | GIL-bound, poor multiplexing performance. See `asyncssh` rationale above. |
| `fabric` | Builds on Paramiko; same performance issues. Adds deployment-centric abstractions irrelevant to slurp. |
| `textual` | Full TUI framework is overkill for inline tables. Heavy dependency tree. Deferred to v0.4. |
| `argenta` / `pytermgui` | Niche TUI frameworks with small communities and uncertain maintenance. |
| `sqlalchemy` / `sqlite3` | Local state is a simple JSON file. A relational database adds schema migration burden with no benefit for <1000 jobs. |
| `redis` | No distributed state machine in v0.1. SLURM is the source of truth. |
| `celery` / `dramatiq` | Job queuing is SLURM's job, not slurp's. Adding a task queue would introduce a second layer of scheduling. |
| `wandb` / `mlflow` | Experiment tracking is explicitly out of scope. The `experiment` tag is a string filter, not a metrics database. |
| `hydra` | Configuration hierarchy is deliberately flat (two layers). Hydra's composition and defaults list would violate the zero-surprise principle. |
| `nest_asyncio` | Jupyter compatibility is handled by managing the event loop internally, not by patching asyncio. `nest_asyncio` is a hack that masks underlying loop conflicts. |

---

## 7. Version Pinning and Upgrade Policy

- **Core dependencies:** Minimum versions only (`>=`). Patch releases of `asyncssh`, `pydantic`, and `typer` are expected to be backward-compatible. Upper bounds (`<`) are added only if a known breaking change is upstream (e.g., Pydantic v3 hypothetical).
- **Lock files:** The repository includes a `uv.lock` file for reproducible dev environments. End users installing from PyPI use the loose version constraints in `pyproject.toml`.
- **Security updates:** Dependabot or Renovate monitors `asyncssh` (cryptography backend) and `pydantic_core` (Rust extension) for CVEs. Critical security patches trigger a patch release of `slurp` with updated minimum versions.
