# 11 — Web UI Specification (v0.3)

The slurp web UI is an optional, first-party dashboard for monitoring SLURM jobs in real time. It is installed via `pip install slurp[web]` and runs as a local FastAPI server. The UI imports `slurp.client` directly — there is no `--json` passthrough, no subprocess shellout, and no external package dependency.

---

## 1. Design Philosophy

The web UI is a **presentation layer only**. It does not manage state, queue jobs, or run background workers. All business logic lives in `slurp.client.SyncClient`; the web layer is a thin HTTP wrapper that calls the same methods the CLI uses.

**Anti-features (explicitly excluded):**

- No local database (SQLite, PostgreSQL, etc.). Job state comes from `sacct` or the local JSON store, refreshed on every request.
- No background pollers or cron threads. Live updates are pushed via Server-Sent Events (SSE) from the browser's open connection.
- No workflow editor, drag-and-drop DAG builder, or visual pipeline designer.
- No user management, authentication providers, or role-based access control. The web UI is single-user and runs on `localhost`.

These exclusions keep the codebase small, the security surface minimal, and the mental model identical between CLI and web users.

---

## 2. Architecture

```
src/slurp/webui/
├── app.py              # FastAPI factory, lifespan events, middleware
├── routes.py           # REST endpoints and SSE stream
├── security.py         # Token generation, CSRF validation
├── sse.py              # JobStream: async generator for SSE events
├── templates/
│   └── index.html      # Single-page Jinja2 template
└── static/
    ├── dashboard.css   # ~200 lines, no external CSS framework
    └── dashboard.js    # ~400 lines, vanilla JS, no build step
```

### FastAPI + Uvicorn

The server is a standard Uvicorn process running on `127.0.0.1:8745` (default, overridable via `--port`). It is started with:

```bash
slurp webui
# or
python -m slurp.webui.app --port 8745
```

The optional-dependency design means the web UI code is only importable when `fastapi`, `uvicorn`, and `jinja2` are installed. If the user runs `slurp webui` without the extra, the CLI prints:

```
Error: Web UI dependencies not installed. Run: pip install slurp[web]
```

### SSE for Live Updates

There is **one** SSE endpoint: `GET /stream`. It returns a `text/event-stream` response that pushes JSON events as job states change. The browser opens this connection on page load and keeps it open.

```python
# routes.py — SSE endpoint
@router.get("/stream")
async def event_stream(request: Request, token: str = Query(...)):
    security.validate_stream_token(token)
    return EventSourceResponse(job_stream.listen())
```

```javascript
// dashboard.js — client-side SSE
const es = new EventSource(`/stream?token=${STREAM_TOKEN}`);
es.addEventListener("job_update", (e) => {
    const data = JSON.parse(e.data);
    updateJobRow(data.job_id, data.status, data.metrics);
});
es.addEventListener("log_append", (e) => {
    appendToLogPanel(e.data.job_id, e.data.lines);
});
```

**Event types:**

| Event | Payload | Frequency |
|-------|---------|-----------|
| `job_update` | `{"job_id": "12345", "status": "RUNNING", "metrics": {...}}` | Every poll cycle (~5s) or on status transition |
| `log_append` | `{"job_id": "12345", "lines": ["...", "..."]}` | When new log bytes are available |
| `heartbeat` | `{}` | Every 15s to keep connection alive behind proxies |

The server does not maintain a persistent connection to SLURM. On each SSE broadcast cycle, it calls `SyncClient.list_jobs()` and `SyncClient.logs()`, compares results to the previous cycle, and emits events only when deltas are detected. This keeps the server stateless.

---

## 3. Security Model

The web UI runs on localhost and is intended for single-user, local development. The security model is **minimal but sufficient** for that threat model.

### Random Token URL Prefix

On first launch, `security.py` generates a 32-byte random hex string and prints it to the console:

```
$ slurp webui
Web UI running at http://127.0.0.1:8745/?token=a3f7e2...
```

The token is required as a query parameter on every request (`/?token=...`, `/stream?token=...`, `/api/jobs?token=...`). Without it, the server returns `403 Forbidden`. This prevents casual cross-site attacks from other tabs or local applications.

**Implementation:**

```python
def generate_stream_token() -> str:
    return secrets.token_urlsafe(24)

def validate_stream_token(provided: str) -> None:
    if not secrets.compare_digest(provided, _stored_token):
        raise HTTPException(status_code=403, detail="Invalid token")
```

### CSRF Tokens for Mutations

Read-only operations (`GET /api/jobs`, `GET /stream`) use the URL token. Mutations (`POST /api/jobs/{id}/cancel`, `POST /api/sync`) require an additional `X-CSRF-Token` header fetched from `GET /api/csrf-token`.

```javascript
// dashboard.js — cancel flow
const csrf = await fetch(`/api/csrf-token?token=${TOKEN}`).then(r => r.json());
await fetch(`/api/jobs/12345/cancel?token=${TOKEN}`, {
    method: "POST",
    headers: {"X-CSRF-Token": csrf.token}
});
```

This protects against malicious HTML forms or scripts on other origins that might know the URL token but cannot read the CSRF token due to Same-Origin Policy.

### No HTTPS, No Auth

HTTPS and user authentication are explicitly out of scope for v0.3. If a team needs shared remote access, the recommended path is SSH port forwarding:

```bash
ssh -L 8745:localhost:8745 jrlogin
```

---

## 4. Dashboard Layout

The UI is a single-page application rendered from `templates/index.html`. It uses a lightweight CSS grid, no external JS framework, and no build step.

### Sections

1. **Header** — Profile selector, active experiment filter, refresh button.
2. **Job Table** — Sortable columns: ID, Name, Status, Partition, Nodes, GPUs, Time, Progress, ETA.
3. **Detail Panel** — Clicking a row expands inline details: resource request, log preview (last 50 lines), metrics sparkline.
4. **Log Panel** — Toggleable bottom drawer with live tail for the selected job.
5. **Actions** — Cancel, Sync, and (future) TensorBoard launch buttons.

### Status Badges

Status is color-coded with CSS classes derived from `JobStatus`:

| Status | Color | Indicator |
|--------|-------|-----------|
| PENDING | Amber | `●` pulsing |
| RUNNING | Green | `●` pulsing |
| COMPLETED | Blue | `✓` |
| FAILED | Red | `✗` |
| CANCELLED | Gray | `⊘` |
| TIMEOUT | Orange | `⏱` |

### Responsive Design

- **Desktop (>1024px):** Full three-pane layout (sidebar filter, main table, detail drawer).
- **Tablet (768–1024px):** Stack sidebar into a collapsible hamburger menu; table remains.
- **Mobile (<768px):** Single column; job table becomes a card list; detail panel is a full-screen overlay.

The responsive behavior is implemented with CSS Grid and `@media` queries — no JS breakpoint logic.

---

## 5. REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/?token=...` | Serve `index.html` |
| GET | `/stream?token=...` | SSE event stream |
| GET | `/api/csrf-token?token=...` | Fetch CSRF token for mutations |
| GET | `/api/jobs?token=...&experiment=...` | List jobs (reconciles with SLURM) |
| GET | `/api/jobs/{id}?token=...` | Single job detail |
| GET | `/api/jobs/{id}/logs?token=...&follow=false` | Fetch log text |
| POST | `/api/jobs/{id}/cancel?token=...` | Cancel job (requires CSRF) |
| POST | `/api/sync?token=...` | Trigger code sync (requires CSRF) |

All endpoints return JSON except `/stream` (SSE) and the root (`text/html`).

---

## 6. Failure Modes

### SSH Connection Drop During SSE Stream

If the control master dies while `/stream` is open, the next `SyncClient.list_jobs()` call raises `SSHError`. The SSE generator catches this, emits a `server_error` event with `{"hint": "SSH connection lost. Retry?"}`, and closes the stream. The browser reconnects automatically with exponential backoff (1s, 2s, 4s, max 30s).

### High-Frequency Polling Overload

If the user opens the web UI and the CLI `watch` simultaneously, both call `sacct` independently. There is no rate-limiting coordination in v0.3. The mitigation is: `sacct` is cheap (~50ms), and typical usage is one active client at a time. If needed, future versions can add a server-side TTL cache (e.g., 2s) for `sacct` results.

### Browser Tab Left Open Overnight

An idle tab keeps the SSE connection alive. To prevent unnecessary polling, the server stops calling `sacct` if no `job_update` listeners have been active for 60 seconds, and sends periodic `heartbeat` events instead. The browser reconnects on the next user interaction.

### Token Leak in Browser History

The URL token appears in browser history and server logs. This is acceptable for a localhost-only, single-user tool. If the token is leaked to another local process, that process can read job status but cannot cancel jobs without the CSRF token.

---

## 7. Deployment as Optional Extra

The web UI is distributed as a `pip` optional extra to keep the core package lightweight. Users who never need the web UI do not install FastAPI, Uvicorn, or Jinja2.

```toml
[project.optional-dependencies]
web = ["fastapi>=0.100", "uvicorn>=0.23", "jinja2>=3.0"]
```

The CLI entry point (`slurp webui`) checks for the extra at runtime and fails gracefully. The web module is never imported during normal CLI usage (`slurp submit`, `slurp watch`, etc.), avoiding import-time overhead.
