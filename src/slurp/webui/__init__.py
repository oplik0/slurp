"""Web UI for slurp — optional FastAPI backend."""

from __future__ import annotations

# Graceful degradation: if fastapi/uvicorn/jinja2 are missing, imports below
# will raise ImportError at runtime, but the package itself is importable.
# The CLI checks for these before starting the server.

try:
    from slurp.webui.app import create_app
    from slurp.webui.security import STREAM_TOKEN
except ImportError:
    create_app = None  # type: ignore[assignment]
    STREAM_TOKEN = None  # type: ignore[assignment]

__all__ = ["create_app", "STREAM_TOKEN"]
