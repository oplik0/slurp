"""FastAPI application factory for the slurp web UI.

Provides ``create_app()`` to build a configured FastAPI instance, and a
``__main__`` block so the server can be launched with
``python -m slurp.webui.app --port 8745``.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
except ImportError:
    # Defer the error until runtime so the CLI never crashes on import.
    FastAPI = None  # type: ignore[misc, assignment]
    StaticFiles = None  # type: ignore[misc, assignment]
    uvicorn = None  # type: ignore[assignment]

import structlog

from slurp.webui.routes import _STATIC_DIR, _TEMPLATES_DIR, router
from slurp.webui.security import STREAM_TOKEN

logger = structlog.get_logger()

# Default server settings
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8745


def _ensure_directories() -> None:
    """Create template and static directories if they do not exist."""
    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    _STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> Any:
    """Lifespan context manager: print the token URL on startup."""
    host = _DEFAULT_HOST
    port = _DEFAULT_PORT
    # Uvicorn config may override defaults; we print the canonical URL.
    url = f"http://{host}:{port}/?token={STREAM_TOKEN}"
    logger.info("slurp web UI started", url=url)
    print(f"\n  slurp web UI running at: {url}\n")
    yield
    logger.info("slurp web UI shutting down")


def create_app() -> FastAPI:
    """Build and return a configured FastAPI application.

    Raises:
        RuntimeError: If the optional web dependencies (fastapi, uvicorn,
            jinja2) are not installed.
    """
    if FastAPI is None:
        raise RuntimeError(
            "Web UI dependencies are missing. "
            "Install them with: pip install slurp[web]"
        )

    _ensure_directories()

    app = FastAPI(
        title="slurp",
        description="Web UI for managing SLURM jobs",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # Mount static files
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Include REST + SSE routes
    app.include_router(router)

    return app


def main() -> None:
    """CLI entry point for ``python -m slurp.webui.app``."""
    parser = argparse.ArgumentParser(description="slurp web UI server")
    parser.add_argument(
        "--host",
        default=_DEFAULT_HOST,
        help=f"Bind address (default: {_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_DEFAULT_PORT,
        help=f"Bind port (default: {_DEFAULT_PORT})",
    )
    args = parser.parse_args()

    if uvicorn is None:
        print(
            "Error: uvicorn is required to run the web UI.\n"
            "Install with: pip install slurp[web]",
            file=sys.stderr,
        )
        sys.exit(1)

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
