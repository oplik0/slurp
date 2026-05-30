"""Security helpers for the slurp web UI.

Provides a single global stream token and CSRF token generation/validation.
"""

from __future__ import annotations

import secrets

# Global singleton stream token generated on first import.
STREAM_TOKEN: str = secrets.token_urlsafe(24)

# In-memory CSRF token store (single-user, local-only).
_csrf_tokens: dict[str, str] = {}


def generate_stream_token() -> str:
    """Generate a new cryptographically secure stream token.

    Returns:
        A URL-safe base64-encoded token.
    """
    return secrets.token_urlsafe(24)


def validate_stream_token(token: str | None) -> bool:
    """Validate the provided stream token against the global singleton.

    Args:
        token: The token supplied by the client, typically via query string.

    Returns:
        ``True`` if the token matches the global singleton, otherwise ``False``.
    """
    if token is None:
        return False
    return secrets.compare_digest(token, STREAM_TOKEN)


def generate_csrf_token(session_id: str) -> str:
    """Generate a CSRF token tied to a session identifier.

    Args:
        session_id: An opaque session identifier (e.g. the stream token).

    Returns:
        A new CSRF token string.
    """
    token = secrets.token_urlsafe(24)
    _csrf_tokens[session_id] = token
    return token


def validate_csrf_token(session_id: str | None, token: str | None) -> bool:
    """Validate a CSRF token for the given session.

    Args:
        session_id: The session identifier the token was issued for.
        token: The CSRF token provided by the client.

    Returns:
        ``True`` if the token is valid and matches the stored token.
    """
    if session_id is None or token is None:
        return False
    expected = _csrf_tokens.get(session_id)
    if expected is None:
        return False
    return secrets.compare_digest(token, expected)


def _reset_for_tests() -> None:
    """Clear the in-memory CSRF store (test helper only)."""
    _csrf_tokens.clear()
