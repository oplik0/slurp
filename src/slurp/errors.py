"""Exception hierarchy for slurp."""

from __future__ import annotations


class SlurpError(Exception):
    """Base exception with .message, .hint, .retryable."""

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.retryable = retryable

    def __str__(self) -> str:
        parts = [self.message]
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        if self.retryable:
            parts.append("This error may resolve if you retry the command.")
        return "\n".join(parts)


class SSHError(SlurpError):
    """Control master or asyncssh transport failure."""

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        retryable: bool = False,
        stderr_fragment: str | None = None,
    ):
        super().__init__(message, hint=hint, retryable=retryable)
        self.stderr_fragment = stderr_fragment


class SlurmError(SlurpError):
    """SLURM binary returned non-zero or malformed output."""

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        retryable: bool = False,
        stderr_fragment: str | None = None,
    ):
        super().__init__(message, hint=hint, retryable=retryable)
        self.stderr_fragment = stderr_fragment


class JobFailedError(SlurmError):
    """Job reached a terminal state with non-zero exit code."""

    def __init__(
        self,
        job_id: str,
        exit_code: int | None,
        status: str,
        *,
        stdout_tail: str = "",
        stderr_tail: str = "",
        hint: str | None = None,
    ):
        super().__init__(
            f"Job {job_id} failed with status {status} (exit code: {exit_code})",
            hint=hint or "Check the job logs with: slurp logs <job_id>",
            retryable=False,
        )
        self.job_id = job_id
        self.exit_code = exit_code
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail


class SyncError(SlurpError):
    """rsync or file transfer failure."""

    pass


class ProfileError(SlurpError):
    """Profile missing or invalid."""

    pass


class ConfigError(SlurpError):
    """Config directory not writable or corrupt."""

    pass


class IdempotencyError(SlurpError):
    """User declined duplicate submission."""

    pass
