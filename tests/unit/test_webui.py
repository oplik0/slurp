"""Unit tests for the slurp web UI.

These tests exercise FastAPI routes with mocked SyncClient and
verify security (token + CSRF) without network calls.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

# These tests need the optional `web` extra (fastapi/uvicorn). Skip the whole
# module gracefully when it isn't installed instead of failing at collection.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
from fastapi.testclient import TestClient  # noqa: E402

from slurp.domain import Job, JobStatus, ResourceRequest

# Import the app factory
from slurp.webui.app import create_app

# Security helpers must be importable even without fastapi
from slurp.webui.security import (
    STREAM_TOKEN,
    _reset_for_tests,
    generate_csrf_token,
    validate_csrf_token,
    validate_stream_token,
)


@pytest.fixture(autouse=True)
def reset_csrf_store() -> Generator[None, None, None]:
    """Clear the in-memory CSRF store between tests."""
    _reset_for_tests()
    yield


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Yield a TestClient for the web UI app with mocked SyncClient."""
    with patch("slurp.webui.routes.SyncClient") as mock_cls:
        instance = MagicMock()
        instance.profile.name = "default"
        instance.profile.partition = "gpu"
        instance.profile.account = "lab"
        instance.list_jobs.return_value = [
            Job(
                job_id="12345",
                name="train",
                status=JobStatus.RUNNING,
                profile="default",
                command="python train.py",
                resources=ResourceRequest(),
                working_dir="/home/test",
            ),
            Job(
                job_id="12346",
                name="eval",
                status=JobStatus.PENDING,
                profile="default",
                command="python eval.py",
                resources=ResourceRequest(),
                working_dir="/home/test",
            ),
        ]
        instance.status.side_effect = lambda job_id: (
            Job(
                job_id=job_id,
                name="train",
                status=JobStatus.RUNNING,
                profile="default",
                command="python train.py",
                resources=ResourceRequest(),
                working_dir="/home/test",
            )
            if job_id == "12345"
            else None
        )
        instance.refresh_job.return_value = Job(
            job_id="12345",
            name="train",
            status=JobStatus.RUNNING,
            profile="default",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/home/test",
        )
        instance.job_logs.return_value = iter(["epoch 1: loss 0.5\n"])
        instance.cancel_job.return_value = Job(
            job_id="12345",
            name="train",
            status=JobStatus.CANCELLED,
            profile="default",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/home/test",
        )
        mock_cls.return_value = instance
        app = create_app()
        with TestClient(app) as tc:
            yield tc


class TestSecurity:
    def test_stream_token_matches(self) -> None:
        assert validate_stream_token(STREAM_TOKEN) is True

    def test_stream_token_wrong(self) -> None:
        assert validate_stream_token("wrong-token") is False

    def test_stream_token_none(self) -> None:
        assert validate_stream_token(None) is False

    def test_csrf_lifecycle(self) -> None:
        session = STREAM_TOKEN
        csrf = generate_csrf_token(session)
        assert validate_csrf_token(session, csrf) is True
        assert validate_csrf_token("other-session", csrf) is False
        assert validate_csrf_token(session, "wrong") is False
        assert validate_csrf_token(None, csrf) is False


class TestIndex:
    def test_index_without_token(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 401

    def test_index_with_token(self, client: TestClient) -> None:
        response = client.get(f"/?token={STREAM_TOKEN}")
        assert response.status_code == 200
        assert "slurp" in response.text.lower()
        assert STREAM_TOKEN in response.text


class TestStreamSSE:
    def test_stream_without_token(self, client: TestClient) -> None:
        response = client.get("/stream")
        assert response.status_code == 401

    def test_stream_with_token(self, client: TestClient) -> None:
        """SSE stream should yield at least a heartbeat event."""
        with client.stream(
            "get", f"/stream?token={STREAM_TOKEN}&max_events=3"
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            # Read a few lines to verify SSE format
            text = ""
            for chunk in response.iter_text(chunk_size=256):
                text += chunk
                if "event:" in text:
                    break
            assert "event:" in text


class TestJobsAPI:
    def test_list_jobs_without_token(self, client: TestClient) -> None:
        response = client.get("/api/jobs")
        assert response.status_code == 401

    def test_list_jobs_with_token(self, client: TestClient) -> None:
        response = client.get(f"/api/jobs?token={STREAM_TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["job_id"] == "12345"

    def test_list_jobs_filter_experiment(self, client: TestClient) -> None:
        response = client.get(f"/api/jobs?token={STREAM_TOKEN}&experiment=sweep")
        assert response.status_code == 200

    def test_get_job_without_token(self, client: TestClient) -> None:
        response = client.get("/api/jobs/12345")
        assert response.status_code == 401

    def test_get_job_with_token(self, client: TestClient) -> None:
        response = client.get(f"/api/jobs/12345?token={STREAM_TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "12345"

    def test_get_job_not_found(self, client: TestClient) -> None:
        response = client.get(f"/api/jobs/99999?token={STREAM_TOKEN}")
        assert response.status_code == 404


class TestLogsAPI:
    def test_logs_without_token(self, client: TestClient) -> None:
        response = client.get("/api/jobs/12345/logs")
        assert response.status_code == 401

    def test_logs_with_token(self, client: TestClient) -> None:
        response = client.get(f"/api/jobs/12345/logs?token={STREAM_TOKEN}")
        assert response.status_code == 200
        assert "epoch 1" in response.text

    def test_logs_not_found(self, client: TestClient) -> None:
        response = client.get(f"/api/jobs/99999/logs?token={STREAM_TOKEN}")
        assert response.status_code == 404


class TestCsrfAPI:
    def test_csrf_token_without_token(self, client: TestClient) -> None:
        response = client.get("/api/csrf-token")
        assert response.status_code == 401

    def test_csrf_token_with_token(self, client: TestClient) -> None:
        response = client.get(f"/api/csrf-token?token={STREAM_TOKEN}")
        assert response.status_code == 200
        data = response.json()
        assert "csrf_token" in data
        assert data["csrf_token"]


class TestCancelAPI:
    def test_cancel_without_token(self, client: TestClient) -> None:
        response = client.post("/api/jobs/12345/cancel")
        assert response.status_code == 401

    def test_cancel_without_csrf(self, client: TestClient) -> None:
        response = client.post(f"/api/jobs/12345/cancel?token={STREAM_TOKEN}")
        assert response.status_code == 403

    def test_cancel_with_csrf(self, client: TestClient) -> None:
        csrf = generate_csrf_token(STREAM_TOKEN)
        response = client.post(
            f"/api/jobs/12345/cancel?token={STREAM_TOKEN}",
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Cancel requested."
        assert data["job"]["job_id"] == "12345"

    def test_cancel_not_found(self, client: TestClient) -> None:
        csrf = generate_csrf_token(STREAM_TOKEN)
        response = client.post(
            f"/api/jobs/99999/cancel?token={STREAM_TOKEN}",
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 404


class TestSyncAPI:
    def test_sync_without_token(self, client: TestClient) -> None:
        response = client.post("/api/sync")
        assert response.status_code == 401

    def test_sync_without_csrf(self, client: TestClient) -> None:
        response = client.post(f"/api/sync?token={STREAM_TOKEN}")
        assert response.status_code == 403

    def test_sync_with_csrf(self, client: TestClient) -> None:
        csrf = generate_csrf_token(STREAM_TOKEN)
        response = client.post(
            f"/api/sync?token={STREAM_TOKEN}",
            headers={"X-CSRF-Token": csrf},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Sync complete."
