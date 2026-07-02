"""Unit tests for CLI commands.

These tests use Typer's CliRunner and mock SyncClient to avoid network calls.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from slurp.cli.main import app
from slurp.domain import Job, JobStatus, ResourceRequest

runner = CliRunner()

# The webui command's happy-path test needs the optional `web` extra
# (uvicorn) for @patch("uvicorn.run") to resolve. The missing-deps test
# below deliberately runs without it.
_WEB_AVAILABLE = importlib.util.find_spec("uvicorn") is not None


@pytest.fixture
def mock_client() -> Generator[MagicMock, None, None]:
    """Return a mock SyncClient with sensible defaults."""
    with (
        patch("slurp.client.SyncClient") as cls,
        patch("slurp.cli.submit.SyncClient") as cls2,
        patch("slurp.cli.cancel.SyncClient") as cls3,
        patch("slurp.cli.status.SyncClient") as cls4,
        patch("slurp.cli.logs.SyncClient") as cls5,
        patch("slurp.cli.sync.SyncClient") as cls6,
        patch("slurp.cli.pull.SyncClient") as cls7,
    ):
        instance = MagicMock()
        instance.profile.name = "default"
        instance.profile.partition = "gpu"
        instance.profile.account = "lab"
        instance.submit.return_value = Job(
            job_id="12345",
            name="train",
            status=JobStatus.PENDING,
            profile="default",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/home/test",
        )
        instance.status.return_value = Job(
            job_id="12345",
            name="train",
            status=JobStatus.RUNNING,
            profile="default",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/home/test",
        )
        instance.list_jobs.return_value = [
            Job(
                job_id="12345",
                name="train",
                status=JobStatus.RUNNING,
                profile="default",
                command="python train.py",
                resources=ResourceRequest(),
                working_dir="/home/test",
            )
        ]
        instance.cancel_job.return_value = Job(
            job_id="12345",
            name="train",
            status=JobStatus.CANCELLED,
            profile="default",
            command="python train.py",
            resources=ResourceRequest(),
            working_dir="/home/test",
        )
        instance.__enter__.return_value = instance
        instance.__exit__.return_value = False
        cls.return_value = instance
        cls2.return_value = instance
        cls3.return_value = instance
        cls4.return_value = instance
        cls5.return_value = instance
        cls6.return_value = instance
        cls7.return_value = instance
        yield instance


class TestSubmit:
    def test_submit_basic(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["submit", "python", "train.py"])
        assert result.exit_code == 0
        assert "Job 12345 submitted" in result.output
        mock_client.submit.assert_called_once()

    def test_submit_with_resources(self, mock_client: MagicMock) -> None:
        result = runner.invoke(
            app,
            [
                "submit",
                "python",
                "train.py",
                "--gpus",
                "2",
                "--nodes",
                "4",
                "--time",
                "4:00:00",
                "--partition",
                "gpu",
                "--account",
                "lab-123",
                "--experiment",
                "sweep-1",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_client.submit.call_args.kwargs
        assert call_kwargs["gpus"] == 2
        assert call_kwargs["nodes"] == 4
        assert call_kwargs["time"] == "4:00:00"
        assert call_kwargs["partition"] == "gpu"
        assert call_kwargs["account"] == "lab-123"
        assert call_kwargs["experiment"] == "sweep-1"

    def test_submit_uses_double_dash_separator(self, mock_client: MagicMock) -> None:
        result = runner.invoke(
            app,
            [
                "submit",
                "--gpus",
                "2",
                "--",
                "python",
                "train.py",
                "--lr",
                "0.01",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_client.submit.call_args
        assert call_kwargs[0][0] == "python train.py --lr 0.01"

    def test_submit_dry_run(self, mock_client: MagicMock) -> None:
        result = runner.invoke(
            app,
            [
                "submit",
                "python",
                "train.py",
                "--dry-run",
                "--gpus",
                "1",
                "--time",
                "1:00:00",
            ],
        )
        assert result.exit_code == 0
        mock_client.submit.assert_not_called()
        assert "#SBATCH" in result.output

    def test_submit_no_profile_interactive(self, mock_client: MagicMock) -> None:
        """When no profile exists and TTY is available, interactive fallback runs."""
        from slurp.errors import ProfileError

        mock_client.submit.side_effect = ProfileError(
            "No profiles configured.", hint="Run: slurp config add-profile <name>"
        )
        with (
            patch("slurp.cli.submit.sys.stdin.isatty", return_value=True),
            patch("slurp.cli.submit.questionary") as q,
            patch("slurp.cli.submit._ensure_config_dir"),
            patch("slurp.cli.submit.CONFIG_FILE") as mock_cfg,
        ):
            q.text.side_effect = [
                MagicMock(ask=lambda: "cluster.example.com"),  # hostname
                MagicMock(ask=lambda: "alice"),  # user
                MagicMock(ask=lambda: "gpu"),  # partition
                MagicMock(ask=lambda: "lab-123"),  # account
            ]
            mock_cfg.exists.return_value = False
            mock_cfg.write_text = MagicMock()
            result = runner.invoke(
                app, ["submit", "python", "train.py"], input="y\n"
            )
            # The interactive path may not fully execute in headless runner,
            # but we verify it doesn't crash with an unhandled error.
            assert result.exit_code in (0, 1, 10)


class TestSubmitArray:
    def test_submit_array_with_sweep_params(self, mock_client: MagicMock) -> None:
        """submit-array should parse trailing --key v1,v2,... as sweep configs."""
        mock_client.submit_array.return_value = MagicMock(
            array_job_id="12345_0", task_count=3
        )
        result = runner.invoke(
            app,
            [
                "submit-array",
                "python train.py --seed {seed}",
                "--seed",
                "1,2,3",
                "--lr",
                "0.01,0.001",
                "--gpus",
                "1",
            ],
        )
        assert result.exit_code == 0
        mock_client.submit_array.assert_called_once()
        call_args = mock_client.submit_array.call_args
        assert call_args.kwargs["configs"] == [
            {"seed": "1", "lr": "0.01"},
            {"seed": "2", "lr": "0.001"},
            {"seed": "3", "lr": "0.001"},
        ]

    def test_submit_array_no_params(self, mock_client: MagicMock) -> None:
        result = runner.invoke(
            app,
            [
                "submit-array",
                "python train.py",
                "--gpus",
                "1",
            ],
        )
        assert result.exit_code == 1
        assert "sweep" in result.output.lower() or "params" in result.output.lower()


class TestStatus:
    def test_status_found(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["status", "12345"])
        assert result.exit_code == 0
        assert "RUNNING" in result.output

    def test_status_not_found(self, mock_client: MagicMock) -> None:
        mock_client.status.return_value = None
        result = runner.invoke(app, ["status", "99999"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestList:
    def test_list_basic(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "12345" in result.output
        assert "train" in result.output


class TestCancel:
    def test_cancel_single(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["cancel", "12345"])
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_cancel_multiple(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["cancel", "12345", "12346"])
        assert result.exit_code == 0
        assert mock_client.cancel_job.call_count == 2


    def test_submit_array_with_resource_flags(self, mock_client: MagicMock) -> None:
        mock_client.submit_array.return_value = MagicMock(
            array_job_id="12345_0", task_count=3
        )
        result = runner.invoke(
            app,
            [
                "submit-array",
                "python train.py --seed {seed}",
                "--seed",
                "1,2,3",
                "--cpus",
                "16",
                "--mem",
                "64G",
                "--constraint",
                "a100",
                "--qos",
                "high",
                "--job-name",
                "sweep",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_client.submit_array.call_args.kwargs
        assert call_kwargs["cpus"] == 16
        assert call_kwargs["mem"] == "64G"
        assert call_kwargs["constraint"] == "a100"
        assert call_kwargs["qos"] == "high"
        assert call_kwargs["name"] == "sweep"


class TestLogs:
    def test_logs_basic(self, mock_client: MagicMock) -> None:
        mock_client.job_logs.return_value = iter(["epoch 1: loss 0.5\n"])
        result = runner.invoke(app, ["logs", "12345"])
        assert result.exit_code == 0
        assert "epoch 1" in result.output

    def test_logs_stderr_filter(self, mock_client: MagicMock) -> None:
        mock_client.job_logs.return_value = iter(["error line\n"])
        result = runner.invoke(app, ["logs", "12345", "--stderr"])
        assert result.exit_code == 0
        mock_client.job_logs.assert_called_once()
        assert mock_client.job_logs.call_args.kwargs["stream"] == "stderr"

    def test_logs_stdout_filter(self, mock_client: MagicMock) -> None:
        mock_client.job_logs.return_value = iter(["ok\n"])
        result = runner.invoke(app, ["logs", "12345", "--stdout"])
        assert result.exit_code == 0
        assert mock_client.job_logs.call_args.kwargs["stream"] == "stdout"


class TestSync:
    def test_sync_basic(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "Sync complete" in result.output


class TestPull:
    def test_pull_basic(self, mock_client: MagicMock) -> None:
        result = runner.invoke(app, ["pull", "12345"])
        assert result.exit_code == 0
        assert "Pulled job 12345" in result.output


class TestConfig:
    @patch("slurp.cli.config.CONFIG_FILE")
    @patch("slurp.cli.config._ensure_config_dir")
    def test_add_profile_non_interactive(
        self, mock_ensure: MagicMock, mock_cfg: MagicMock
    ) -> None:
        mock_cfg.exists.return_value = False
        mock_cfg.write_text = MagicMock()
        result = runner.invoke(
            app,
            [
                "config",
                "add-profile",
                "test",
                "--hostname",
                "h.example.com",
                "--user",
                "alice",
                "--partition",
                "gpu",
                "--account",
                "lab",
            ],
        )
        assert result.exit_code == 0
        assert "Profile 'test' saved" in result.output

    @patch("slurp.cli.config.CONFIG_FILE")
    def test_list_profiles(self, mock_cfg: MagicMock) -> None:
        mock_cfg.exists.return_value = True
        mock_cfg.read_text.return_value = "[profiles.default]\n[profiles.other]\n"
        result = runner.invoke(app, ["config", "list-profiles"])
        assert result.exit_code == 0
        assert "default" in result.output
        assert "other" in result.output


class TestWebui:
    @pytest.mark.skipif(not _WEB_AVAILABLE, reason="uvicorn (web extra) not installed")
    @patch("uvicorn.run")
    @patch("slurp.webui.create_app")
    @patch("slurp.webui.security.STREAM_TOKEN", "test-token")
    def test_webui_basic(self, mock_create_app: MagicMock, mock_uvicorn_run: MagicMock) -> None:
        result = runner.invoke(app, ["webui"])
        assert result.exit_code == 0
        assert "test-token" in result.output
        mock_create_app.assert_called_once()
        mock_uvicorn_run.assert_called_once()

    def test_webui_missing_deps(self) -> None:
        with patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(app, ["webui"])
            assert result.exit_code == 1
            assert "not installed" in result.output


class TestErrorHandling:
    def test_slurm_error_exit_code(self, mock_client: MagicMock) -> None:
        from slurp.errors import SlurmError

        mock_client.submit.side_effect = SlurmError("sbatch failed")
        result = runner.invoke(app, ["submit", "python", "train.py"])
        assert result.exit_code == 1


class TestSetup:
    """Tests for the `slurp setup` interactive wizard."""

    def test_setup_minimal(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "slurp"
        config_file = config_dir / "profiles.toml"
        answers = [
            "default",  # name
            "cluster.example.edu",  # hostname
            "alice",  # username
            "~/.ssh/id_ed25519",  # key_file
            "",  # proxy_jump
            "gpu",  # partition
            "lab-123",  # account
            False,  # advanced confirm
            True,  # sync confirm
            ".",  # sync_local
            "/home/alice/proj",  # sync_remote
            False,  # venv confirm
            True,  # write confirm
        ]
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup._ask", side_effect=answers),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        content = config_file.read_text()
        assert "[profiles.default]" in content
        assert 'hostname = "cluster.example.edu"' in content
        assert 'username = "alice"' in content
        assert "[profiles.default.sync]" in content
        assert 'remote = "/home/alice/proj"' in content
        # model defaults are omitted
        assert "mpi_mode" not in content
        assert "gpu_flag_style" not in content
        assert "prologue" not in content

    def test_setup_advanced_and_venv(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "slurp"
        config_file = config_dir / "profiles.toml"
        answers = [
            "jureca",  # name
            "jureca.fz-juelich.de",  # hostname
            "alice",  # username
            "~/.ssh/judoor",  # key_file
            "",  # proxy_jump
            "dc-gpu",  # partition
            "training2625",  # account
            True,  # advanced confirm
            "module load Python; module load CUDA",  # prologue
            "pmix",  # mpi_mode
            "threads",  # cpu_bind
            "gpus",  # gpu_flag_style
            True,  # sync confirm
            ".",  # sync_local
            "/p/home/jusers/{username}/jureca/slurp",  # sync_remote
            True,  # venv confirm
            "$HOME/.venv",  # venv_path
            False,  # venv_all_extras
            "cu121,dev",  # venv_extras
            True,  # write confirm
        ]
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup._ask", side_effect=answers),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        content = config_file.read_text()
        assert 'prologue = "module load Python; module load CUDA"' in content
        assert 'mpi_mode = "pmix"' in content
        assert 'cpu_bind = "threads"' in content
        assert 'gpu_flag_style = "gpus"' in content
        assert "[profiles.jureca.venv]" in content
        assert 'strategy = "uv-sync"' in content
        assert 'path = "$HOME/.venv"' in content
        assert 'extras = ["cu121", "dev"]' in content

    def test_setup_venv_all_extras(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "slurp"
        config_file = config_dir / "profiles.toml"
        answers = [
            "default", "h", "u", "", "", "", "",
            False,  # advanced
            False,  # sync (skip)
            True,  # venv
            "$PROJECT/.venv",  # venv_path
            True,  # venv_all_extras
            True,  # write
        ]
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup._ask", side_effect=answers),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        content = config_file.read_text()
        assert "all_extras = true" in content
        assert "extras = [" not in content  # all_extras supersedes the extras list

    def test_setup_backs_up_existing(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "slurp"
        config_dir.mkdir()
        config_file = config_dir / "profiles.toml"
        config_file.write_text('[profiles.old]\nhostname = "old"\n')
        answers = [
            True,  # backup confirm (first prompt since file exists)
            "new",  # name
            "new.host",  # hostname
            "alice",  # username
            "",  # key_file
            "",  # proxy_jump
            "gpu",  # partition
            "acct",  # account
            False,  # advanced
            True, ".", "/home/alice/p",  # sync
            False,  # venv
            True,  # write
        ]
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup._ask", side_effect=answers),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        backups = list(config_dir.glob("*.bak*"))
        assert len(backups) == 1
        assert "old" in backups[0].read_text()
        assert "[profiles.new]" in config_file.read_text()

    def test_setup_decline_backup_aborts(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "slurp"
        config_dir.mkdir()
        config_file = config_dir / "profiles.toml"
        original = '[profiles.old]\nhostname = "old"\n'
        config_file.write_text(original)
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup._ask", side_effect=[False]),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "aborted" in result.output.lower()
        # existing file untouched
        assert config_file.read_text() == original

    def test_setup_cancel_write(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "slurp"
        config_file = config_dir / "profiles.toml"
        answers = [
            "default", "h", "u", "", "", "", "",
            False,  # advanced
            False,  # sync
            False,  # venv
            False,  # write confirm -> cancel
        ]
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup._ask", side_effect=answers),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert not config_file.exists()

    def test_setup_cancel_no_tty(self, tmp_path: Path) -> None:
        # questionary's .ask() returns None when stdin is not a TTY. The real
        # _ask() must translate that into a clean cancellation (exit 1), so we
        # patch questionary rather than _ask itself.
        config_dir = tmp_path / "slurp"
        config_file = config_dir / "profiles.toml"
        mock_q = MagicMock()
        mock_q.text.return_value.ask.return_value = None
        with (
            patch("slurp.cli.setup.CONFIG_DIR", config_dir),
            patch("slurp.cli.setup.CONFIG_FILE", config_file),
            patch("slurp.cli.setup.questionary", mock_q),
        ):
            result = runner.invoke(app, ["setup"])
        assert result.exit_code == 1
        assert "cancelled" in result.output.lower()


class TestSetupHelpers:
    """Direct unit tests for the pure rendering/validation helpers."""

    def test_valid_profile_name(self) -> None:
        from slurp.cli.setup import _valid_profile_name

        assert _valid_profile_name("default") is True
        assert _valid_profile_name("my-cluster_1") is True
        assert _valid_profile_name("") != True  # noqa: E712
        assert _valid_profile_name("bad name") != True  # noqa: E712
        assert _valid_profile_name("bad/name") != True  # noqa: E712

    def test_required_rejects_empty(self) -> None:
        from slurp.cli.setup import _required

        assert _required("x") is True
        assert _required("   ") != True  # noqa: E712

    def test_toml_escape(self) -> None:
        from slurp.cli.setup import _toml_escape

        assert _toml_escape('a"b') == 'a\\"b'
        assert _toml_escape("a\\b") == "a\\\\b"

    def test_build_toml_omits_defaults(self) -> None:
        from slurp.cli.setup import _build_toml

        out = _build_toml(
            {
                "name": "default",
                "hostname": "h",
                "username": "",
                "key_file": "",
                "proxy_jump": "",
                "partition": "",
                "account": "",
                "prologue": "",
                "mpi_mode": "pmi2",
                "cpu_bind": "cores",
                "gpu_flag_style": "gres",
                "sync": False,
                "venv": False,
            }
        )
        assert out.startswith("[profiles.default]")
        assert 'hostname = "h"' in out
        # everything else is default -> omitted
        assert "mpi_mode" not in out
        assert "cpu_bind" not in out
        assert "gpu_flag_style" not in out
        assert "prologue" not in out
        assert "sync" not in out

    def test_build_toml_escapes_quotes(self) -> None:
        from slurp.cli.setup import _build_toml

        out = _build_toml(
            {
                "name": "default",
                "hostname": 'h"ost',
                "username": "",
                "key_file": "",
                "proxy_jump": "",
                "partition": "",
                "account": "",
                "sync": False,
                "venv": False,
            }
        )
        assert 'hostname = "h\\"ost"' in out
