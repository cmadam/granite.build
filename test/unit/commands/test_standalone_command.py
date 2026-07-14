#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the gbserver standalone command."""

import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "test-data"
STANDALONE_SPACE_DIR = TEST_DATA_DIR / "e2e" / "standalone" / "standalone-quickstart"


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestStandaloneCommand:
    """Tests for the gbserver standalone CLI command."""

    def test_command_is_discoverable(self):
        """Verify the standalone command file is auto-discovered by the CLI."""
        from gbserver.cli import GraniteBuildServerCLI

        cli_group = GraniteBuildServerCLI(name="gbserver")
        commands = cli_group.list_commands(ctx=None)
        assert (
            "standalone" in commands
        ), f"'standalone' not found in discovered commands: {commands}"

    def test_standalone_starts_and_serves_api(self):
        """Start the standalone server in a background thread and verify the REST API responds."""
        # Environment vars MUST be set before importing _run_standalone
        # because root_api imports trigger storage initialization at module level.
        env = {
            "GBSERVER_METADATA_STORAGE": "sqlite",
            "GBSERVER_DEFAULT_BUILDRUNNER_TYPE": "thread",
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": "",
        }

        port = _find_free_port()
        started_event = threading.Event()

        def on_started():
            started_event.set()

        with patch.dict(os.environ, env):
            # Reset singleton storage so it picks up the sqlite backend from env.
            from gbserver.storage import singleton_storage
            from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

            singleton_storage.set_storage_factory(SqliteStorageFactory())

            from gbserver.commands.command_standalone import _run_standalone

            server_holder: dict = {}

            thread = threading.Thread(
                target=_run_standalone,
                kwargs={
                    "port": port,
                    "space_dir": str(STANDALONE_SPACE_DIR),
                    "on_started": on_started,
                    "on_server_created": lambda s: server_holder.__setitem__(
                        "server", s
                    ),
                },
                daemon=True,
                name="test-standalone-server",
            )
            thread.start()

            try:
                # Wait for server startup (up to 30 seconds)
                assert started_event.wait(
                    timeout=30
                ), "Standalone server did not start within 30 seconds"

                # Retry until uvicorn is fully accepting connections.
                last_err = None
                for _ in range(20):
                    try:
                        response = httpx.get(
                            f"http://127.0.0.1:{port}/api/v1", timeout=2
                        )
                        assert (
                            response.status_code == 200
                        ), f"Expected 200, got {response.status_code}: {response.text}"
                        data = response.json()
                        assert (
                            "message" in data
                        ), f"Response missing 'message' key: {data}"
                        last_err = None
                        break
                    except httpx.ConnectError as e:
                        last_err = e
                        time.sleep(0.25)
                if last_err is not None:
                    pytest.fail(
                        f"Could not connect to standalone server on port {port}: {last_err}"
                    )
            finally:
                # Stop the server so its BuildWatcher thread is stopped rather than
                # left polling global storage for the rest of the test session
                # (the daemon flag prevents a hang, but a live watcher could still
                # pick up builds submitted by later tests on this worker).
                server = server_holder.get("server")
                if server is not None:
                    server.should_exit = True
                thread.join(timeout=15)


class TestAutoResumeConfig:
    """The GBSERVER_STANDALONE_AUTO_RESUME standalone default (issue #50)."""

    def test_env_var_in_standalone_defaults(self):
        """Auto-resume is part of STANDALONE_ENV_DEFAULTS, defaulting to 'false'."""
        from gbserver.types.constants import (
            ENV_VAR_STANDALONE_AUTO_RESUME,
            STANDALONE_ENV_DEFAULTS,
        )

        assert ENV_VAR_STANDALONE_AUTO_RESUME in STANDALONE_ENV_DEFAULTS
        assert STANDALONE_ENV_DEFAULTS[ENV_VAR_STANDALONE_AUTO_RESUME] == "false"

    def test_parsed_constant_defaults_false(self):
        """With the env var unset, the parsed boolean constant is False."""
        from gbserver.types.constants import (
            ENV_VAR_STANDALONE_AUTO_RESUME,
            getenv_boolean,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_VAR_STANDALONE_AUTO_RESUME, None)
            assert getenv_boolean(ENV_VAR_STANDALONE_AUTO_RESUME, False) is False


class TestResumeFlagWiring:
    """The `standalone --resume` flag is threaded through to _run_standalone."""

    def _invoke(self, *args):
        # Invoke via cli.main(standalone_mode=False) rather than CliRunner: the
        # latter swaps sys.stdout for a buffer it later closes, which collides with
        # the module's logging StreamHandler ("I/O operation on closed file").
        from gbserver.commands import command_standalone

        with patch.object(command_standalone, "_run_standalone") as mock_run:
            command_standalone.cli.main(
                ["--space-dir", str(STANDALONE_SPACE_DIR), *args],
                standalone_mode=False,
            )
        assert mock_run.called
        return mock_run

    def test_resume_flag_forwards_true(self):
        mock_run = self._invoke("--resume")
        assert mock_run.call_args.kwargs["resume"] is True

    def test_no_resume_flag_forwards_false(self):
        mock_run = self._invoke()
        assert mock_run.call_args.kwargs["resume"] is False


class TestMaybeResumeRunningBuilds:
    """The startup resume trigger: --resume flag OR the auto-resume config."""

    def _run(self, resume, auto_resume_env):
        import gbserver.types.constants as constants
        from gbserver.commands import command_standalone

        watcher = MagicMock()
        # The helper imports GBSERVER_STANDALONE_AUTO_RESUME from constants at call
        # time, so patching the module attribute controls the config branch.
        with patch.object(
            constants, "GBSERVER_STANDALONE_AUTO_RESUME", auto_resume_env
        ):
            command_standalone._maybe_resume_running_builds(watcher, resume)
        return watcher

    def test_resume_flag_triggers_scan(self):
        watcher = self._run(resume=True, auto_resume_env=False)
        watcher.resume_running_builds.assert_called_once_with()

    def test_auto_resume_env_triggers_scan(self):
        watcher = self._run(resume=False, auto_resume_env=True)
        watcher.resume_running_builds.assert_called_once_with()

    def test_disabled_does_not_scan(self):
        watcher = self._run(resume=False, auto_resume_env=False)
        watcher.resume_running_builds.assert_not_called()
