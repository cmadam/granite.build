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

"""Unit tests for BuildWatcher's RUNNING-build resume scan (issue #50).

These exercise the scan/dispatch logic in isolation by building a bare
``BuildWatcher`` (via ``__new__``) and wiring only the attributes the resume
path touches — no real storage, threads, or runners.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gbserver.buildwatcher.buildwatcher import BuildWatcher
from gbserver.types.constants import COMMAND_RUN_BUILD_WATCH_BUILD_NAME
from gbserver.types.status import Status


def _bare_watcher(running_builds, watched_spaces=("standalone",)):
    """Return a BuildWatcher with just enough wiring for the resume scan."""
    watcher = BuildWatcher.__new__(BuildWatcher)
    storage = MagicMock()
    storage.build_storage.get_by_where.return_value = list(running_builds)
    watcher.storage = storage
    # _spaces must be a real mapping — the scan does `b.space_name in _spaces.keys()`.
    watcher.config = SimpleNamespace(_spaces={s: object() for s in watched_spaces})
    return watcher


def _build(uuid, space_name="standalone", name="my-build"):
    return SimpleNamespace(uuid=uuid, space_name=space_name, name=name)


class TestResumeRunningBuilds:
    def test_redispatches_running_builds_with_enable_resume(self):
        """Each watched, non-local RUNNING build is re-dispatched with resume on."""
        builds = [_build("b1"), _build("b2")]
        watcher = _bare_watcher(builds)
        # Patch the (name-mangled) private dispatch so no thread/runner is created.
        watcher._BuildWatcher__start_build = MagicMock()

        resumed = watcher.resume_running_builds()

        assert resumed == 2
        # Queried specifically for RUNNING builds.
        where = watcher.storage.build_storage.get_by_where.call_args.kwargs["where"]
        assert where == {"status": Status.RUNNING.name}
        # Every dispatch carried enable_resume=True.
        assert watcher._BuildWatcher__start_build.call_count == 2
        for call in watcher._BuildWatcher__start_build.call_args_list:
            assert call.kwargs.get("enable_resume") is True

    def test_skips_builds_in_unwatched_spaces(self):
        """A RUNNING build outside a watched space is not re-dispatched."""
        builds = [_build("mine"), _build("other", space_name="not-watched")]
        watcher = _bare_watcher(builds, watched_spaces=("standalone",))
        watcher._BuildWatcher__start_build = MagicMock()

        resumed = watcher.resume_running_builds()

        assert resumed == 1
        dispatched = [
            c.args[0].uuid for c in watcher._BuildWatcher__start_build.call_args_list
        ]
        assert dispatched == ["mine"]

    def test_skips_local_build_watch_build(self):
        """The internal build-watch build is skipped during resume."""
        builds = [
            _build("real"),
            _build("local", name=COMMAND_RUN_BUILD_WATCH_BUILD_NAME),
        ]
        watcher = _bare_watcher(builds)
        watcher._BuildWatcher__start_build = MagicMock()

        resumed = watcher.resume_running_builds()

        assert resumed == 1
        watcher._BuildWatcher__start_build.assert_called_once()
        assert watcher._BuildWatcher__start_build.call_args.args[0].uuid == "real"

    def test_no_running_builds_dispatches_nothing(self):
        """An empty RUNNING set re-dispatches nothing and returns zero."""
        watcher = _bare_watcher([])
        watcher._BuildWatcher__start_build = MagicMock()

        resumed = watcher.resume_running_builds()

        assert resumed == 0
        watcher._BuildWatcher__start_build.assert_not_called()

    def test_one_failing_build_does_not_abort_the_rest(self):
        """A dispatch failure is logged and the remaining builds still resume."""
        builds = [_build("b1"), _build("b2"), _build("b3")]
        watcher = _bare_watcher(builds)
        watcher._BuildWatcher__start_build = MagicMock(
            side_effect=[RuntimeError("boom"), None, None]
        )

        resumed = watcher.resume_running_builds()

        # b1 failed; b2 and b3 still dispatched.
        assert resumed == 2
        assert watcher._BuildWatcher__start_build.call_count == 3


class TestCreateBuildRunnerResume:
    def _watcher_with_config(self, buildrunner_type):
        watcher = BuildWatcher.__new__(BuildWatcher)
        watcher.gh_token = ""
        watcher.all_build_space_uri = None
        watcher.config = SimpleNamespace(
            buildrunner_type=buildrunner_type,
            gh_api_endpoint="https://api.github.com",
            monitoring_interval=5,
            workspace_dir="/tmp/ws",
        )
        return watcher

    def test_thread_runner_receives_enable_resume(self):
        """The thread runner (standalone default) is built with enable_resume."""
        watcher = self._watcher_with_config("thread")
        build = _build("b1")

        with patch(
            "gbserver.buildwatcher.buildwatcher.BuildRunner"
        ) as mock_runner:
            watcher._BuildWatcher__create_build_runner(build, enable_resume=True)

        assert mock_runner.call_args.kwargs["enable_resume"] is True

    def test_non_thread_runner_ignores_resume(self):
        """Process/job runners don't support resume; the flag is dropped, not passed."""
        watcher = self._watcher_with_config("process")
        build = _build("b1")

        with patch(
            "gbserver.buildwatcher.buildwatcher.BuildRunnerProcess"
        ) as mock_proc, patch(
            "gbserver.buildwatcher.buildwatcher.logger"
        ) as mock_logger:
            watcher._BuildWatcher__create_build_runner(build, enable_resume=True)

        # BuildRunnerProcess does not accept enable_resume, so it must not be forwarded.
        assert "enable_resume" not in mock_proc.call_args.kwargs
        # The unsupported-resume request is surfaced as a warning.
        assert mock_logger.warning.called
