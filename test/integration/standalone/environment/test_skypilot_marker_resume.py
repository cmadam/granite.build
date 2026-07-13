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

"""Restart-simulation for the F2 (epic #46) completion-marker short-circuit.

If a job finished while gbserver was offline, the durable ``.gb_done`` marker is
present on the cluster's persistent login node. On relaunch a standalone
gbserver reads that marker BEFORE the F1 reattach probe; if present it skips
provisioning entirely and lets the step complete as SUCCESS (so resume does not
wrongly re-run a step that already finished). If absent, it falls through to the
F1 reattach / fresh-launch path.
"""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.standalone


def _make_env():
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    event_q = asyncio.Queue()
    config = EnvironmentConfig(
        name="test-skypilot",
        type="Skypilot",
        config={"default_cloud": "k8s", "idle_minutes_to_autostop": 15},
    )
    return Skypilot(event_q=event_q, environment_config=config)


class _FakeStepStore:
    def __init__(self):
        self._rows = {}

    def update_fields(self, uuid, fields, *args, **kwargs):
        row = self._rows.setdefault(uuid, types.SimpleNamespace(skypilot_handle=None))
        for k, v in fields.items():
            setattr(row, k, v)
        return row

    def get_by_uuid(self, uuid):
        return self._rows.get(uuid)


_RUN_METADATA = {
    "build_id": "b-1",
    "username": "bob",
    "target_name": "t",
    "target_step_index": 0,
    "targetsteprun_id": "tsr-1",
}

_BUILD_WORKDIR = "/shared/builds/b-1/runs/tr-1"


@pytest.mark.asyncio
async def test_marker_present_marks_success_without_relaunch():
    """Marker present → no provisioning, no cluster recorded, monitors
    released (so the step completes SUCCESS through the normal path)."""
    fake_store = _FakeStepStore()
    admin_storage_mock = MagicMock(step_storage=fake_store, build_storage=MagicMock())

    mock_sky = MagicMock()
    mock_sky.Resources = MagicMock(return_value=MagicMock())
    mock_sky.Task = MagicMock(return_value=MagicMock())
    mock_sky.launch = MagicMock(return_value="req-1")

    launch_id = "lid-1"

    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        patch("gbserver.environment.skypilot.is_standalone", return_value=True),
        patch(
            "gbserver.environment.skypilot.get_admin_storage",
            return_value=admin_storage_mock,
        ),
        patch(
            "gbserver.environment.skypilot.read_done_marker_via_ssh",
            new=AsyncMock(return_value=True),
        ) as marker_probe,
    ):
        env = _make_env()
        with patch.object(env, "_build_name_for", return_value="mybuild"):
            env._get_launch_ready_event(launch_id)
            await env.launch_skypilot(
                launch_id=launch_id,
                launcher_config={
                    "run": "echo hi",
                    "resources": {"infra": "lsf/bluevela"},
                },
                run_metadata=_RUN_METADATA,
                setup_config={"skypilot": {"build_workdir": _BUILD_WORKDIR}},
            )

        # The marker was probed with the right cloud/alias/path.
        marker_probe.assert_awaited_once()
        _, kwargs = marker_probe.call_args
        args = marker_probe.call_args.args
        called = list(args) + list(kwargs.values())
        assert "lsf" in called
        assert "bluevela" in called
        assert f"{_BUILD_WORKDIR}/.gb_done/tsr-1" in called

        # No provisioning happened, and no cluster was recorded for the launch
        # (so the poller no-ops and cleanup no-ops).
        mock_sky.launch.assert_not_called()
        assert launch_id not in env._cluster_names

        # Monitors were released so the step can complete through Run.run.
        assert env._get_launch_ready_event(launch_id).is_set()


@pytest.mark.asyncio
async def test_marker_absent_falls_through_to_launch():
    """Marker absent → normal fresh launch (F1 path unchanged)."""
    fake_store = _FakeStepStore()
    admin_storage_mock = MagicMock(step_storage=fake_store, build_storage=MagicMock())

    mock_sky = MagicMock()
    mock_sky.Resources = MagicMock(return_value=MagicMock())
    mock_sky.Task = MagicMock(return_value=MagicMock())
    mock_sky.launch = MagicMock(return_value="req-1")
    mock_sky.stream_and_get = MagicMock(return_value=(7, MagicMock()))

    launch_id = "lid-1"

    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        patch("gbserver.environment.skypilot.is_standalone", return_value=True),
        patch(
            "gbserver.environment.skypilot.get_admin_storage",
            return_value=admin_storage_mock,
        ),
        patch(
            "gbserver.environment.skypilot.read_done_marker_via_ssh",
            new=AsyncMock(return_value=False),
        ),
    ):
        env = _make_env()
        with patch.object(env, "_build_name_for", return_value="mybuild"):
            env._get_launch_ready_event(launch_id)
            await env.launch_skypilot(
                launch_id=launch_id,
                launcher_config={
                    "run": "echo hi",
                    "resources": {"infra": "lsf/bluevela"},
                },
                run_metadata=_RUN_METADATA,
                setup_config={"skypilot": {"build_workdir": _BUILD_WORKDIR}},
            )

        # No marker → provisioned a fresh cluster.
        mock_sky.launch.assert_called_once()
        assert env._job_ids[launch_id] == 7
