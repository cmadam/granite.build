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

"""
Restart-simulation integration test for the SkyPilot reattach guarantee (F1).

A live BlueVela/SkyPilot cluster is not available in CI, so we simulate a
gbserver process restart with two independent ``Skypilot`` instances that share
one stateful fake step-store and a mocked ``sky`` SDK. The first instance does a
fresh launch and persists its handle; the second instance (empty in-memory
dicts, modelling the restarted process) must read that handle back and
REATTACH to the still-running cluster instead of launching a duplicate.
"""

import asyncio
import types
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.standalone


def _make_env():
    """Construct a Skypilot the same way the unit-test fixture does."""
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    event_q = asyncio.Queue()
    config = EnvironmentConfig(
        name="test-skypilot",
        type="Skypilot",
        config={
            "default_cloud": "k8s",
            "idle_minutes_to_autostop": 15,
        },
    )
    return Skypilot(event_q=event_q, environment_config=config)


class _FakeStepStore:
    """Stateful in-memory stand-in for step_storage, shared across instances."""

    def __init__(self):
        self._rows = {}  # uuid -> SimpleNamespace with .skypilot_handle

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


def _running_status():
    running = MagicMock()
    running.is_terminal = MagicMock(return_value=False)
    running.name = "RUNNING"
    return running


@pytest.mark.asyncio
async def test_restart_adopts_existing_cluster():
    """After a simulated restart, a fresh Skypilot instance reads the persisted
    handle and reattaches to the running cluster instead of relaunching."""
    fake_store = _FakeStepStore()
    admin_storage_mock = MagicMock(step_storage=fake_store, build_storage=MagicMock())

    # A sky mock supporting BOTH a fresh launch and a reattach probe.
    mock_sky = MagicMock()
    mock_sky.Resources = MagicMock(return_value=MagicMock())
    mock_sky.Task = MagicMock(return_value=MagicMock())
    mock_sky.launch = MagicMock(return_value="req-1")
    mock_sky.stream_and_get = MagicMock(return_value=(7, MagicMock()))
    mock_sky.job_status = MagicMock(return_value="rq")
    # The persisted handle from env1's fresh launch has job_id 7, so the probe
    # must report job 7 as running.
    mock_sky.get = MagicMock(return_value={7: _running_status()})

    launch_id = "lid-1"

    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        patch("gbserver.environment.skypilot.is_standalone", return_value=True),
        patch(
            "gbserver.environment.skypilot.get_admin_storage",
            return_value=admin_storage_mock,
        ),
    ):
        # --- env1: original process, fresh launch ---
        env1 = _make_env()
        with patch.object(env1, "_build_name_for", return_value="mybuild"):
            env1._get_launch_ready_event(launch_id)
            await env1.launch_skypilot(
                launch_id=launch_id,
                launcher_config={"run": "echo hi"},
                run_metadata=_RUN_METADATA,
            )

        # env1 did a FRESH launch and persisted the handle.
        mock_sky.launch.assert_called_once()
        stored = fake_store.get_by_uuid("tsr-1")
        assert stored is not None
        assert stored.skypilot_handle is not None
        assert stored.skypilot_handle["cluster_name"] == env1._cluster_names[launch_id]
        assert stored.skypilot_handle["job_id"] == 7

        mock_sky.launch.reset_mock()

        # --- env2: restarted process, empty in-memory state ---
        env2 = _make_env()
        assert env2._cluster_names == {}
        assert env2._job_ids == {}
        with patch.object(env2, "_build_name_for", return_value="mybuild"):
            env2._get_launch_ready_event(launch_id)
            await env2.launch_skypilot(
                launch_id=launch_id,
                launcher_config={"run": "echo hi"},
                run_metadata=_RUN_METADATA,
            )

        # env2 ADOPTED the running cluster: no fresh launch, adopted name/job.
        mock_sky.launch.assert_not_called()
        assert env2._cluster_names[launch_id] == env1._cluster_names[launch_id]
        assert env2._job_ids[launch_id] == 7


@pytest.mark.asyncio
async def test_restart_relaunches_when_cluster_gone():
    """If the persisted cluster no longer exists after restart, the fresh
    instance falls through to a brand-new launch."""
    fake_store = _FakeStepStore()
    admin_storage_mock = MagicMock(step_storage=fake_store, build_storage=MagicMock())

    mock_sky = MagicMock()
    mock_sky.Resources = MagicMock(return_value=MagicMock())
    mock_sky.Task = MagicMock(return_value=MagicMock())
    mock_sky.launch = MagicMock(return_value="req-1")
    mock_sky.stream_and_get = MagicMock(return_value=(7, MagicMock()))
    # By default the reattach probe finds the cluster running; env2 overrides
    # job_status to raise (cluster gone).
    mock_sky.job_status = MagicMock(return_value="rq")
    mock_sky.get = MagicMock(return_value={7: _running_status()})

    launch_id = "lid-1"

    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        patch("gbserver.environment.skypilot.is_standalone", return_value=True),
        patch(
            "gbserver.environment.skypilot.get_admin_storage",
            return_value=admin_storage_mock,
        ),
    ):
        # --- env1: original process, fresh launch ---
        env1 = _make_env()
        with patch.object(env1, "_build_name_for", return_value="mybuild"):
            env1._get_launch_ready_event(launch_id)
            await env1.launch_skypilot(
                launch_id=launch_id,
                launcher_config={"run": "echo hi"},
                run_metadata=_RUN_METADATA,
            )
        mock_sky.launch.assert_called_once()
        mock_sky.launch.reset_mock()

        # --- env2: restarted process; the stored cluster is gone ---
        mock_sky.job_status = MagicMock(
            side_effect=Exception(
                f"Cluster {env1._cluster_names[launch_id]} does not exist"
            )
        )
        env2 = _make_env()
        with patch.object(env2, "_build_name_for", return_value="mybuild"):
            env2._get_launch_ready_event(launch_id)
            await env2.launch_skypilot(
                launch_id=launch_id,
                launcher_config={"run": "echo hi"},
                run_metadata=_RUN_METADATA,
            )

        # env2 could not reattach, so it did a FRESH relaunch.
        mock_sky.launch.assert_called_once()
        assert env2._job_ids[launch_id] == 7
