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

"""Unit tests for the F2 (epic #46) login-node completion-marker reader."""

from unittest.mock import MagicMock, patch

import pytest


def test_ssh_alias_and_command_built_for_marker_check():
    from gbserver.environment._skypilot_marker import _build_marker_check_argv

    argv = _build_marker_check_argv(
        cloud="lsf", alias="bluevela", marker_path="/wd/.gb_done/tsr-1"
    )
    # Uses the cloud-specific OpenSSH config file materialize wrote.
    assert "-F" in argv
    fidx = argv.index("-F")
    assert argv[fidx + 1].endswith("/.lsf/config")
    # Targets the host alias and runs a non-destructive existence probe.
    assert "bluevela" in argv
    joined = " ".join(argv)
    assert "test -f" in joined
    assert "/wd/.gb_done/tsr-1" in joined


@pytest.mark.asyncio
async def test_marker_present_returns_true():
    from gbserver.environment import _skypilot_marker as m

    completed = MagicMock(returncode=0)
    with patch.object(m.subprocess, "run", return_value=completed):
        found = await m.read_done_marker_via_ssh(
            cloud="lsf", alias="bluevela", marker_path="/wd/.gb_done/tsr-1"
        )
    assert found is True


@pytest.mark.asyncio
async def test_marker_absent_returns_false():
    from gbserver.environment import _skypilot_marker as m

    completed = MagicMock(returncode=1)  # `test -f` → not found
    with patch.object(m.subprocess, "run", return_value=completed):
        found = await m.read_done_marker_via_ssh(
            cloud="lsf", alias="bluevela", marker_path="/wd/.gb_done/tsr-1"
        )
    assert found is False


@pytest.mark.asyncio
async def test_ssh_failure_returns_false_not_raises():
    """An unreachable login node must NOT crash resume; it degrades to
    'no marker' so the caller falls through to reattach/relaunch."""
    from gbserver.environment import _skypilot_marker as m

    with patch.object(m.subprocess, "run", side_effect=OSError("ssh boom")):
        found = await m.read_done_marker_via_ssh(
            cloud="lsf", alias="bluevela", marker_path="/wd/.gb_done/tsr-1"
        )
    assert found is False


@pytest.mark.asyncio
async def test_missing_inputs_return_false_without_ssh():
    """No cloud / alias / marker_path → no SSH attempt, False."""
    from gbserver.environment import _skypilot_marker as m

    with patch.object(m.subprocess, "run") as run:
        assert await m.read_done_marker_via_ssh("", "bluevela", "/wd/x") is False
        assert await m.read_done_marker_via_ssh("lsf", "", "/wd/x") is False
        assert await m.read_done_marker_via_ssh("lsf", "bluevela", "") is False
        run.assert_not_called()
