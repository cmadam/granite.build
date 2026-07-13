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

"""F2 (epic #46): read a job's cluster-side completion marker over SSH.

When a standalone gbserver restarts AFTER a job finished, the compute
allocation is gone but the shared filesystem is still reachable from the
cluster's persistent login node. SkyPilot's slurm/lsf provisioners read login
connection details from ``~/.<cloud>/config`` (written by
``skypilot_config.materialize``), and those OpenSSH ``Host <alias>`` blocks
survive the dead allocation. We reuse that same file to probe the marker with a
non-destructive ``test -f``.

gbserver only READS here — it never writes the marker (the job does, on exit 0).
Any SSH/connection failure degrades to "no marker" so resume falls through to
the F1 reattach / fresh-launch path rather than crashing.
"""

import asyncio
import shlex
import subprocess
from pathlib import Path
from typing import List

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Match the connect-timeout posture of _skypilot_ssh so an unreachable login
# node fails fast instead of hanging resume. BatchMode disables interactive
# prompts (password / host-key) that would otherwise block indefinitely.
_SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "BatchMode=yes",
]


def _cloud_ssh_config_path(cloud: str) -> str:
    """Path ``materialize()`` writes the login-node Host blocks to for a cloud."""
    return str(Path.home() / f".{cloud}" / "config")


def _build_marker_check_argv(cloud: str, alias: str, marker_path: str) -> List[str]:
    """Argv for a non-destructive existence probe of the marker on the login
    node, using the cloud-specific OpenSSH config ``materialize`` wrote."""
    config_path = _cloud_ssh_config_path(cloud)
    # `test -f <path>` → exit 0 if the marker exists, 1 otherwise.
    remote_cmd = f"test -f {shlex.quote(marker_path)}"
    return ["ssh", "-F", config_path, *_SSH_OPTS, alias, remote_cmd]


async def read_done_marker_via_ssh(
    cloud: str, alias: str, marker_path: str, timeout: int = 20
) -> bool:
    """Return True iff the completion marker exists on the login node.

    Never raises: any failure (unreachable host, missing config, timeout) is
    logged and returns False so the caller safely falls through to relaunch.

    :param cloud: Cloud name (e.g. ``lsf``/``slurm``); selects ``~/.<cloud>/config``.
    :param alias: OpenSSH ``Host`` alias for the login node (the cluster name).
    :param marker_path: Absolute marker path to probe on the remote filesystem.
    :param timeout: Max seconds to wait for the SSH probe.
    """
    if not (cloud and alias and marker_path):
        return False
    argv = _build_marker_check_argv(cloud, alias, marker_path)
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            argv,
            timeout=timeout,
            capture_output=True,
        )
    except Exception as e:  # unreachable / timeout / no ssh — degrade gracefully
        logger.info(
            "Completion-marker SSH probe failed (%s); treating as no marker "
            "(cloud=%s alias=%s path=%s)",
            e,
            cloud,
            alias,
            marker_path,
        )
        return False
    found = result.returncode == 0
    logger.info(
        "Completion-marker probe on %s (%s): %s (rc=%s)",
        alias,
        marker_path,
        "PRESENT" if found else "absent",
        result.returncode,
    )
    return found
