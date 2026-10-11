"""Unit tests for the sage-eval steps' vLLM ipc socket directory.

Each sage-eval step points TMPDIR at a per-task scratch directory named after
SKYPILOT_TASK_ID. vLLM's V1 engine binds its ZMQ ipc sockets under
tempfile.gettempdir(), so they land in that directory too, and a unix socket path
may be at most 107 characters (sizeof(sockaddr_un.sun_path) - 1).

The task id embeds the cluster name, so the socket path is far over that limit.
On BlueVela the dense granite-4.1-3b model failed SALAD-Bench this way at engine
start:

    zmq.error.ZMQError: ipc path "/opt/nvme/granitebuild/tmp-sky-2026-10-11-03-40-54-
    185155_gb-distill-350m-model-s-eval-salad-bench-m1-72e9eaa5-641_1/420bb520-...
    " is longer than 107 characters

Hence VLLM_RPC_BASE_PATH, which vLLM reads instead of tempfile.gettempdir() for
those sockets only.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
ENVIRONMENTS = REPO_ROOT / "configurations/assets/environments"
SAGE_EVAL_STEPS = [
    ENVIRONMENTS / "skypilot/lsf/ibm-bluevela/steps/sage-eval/step.yaml",
    ENVIRONMENTS / "skypilot/aws/steps/sage-eval/step.yaml",
    ENVIRONMENTS / "skypilot/aws/steps/sage-eval-code-grouped/step.yaml",
]

# The task id of the failing BlueVela run above, and vLLM's socket name (a uuid4).
TASK_ID = (
    "sky-2026-10-11-03-40-54-185155_"
    "gb-distill-350m-model-s-eval-salad-bench-m1-72e9eaa5-641_1"
)
SOCKET_NAME = "420bb520-b3a8-41b9-b2f3-cf3dd3b25e7c"
SUN_PATH_MAX = 107


def _scripts(node):
    """Every string in the step's YAML that sets TMPDIR: its run script(s)."""
    if isinstance(node, dict):
        for value in node.values():
            yield from _scripts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _scripts(value)
    elif isinstance(node, str) and "export TMPDIR=" in node:
        yield node


def _run_script(path):
    scripts = list(_scripts(yaml.safe_load(path.read_text(encoding="utf-8"))))
    assert len(scripts) == 1, f"{path}: expected one script setting TMPDIR"
    return scripts[0]


def _tmpdirs(script):
    """Each value the script may give TMPDIR, with the task id substituted."""
    values = re.findall(r'export TMPDIR="([^"]+)"', script)
    return [
        v.replace("${USER:-$(id -un)}", "granitebuild").replace(
            "${SKYPILOT_TASK_ID:-$$}", TASK_ID
        )
        for v in values
    ]


@pytest.mark.parametrize("path", SAGE_EVAL_STEPS, ids=lambda p: p.parent.name)
def test_the_tmpdir_is_too_long_for_a_socket(path):
    """Every TMPDIR branch overflows sun_path, the NVMe one and the /tmp fallback
    alike. If this ever stops holding, VLLM_RPC_BASE_PATH is no longer needed."""
    for tmpdir in _tmpdirs(_run_script(path)):
        assert len(f"{tmpdir}/{SOCKET_NAME}") > SUN_PATH_MAX, tmpdir


@pytest.mark.parametrize("path", SAGE_EVAL_STEPS, ids=lambda p: p.parent.name)
def test_vllm_binds_its_sockets_under_a_short_directory(path):
    script = _run_script(path)
    match = re.search(r'export VLLM_RPC_BASE_PATH="([^"]+)"', script)
    assert match, f"{path}: VLLM_RPC_BASE_PATH is not set"
    assert len(f"{match.group(1)}/{SOCKET_NAME}") <= SUN_PATH_MAX
    # Set after TMPDIR, so nothing in between can point it back under TMPDIR.
    assert match.start() > script.rindex("export TMPDIR=")
