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

"""Docker env:// input: bind-mounted read-only, binding rewritten to the container path."""

import asyncio

import pytest

from gbserver.environment.docker import Docker
from gbserver.environment.environment import BINDING_KEY


@pytest.fixture
def docker_env():
    return Docker(event_q=asyncio.Queue())


def _env_uri(path) -> str:
    # ``env://{{ binding.path }}`` with an absolute path renders as env:///abs/...
    return f"env://{path}"


@pytest.mark.asyncio
async def test_file_input_is_mounted_read_only(docker_env, tmp_path):
    f = tmp_path / "gen" / "generation.json"
    f.parent.mkdir()
    f.write_text("{}")
    uri = _env_uri(f)
    assert uri.startswith("env:///")
    binding, step = await docker_env.pullasset_envstore(uri=uri)
    assert step is None
    path = binding[BINDING_KEY]["path"]
    assert path.startswith("/gb-env-inputs/") and path.endswith("/generation.json")
    vol = docker_env._extra_volumes[str(f.parent)]
    assert vol["mode"] == "ro"
    assert path == f"{vol['bind']}/generation.json"
    assert docker_env._resolve_host_path(path) == str(f)


@pytest.mark.asyncio
async def test_dir_input_is_mounted_whole(docker_env, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(d))
    vol = docker_env._extra_volumes[str(d)]
    assert vol["mode"] == "ro"
    assert binding[BINDING_KEY]["path"] == vol["bind"]


@pytest.mark.asyncio
async def test_missing_path_raises(docker_env, tmp_path):
    with pytest.raises(FileNotFoundError):
        await docker_env.pullasset_envstore(uri=_env_uri(tmp_path / "nope.json"))


@pytest.mark.asyncio
async def test_distinct_inputs_get_distinct_mounts_shared_parent_reused(
    docker_env, tmp_path
):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    a1 = a_dir / "one.json"
    a2 = a_dir / "two.json"
    b1 = b_dir / "one.json"
    for f in (a1, a2, b1):
        f.write_text("{}")

    pa1 = (await docker_env.pullasset_envstore(uri=_env_uri(a1)))[0][BINDING_KEY]
    pa2 = (await docker_env.pullasset_envstore(uri=_env_uri(a2)))[0][BINDING_KEY]
    pb1 = (await docker_env.pullasset_envstore(uri=_env_uri(b1)))[0][BINDING_KEY]

    mount_a = docker_env._extra_volumes[str(a_dir)]["bind"]
    mount_b = docker_env._extra_volumes[str(b_dir)]["bind"]
    assert mount_a != mount_b
    assert pa1["path"] == f"{mount_a}/one.json"
    assert pa2["path"] == f"{mount_a}/two.json"
    assert pb1["path"] == f"{mount_b}/one.json"
    assert pa1["path"] != pb1["path"]
    env_mounts = [
        v
        for v in docker_env._extra_volumes.values()
        if v["bind"].startswith("/gb-env-inputs/")
    ]
    assert len(env_mounts) == 2


@pytest.mark.asyncio
async def test_reuses_existing_read_only_mount(docker_env, tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    docker_env._extra_volumes[str(snap)] = {
        "bind": "/gb-hf-models/org/repo",
        "mode": "ro",
    }
    binding, _ = await docker_env.pullasset_envstore(uri=_env_uri(snap / "config.json"))
    assert binding[BINDING_KEY]["path"] == "/gb-hf-models/org/repo/config.json"
    assert len(docker_env._extra_volumes) == 1


@pytest.mark.asyncio
async def test_input_in_workspace_mount_raises(docker_env, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "gen.json").write_text("{}")
    rw = {"bind": "/gb-workspace", "mode": "rw"}
    docker_env._extra_volumes[str(ws)] = dict(rw)
    with pytest.raises(ValueError, match="Docker workspace dir"):
        await docker_env.pullasset_envstore(uri=_env_uri(ws / "gen.json"))
    assert docker_env._extra_volumes == {str(ws): rw}
