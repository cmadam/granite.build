"""Shared ``.sqsh`` cache for SLURM container images (issue #559).

Covers the pure helpers (naming, enroot URI, config parsing, the rendered
import script), the resolve flow against a scripted fake login node (hit,
miss -> import, an import already queued, failure fallback, in-process
deduplication), and the wiring into ``Skypilot.launch_skypilot``.
"""

import asyncio
import shutil
import subprocess
from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import pytest
from libgbtest.environments.skypilot_mocks import _launch_and_get_resources, _make_env

from gbserver.environment import _slurm_image_cache as cache
from gbserver.environment._slurm_image_cache import (
    SlurmImageCacheConfig,
    enroot_uri,
    import_job_name,
    parse_cache_config,
    render_import_script,
    resolve_cached_image,
    split_image_ref,
    sqsh_name,
)

REF = "us.icr.io/cil15-shared-registry/gb-step-eval-judged:5a6fc8c"
SHARE = "/proj/granite-build/g4os"
SQSH = (
    f"{SHARE}/enroot/us.icr.io-cil15-shared-registry-gb-step-eval-judged-5a6fc8c.sqsh"
)


def _cfg(**kw) -> SlurmImageCacheConfig:
    return SlurmImageCacheConfig(
        share_path=SHARE,
        partition="cpu",
        poll_interval_seconds=0,
        **kw,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestNaming:
    def test_sqsh_name_matches_the_lsf_driver(self):
        # sky/provision/lsf/instance.py:_sqsh_name_from_image -- shared files.
        assert sqsh_name(REF) == (
            "us.icr.io-cil15-shared-registry-gb-step-eval-judged-5a6fc8c.sqsh"
        )

    @pytest.mark.parametrize(
        "ref, uri",
        [
            (REF, "us.icr.io#cil15-shared-registry/gb-step-eval-judged:5a6fc8c"),
            ("localhost:5000/a/b:1", "localhost:5000#a/b:1"),
            ("nvcr.io/nvidia/nemo-rl:v0.7.0", "nvcr.io#nvidia/nemo-rl:v0.7.0"),
            ("vllm/vllm-openai:latest", "vllm/vllm-openai:latest"),
            ("ubuntu:22.04", "ubuntu:22.04"),
        ],
    )
    def test_enroot_uri(self, ref, uri):
        assert enroot_uri(ref) == uri

    def test_job_name_is_deterministic_per_file(self):
        assert import_job_name(SQSH) == import_job_name(SQSH)
        assert import_job_name(SQSH) != import_job_name(SQSH + "x")
        assert import_job_name(SQSH).startswith("gb-image-import-")


class TestSplitImageRef:
    @pytest.mark.parametrize(
        "image_id, expected",
        [
            (f"docker:{REF}", ("docker:", REF)),
            (f"docker://{REF}", ("docker:", REF)),
            (REF, ("", REF)),
            (None, ("", None)),
            ("", ("", None)),
            # Already a .sqsh path: nothing to cache.
            (f"docker:{SQSH}", ("docker:", None)),
            (SQSH, ("", None)),
            # Not a plausible registry ref: never interpolated into a script.
            ("docker:a/b:1;rm -rf /", ("docker:", None)),
            ("docker:a b", ("docker:", None)),
        ],
    )
    def test_split(self, image_id, expected):
        assert split_image_ref(image_id) == expected


class TestParseCacheConfig:
    def test_unset_or_disabled(self):
        assert parse_cache_config(None) is None
        assert parse_cache_config({}) is None
        assert (
            parse_cache_config(
                {"enabled": False, "share_path": SHARE, "partition": "cpu"}
            )
            is None
        )

    def test_full(self):
        cfg = parse_cache_config(
            {
                "share_path": SHARE,
                "partition": "cpu",
                "sbatch_options": {"qos": "cpu"},
                "timeout_seconds": 60,
            }
        )
        assert cfg is not None
        assert cfg.cache_dir == f"{SHARE}/enroot"
        assert cfg.sbatch_options == {"qos": "cpu"}
        assert cfg.timeout_seconds == 60.0

    @pytest.mark.parametrize(
        "raw",
        [
            "yes",
            {"partition": "cpu"},
            {"share_path": "relative/path", "partition": "cpu"},
            {"share_path": "/a b", "partition": "cpu"},
            {"share_path": SHARE},
            {"share_path": SHARE, "partition": "cpu", "sbatch_options": "x"},
        ],
    )
    def test_malformed_fails_loudly(self, raw):
        with pytest.raises(ValueError):
            parse_cache_config(raw)


class TestImportScript:
    def _script(self, **kw) -> str:
        return render_import_script(_cfg(**kw), REF, SQSH)

    def test_directives(self):
        s = self._script(sbatch_options={"qos": "cpu", "account": "grp", "mem": None})
        assert "#SBATCH --partition=cpu" in s
        assert "#SBATCH --qos=cpu" in s
        assert "#SBATCH --account=grp" in s
        assert f"#SBATCH --job-name={import_job_name(SQSH)}" in s
        assert f"#SBATCH --output={SHARE}/enroot/logs/%x-%j.out" in s
        assert "#SBATCH --cpus-per-task=16" in s
        # None drops a default directive.
        assert "--mem" not in s

    def test_reserved_directives_cannot_be_overridden(self):
        s = self._script(sbatch_options={"partition": "gpu-mid", "job-name": "x"})
        assert "#SBATCH --partition=cpu" in s
        assert "--partition=gpu-mid" not in s
        assert "--job-name=x\n" not in s

    def test_locks_the_lsf_lock_file_and_rechecks_under_it(self):
        s = self._script()
        lock = s.index('exec 200>"$SQSH.lock"')
        flock = s.index("flock -x 200")
        recheck = s.index('if sqsh_valid "$SQSH"')
        imp = s.index("enroot import")
        assert lock < flock < recheck < imp

    def test_imports_to_a_job_unique_name_and_renames_last(self):
        s = self._script()
        assert "tmp-$SLURM_JOB_ID.sqsh" in s
        assert "import-$SLURM_JOB_ID.sqsh" in s
        assert 'enroot import -o "$LOCAL_TMP"' in s
        assert s.rindex('mv -f "$FINAL_TMP" "$SQSH"') > s.index(
            'if ! sqsh_valid "$LOCAL_TMP"'
        )
        assert 'trap \'rm -f "$LOCAL_TMP" "$FINAL_TMP"\' EXIT' in s

    def test_uses_the_enroot_registry_syntax(self):
        assert (
            "docker://us.icr.io#cil15-shared-registry/gb-step-eval-judged:5a6fc8c"
            in self._script()
        )

    @pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
    @pytest.mark.parametrize(
        "root, valid",
        [("bin", True), ("usr/bin", True), ("0/bin", False), ("opt", False)],
    )
    def test_validity_check_holds_under_pipefail(self, tmp_path, root, valid):
        # The import script runs `set -euo pipefail`. A `grep -q` reader exits at
        # the first match and SIGPIPEs a long listing, which made a valid image
        # read as invalid on BlueVela. Drive the real function with a fake
        # unsquashfs that lists far more than a pipe buffer.
        fake = tmp_path / "unsquashfs"
        fake.write_text(
            "#!/bin/bash\n"
            "echo squashfs-root\n"
            f"echo squashfs-root/{root}/sh\n"
            "for i in $(seq 200000); do echo squashfs-root/usr/share/f$i; done\n"
        )
        fake.chmod(0o755)
        image = tmp_path / "x.sqsh"
        image.write_bytes(b"x")
        r = subprocess.run(
            [
                "bash",
                "-c",
                f'set -euo pipefail; {cache._VALID_SQSH_FN}; sqsh_valid "$1"',
                "_",
                str(image),
            ],
            env={"PATH": f"{tmp_path}:/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        assert (r.returncode == 0) is valid, r.stderr

    @pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
    def test_is_valid_bash(self):
        r = subprocess.run(
            ["bash", "-n"], input=self._script(), text=True, capture_output=True
        )
        assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# resolve_cached_image against a scripted login node
# ---------------------------------------------------------------------------
class FakeNode:
    """Stands in for ``_LoginNode``: answers the commands resolve issues."""

    def __init__(
        self,
        cached: bool = False,
        queued: Optional[str] = None,
        import_fills: bool = True,
        sbatch_rc: int = 0,
        squeue_polls: int = 1,
        ssh_down: bool = False,
    ) -> None:
        self.cached = cached
        self.queued = queued
        self.import_fills = import_fills
        self.sbatch_rc = sbatch_rc
        self.squeue_polls = squeue_polls
        self.ssh_down = ssh_down
        self.calls: List[str] = []
        self.scripts: List[str] = []

    async def run(
        self, command: str, stdin: Optional[str] = None
    ) -> Tuple[int, str, str]:
        self.calls.append(command)
        await asyncio.sleep(0)
        if self.ssh_down:
            raise RuntimeError("ssh bluevela failed: Connection refused")
        if "sqsh_valid" in command:
            return (0 if self.cached else 1), "", ""
        if command.startswith("squeue -h -n"):
            return 0, (self.queued + "\n") if self.queued else "", ""
        if "sbatch --parsable" in command:
            self.scripts.append(stdin or "")
            if self.sbatch_rc:
                return self.sbatch_rc, "", "sbatch: error: invalid qos"
            return 0, "4242\n", ""
        if command.startswith("squeue -h -j"):
            if self.squeue_polls > 0:
                self.squeue_polls -= 1
                return 0, "RUNNING\n", ""
            self.cached = self.import_fills
            return 1, "", "slurm_load_jobs error: Invalid job id specified"
        if command.startswith("scancel"):
            return 0, "", ""
        raise AssertionError(f"unexpected command {command!r}")

    def count(self, prefix: str) -> int:
        return sum(1 for c in self.calls if prefix in c)


async def _resolve(node: FakeNode, image_id=f"docker:{REF}", **cfg_kw):
    with patch.object(cache, "_LoginNode", lambda *a, **k: node):
        return await resolve_cached_image(image_id, _cfg(**cfg_kw), "bluevela")


class TestResolve:
    @pytest.mark.asyncio
    async def test_hit_rewrites_to_the_sqsh(self):
        node = FakeNode(cached=True)
        assert await _resolve(node) == f"docker:{SQSH}"
        assert node.count("sbatch") == 0

    @pytest.mark.asyncio
    async def test_bare_ref_gets_the_docker_prefix(self):
        assert await _resolve(FakeNode(cached=True), image_id=REF) == f"docker:{SQSH}"

    @pytest.mark.asyncio
    async def test_nothing_to_cache_makes_no_ssh_call(self):
        node = FakeNode()
        assert await _resolve(node, image_id=f"docker:{SQSH}") == f"docker:{SQSH}"
        assert await _resolve(node, image_id=None) is None
        assert node.calls == []

    @pytest.mark.asyncio
    async def test_miss_submits_one_import_and_waits_for_it(self):
        node = FakeNode(squeue_polls=2)
        assert await _resolve(node) == f"docker:{SQSH}"
        assert node.count("sbatch --parsable") == 1
        assert node.count("squeue -h -j 4242") == 3
        assert "#SBATCH --partition=cpu" in node.scripts[0]

    @pytest.mark.asyncio
    async def test_waits_on_an_import_already_queued_instead_of_submitting(self):
        node = FakeNode(queued="999")
        assert await _resolve(node) == f"docker:{SQSH}"
        assert node.count("sbatch") == 0
        assert node.count("squeue -h -j 999") >= 1

    @pytest.mark.asyncio
    async def test_import_that_leaves_no_valid_file_falls_back(self):
        node = FakeNode(import_fills=False)
        assert await _resolve(node) == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_sbatch_rejection_falls_back(self):
        node = FakeNode(sbatch_rc=1)
        assert await _resolve(node) == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_ssh_failure_falls_back(self):
        assert await _resolve(FakeNode(ssh_down=True)) == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_timeout_cancels_the_import_and_falls_back(self):
        node = FakeNode(squeue_polls=10**6)
        assert await _resolve(node, timeout_seconds=0) == f"docker:{REF}"
        assert node.count("scancel 4242") == 1

    @pytest.mark.asyncio
    async def test_poll_ssh_blip_is_retried(self):
        node = FakeNode(squeue_polls=1)
        real_run = node.run
        blips = {"n": 1}

        async def flaky(command, stdin=None):
            if command.startswith("squeue -h -j") and blips["n"]:
                blips["n"] -= 1
                raise RuntimeError("ssh bluevela timed out")
            return await real_run(command, stdin)

        node.run = flaky  # type: ignore[method-assign]
        assert await _resolve(node) == f"docker:{SQSH}"

    @pytest.mark.asyncio
    async def test_waiters_on_a_failed_import_fall_back_without_resubmitting(self):
        node = FakeNode(import_fills=False, squeue_polls=2)
        results = await asyncio.gather(*(_resolve(node) for _ in range(3)))
        assert results == [f"docker:{REF}"] * 3
        assert node.count("sbatch --parsable") == 1

    @pytest.mark.asyncio
    async def test_a_later_launch_retries_after_a_failure(self):
        node = FakeNode(import_fills=False)
        assert await _resolve(node) == f"docker:{REF}"
        node.import_fills = True
        node.squeue_polls = 1
        assert await _resolve(node) == f"docker:{SQSH}"
        assert node.count("sbatch --parsable") == 2

    @pytest.mark.asyncio
    async def test_concurrent_launches_of_one_image_submit_one_import(self):
        node = FakeNode(squeue_polls=3)
        results = await asyncio.gather(*(_resolve(node) for _ in range(5)))
        assert results == [f"docker:{SQSH}"] * 5
        assert node.count("sbatch --parsable") == 1


# ---------------------------------------------------------------------------
# Wiring into Skypilot.launch_skypilot
# ---------------------------------------------------------------------------
_SLURM_ENV = {
    "default_cloud": "slurm",
    "cluster": "bluevela",
    "zone": "gpu-mid",
    "slurm_image_cache": {"share_path": SHARE, "partition": "cpu"},
}
_LAUNCH = {"launcher_config": {"image_id": f"docker:{REF}"}, "config": {}}


class TestLaunchWiring:
    async def _launch(self, env_cfg, resolved=f"docker:{SQSH}", **launch):
        env = _make_env(env_cfg)
        resolver = AsyncMock(return_value=resolved)
        with (
            patch(
                "gbserver.environment._slurm_image_cache.resolve_cached_image",
                resolver,
            ),
            patch.object(env, "_prepare_ssh_for_launch", AsyncMock(return_value=None)),
        ):
            kwargs = await _launch_and_get_resources(env, "l1", **(launch or _LAUNCH))
        return kwargs, resolver

    @pytest.mark.asyncio
    async def test_slurm_launch_uses_the_cached_sqsh(self):
        kwargs, resolver = await self._launch(_SLURM_ENV)
        assert kwargs["image_id"] == f"docker:{SQSH}"
        image_id, cfg, alias = resolver.await_args.args
        assert image_id == f"docker:{REF}"
        assert cfg.partition == "cpu"
        assert alias == "bluevela"

    @pytest.mark.asyncio
    async def test_build_layer_image_is_the_one_resolved(self):
        launch = {
            "launcher_config": {"image_id": "docker:step/default:1"},
            "config": {"launcher_config": {"image_id": f"docker:{REF}"}},
        }
        _, resolver = await self._launch(_SLURM_ENV, **launch)
        assert resolver.await_args.args[0] == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_fallback_keeps_the_step_image(self):
        kwargs, _ = await self._launch(_SLURM_ENV, resolved=f"docker:{REF}")
        assert kwargs["image_id"] == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_no_cache_block_means_no_lookup(self):
        env = {k: v for k, v in _SLURM_ENV.items() if k != "slurm_image_cache"}
        kwargs, resolver = await self._launch(env)
        assert resolver.await_count == 0
        assert kwargs["image_id"] == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_non_slurm_cloud_is_untouched(self):
        env = {**_SLURM_ENV, "default_cloud": "k8s"}
        env.pop("cluster")
        env.pop("zone")
        kwargs, resolver = await self._launch(env)
        assert resolver.await_count == 0
        assert kwargs["image_id"] == f"docker:{REF}"

    @pytest.mark.asyncio
    async def test_bare_node_step_is_untouched(self):
        kwargs, resolver = await self._launch(_SLURM_ENV, launcher_config={}, config={})
        assert resolver.await_count == 0
        assert kwargs["image_id"] is None

    @pytest.mark.asyncio
    async def test_malformed_block_skips_the_cache_without_failing(self):
        env = {**_SLURM_ENV, "slurm_image_cache": {"partition": "cpu"}}
        kwargs, resolver = await self._launch(env)
        assert resolver.await_count == 0
        assert kwargs["image_id"] == f"docker:{REF}"
