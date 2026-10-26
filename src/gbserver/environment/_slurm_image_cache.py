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

"""Shared ``.sqsh`` cache for SLURM container images (issue #559).

Given a registry ``image_id``, SkyPilot's SLURM driver hands Pyxis
``registry#path`` and Pyxis runs a full ``enroot import`` inside the job's
allocation, then discards the result. On BlueVela that is ~22 minutes per
launch with the GPU idle, repeated for every launch of the same image. The
driver already passes an ``image_id`` ending in ``.sqsh`` straight to Pyxis,
so the fix is to make that file exist, once, on shared storage:

  * :func:`resolve_cached_image` checks over SSH (the materialized
    ``~/.slurm/config`` alias) whether ``<share_path>/enroot/<image>.sqsh``
    exists and is a flat root filesystem. On a hit it returns
    ``docker:<that path>`` for ``sky.Resources``.
  * On a miss it submits one short CPU job (the login nodes have no
    ``enroot``) that imports the image, then polls until the job leaves the
    queue and re-checks the file. Any failure falls back to the registry
    reference -- today's behaviour, slow but working.

Concurrency, in three layers:

  * inside gbserver, one :class:`asyncio.Lock` per cache file, so targets of
    one build launching the same image submit a single import job;
  * across gbserver processes, a running import job for the same file (found
    by its deterministic job name) is waited on instead of submitting another;
  * on the cluster, the job takes ``flock`` on ``<image>.sqsh.lock`` -- the
    lock file the SkyPilot LSF driver uses for the same file names -- and
    re-checks under it, so LSF and SLURM imports of one image serialize.

The job imports to a job-unique temporary file and renames it into place only
once it is valid, so a killed import never leaves a truncated ``.sqsh`` at the
final name. That also keeps the result correct if ``flock`` were not honoured
across nodes: two racing imports both rename a complete file.

The file naming matches the LSF driver (``us.icr.io/a/b:tag`` ->
``us.icr.io-a-b-tag.sqsh``), so images either backend imported are reused by
the other.
"""

import asyncio
import hashlib
import re
import shlex
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Characters allowed in an image reference / path that is interpolated into a
# remote shell script. Anything else is not a plausible registry ref, so the
# cache is skipped rather than quoting it into a script.
_IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9._/:@-]+$")
_SHARE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")

# A flat root filesystem has /bin/sh (or /usr/bin/sh when /bin is a symlink).
# A layered OCI sqsh (numbered layer dirs at the root) does not, and Pyxis
# cannot start it. `unsquashfs -l` reads only the directory table: ~0.15s for
# a 5 GB image on GPFS. Not `grep -q`: it exits at the first match, unsquashfs
# dies of SIGPIPE, and under `set -o pipefail` (the import script) a valid
# image then reads as invalid.
_VALID_SQSH_FN = (
    'sqsh_valid() { [ -s "$1" ] && unsquashfs -l "$1" 2>/dev/null '
    "| grep -E '^squashfs-root/(usr/)?bin/sh$' >/dev/null; }"
)

_SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "LogLevel=ERROR",
)

_DEFAULT_IMPORT_SBATCH: Dict[str, Any] = {
    # enroot's mksquashfs uses every CPU it is given; the squash dominates.
    "cpus-per-task": 16,
    "mem": "64G",
    "time": "02:00:00",
}


@dataclass(frozen=True)
class SlurmImageCacheConfig:
    """The environment's ``slurm_image_cache`` block.

    Attributes:
        share_path: Shared-filesystem root; files live in ``<share_path>/enroot``.
            Same meaning as the LSF driver's ``enroot.share_path``.
        partition: Partition the import job runs on (a CPU partition).
        sbatch_options: Extra ``#SBATCH`` directives for the import job (keys
            without the leading ``--``; ``None`` drops a default, ``True``
            emits a bare flag). Merged over cpus/mem/time defaults.
        timeout_seconds: How long to wait for an import (queueing included)
            before falling back to the registry reference.
        poll_interval_seconds: How often to poll the import job.
        ssh_timeout_seconds: Bound on each SSH command to the login node.
    """

    share_path: str
    partition: str
    sbatch_options: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 7200.0
    poll_interval_seconds: float = 30.0
    ssh_timeout_seconds: float = 120.0

    @property
    def cache_dir(self) -> str:
        return f"{self.share_path.rstrip('/')}/enroot"


def parse_cache_config(raw: Any) -> Optional[SlurmImageCacheConfig]:
    """Parse ``config.slurm_image_cache``; ``None`` when unset or disabled.

    :raises ValueError: on a malformed block, so a typo fails the launch
        loudly instead of silently disabling the cache.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"slurm_image_cache must be a mapping, got {raw!r}")
    if raw.get("enabled", True) is False:
        return None
    share_path = raw.get("share_path")
    partition = raw.get("partition")
    if not isinstance(share_path, str) or not _SHARE_PATH_RE.match(share_path):
        raise ValueError(
            f"slurm_image_cache.share_path must be an absolute path, got {share_path!r}"
        )
    if not isinstance(partition, str) or not partition:
        raise ValueError(f"slurm_image_cache.partition is required, got {partition!r}")
    sbatch_options = raw.get("sbatch_options") or {}
    if not isinstance(sbatch_options, dict):
        raise ValueError("slurm_image_cache.sbatch_options must be a mapping")
    kwargs = {
        k: float(raw[k])
        for k in ("timeout_seconds", "poll_interval_seconds", "ssh_timeout_seconds")
        if raw.get(k) is not None
    }
    return SlurmImageCacheConfig(
        share_path=share_path,
        partition=partition,
        sbatch_options=dict(sbatch_options),
        **kwargs,
    )


def split_image_ref(image_id: Optional[str]) -> Tuple[str, Optional[str]]:
    """Split a launcher ``image_id`` into its scheme prefix and registry ref.

    :returns: ``(prefix, ref)``. ``ref`` is ``None`` when there is nothing to
        cache: no image, an image that is already a ``.sqsh`` path, or a value
        that is not a plausible registry reference.
    """
    if not image_id:
        return "", None
    prefix = ""
    for scheme in ("docker://", "docker:"):
        if image_id.startswith(scheme):
            prefix, image_id = "docker:", image_id[len(scheme) :]
            break
    if image_id.endswith(".sqsh") or image_id.startswith("/"):
        return prefix, None
    if not _IMAGE_REF_RE.match(image_id):
        return prefix, None
    return prefix, image_id


def sqsh_name(image_ref: str) -> str:
    """The cache file name; identical to the SkyPilot LSF driver's
    ``_sqsh_name_from_image`` so both backends share one file per image."""
    return image_ref.replace("/", "-").replace(":", "-") + ".sqsh"


def enroot_uri(image_ref: str) -> str:
    """``registry/path`` -> ``registry#path``, the form ``enroot import`` needs
    for a non-Docker-Hub registry (a first segment with a dot, colon, or
    ``localhost``)."""
    registry, sep, rest = image_ref.partition("/")
    if sep and ("." in registry or ":" in registry or registry == "localhost"):
        return f"{registry}#{rest}"
    return image_ref


def import_job_name(sqsh_path: str) -> str:
    """Deterministic job name per cache file, so a second gbserver finds and
    waits on an import already queued instead of submitting another."""
    digest = hashlib.sha256(sqsh_path.encode()).hexdigest()[:12]
    return f"gb-image-import-{digest}"


def _sbatch_directives(cfg: SlurmImageCacheConfig, job_name: str) -> str:
    options: Dict[str, Any] = {
        **_DEFAULT_IMPORT_SBATCH,
        **cfg.sbatch_options,
        "partition": cfg.partition,
        "job-name": job_name,
        "nodes": 1,
        "ntasks": 1,
        "output": f"{cfg.cache_dir}/logs/%x-%j.out",
    }
    lines = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        key = str(key).lstrip("-")
        lines.append(
            f"#SBATCH --{key}" if value is True else f"#SBATCH --{key}={value}"
        )
    return "\n".join(lines)


def render_import_script(
    cfg: SlurmImageCacheConfig, image_ref: str, sqsh_path: str
) -> str:
    """The sbatch script that imports ``image_ref`` into ``sqsh_path``.

    Takes the shared lock, re-checks under it, imports to node-local scratch
    when available, validates, copies to a job-unique temporary name beside
    the target and renames it into place.
    """
    q_sqsh = shlex.quote(sqsh_path)
    q_uri = shlex.quote(f"docker://{enroot_uri(image_ref)}")
    return f"""#!/bin/bash
{_sbatch_directives(cfg, import_job_name(sqsh_path))}
# Generated by gbserver (_slurm_image_cache.py): import a container image once
# into the shared .sqsh cache. See issue #559.
set -euo pipefail
{_VALID_SQSH_FN}
SQSH={q_sqsh}
DIR=$(dirname "$SQSH")
mkdir -p "$DIR"

# Same lock file as the SkyPilot LSF driver, so LSF and SLURM imports of one
# image serialize. Re-check once held: another job may have just finished it.
exec 200>"$SQSH.lock"
echo "[$(date)] waiting for $SQSH.lock"
flock -x 200
if sqsh_valid "$SQSH"; then
    echo "[$(date)] $SQSH already cached ($(du -h "$SQSH" | cut -f1)); nothing to do"
    exit 0
fi

# Job-unique names: a killed job never leaves a partial file at the final name.
FINAL_TMP="$DIR/.$(basename "$SQSH" .sqsh).tmp-$SLURM_JOB_ID.sqsh"
SCRATCH="$DIR"
if [ -d "/opt/nvme/$USER" ] && [ -w "/opt/nvme/$USER" ]; then
    SCRATCH="/opt/nvme/$USER/gb-image-cache"
    mkdir -p "$SCRATCH"
fi
LOCAL_TMP="$SCRATCH/.$(basename "$SQSH" .sqsh).import-$SLURM_JOB_ID.sqsh"
trap 'rm -f "$LOCAL_TMP" "$FINAL_TMP"' EXIT

echo "[$(date)] importing {image_ref} on $(hostname) into $LOCAL_TMP"
enroot import -o "$LOCAL_TMP" {q_uri}
if ! sqsh_valid "$LOCAL_TMP"; then
    echo "[$(date)] ERROR: imported image has no /bin/sh (layered or broken); not caching it" >&2
    exit 1
fi
if [ "$LOCAL_TMP" != "$FINAL_TMP" ]; then
    echo "[$(date)] copying to the shared filesystem"
    cp "$LOCAL_TMP" "$FINAL_TMP"
fi
chmod g+rw,o+r "$FINAL_TMP" 2>/dev/null || true
mv -f "$FINAL_TMP" "$SQSH"
echo "[$(date)] cached $SQSH ($(du -h "$SQSH" | cut -f1))"
"""


class _LoginNode:  # pylint: disable=too-few-public-methods
    """Runs commands on a SLURM login node through the ``~/.slurm/config``
    alias gbserver materializes (``ssh -F``)."""

    def __init__(self, alias: str, ssh_config: Path, timeout: float) -> None:
        self._alias = alias
        self._ssh_config = ssh_config
        self._timeout = timeout

    async def run(
        self, command: str, stdin: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """Run ``command``; return ``(returncode, stdout, stderr)``.

        :raises RuntimeError: on an SSH-level failure (rc 255) or a timeout.
        """
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-F",
            str(self._ssh_config),
            *_SSH_OPTIONS,
            self._alias,
            command,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin.encode() if stdin is not None else None),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"ssh {self._alias} timed out after {self._timeout}s"
            ) from e
        if proc.returncode == 255:
            raise RuntimeError(
                f"ssh {self._alias} failed: {err.decode(errors='replace').strip()}"
            )
        return (
            proc.returncode or 0,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )


async def _is_cached(node: _LoginNode, sqsh_path: str) -> bool:
    rc, _, _ = await node.run(f"{_VALID_SQSH_FN}; sqsh_valid {shlex.quote(sqsh_path)}")
    return rc == 0


async def _queued_job_id(node: _LoginNode, job_name: str) -> Optional[str]:
    rc, out, err = await node.run(f"squeue -h -n {shlex.quote(job_name)} -o %i")
    if rc != 0:
        raise RuntimeError(f"squeue failed (rc={rc}): {err.strip()}")
    ids = out.split()
    return ids[0] if ids else None


async def _submit(
    node: _LoginNode, cfg: SlurmImageCacheConfig, image_ref: str, sqsh_path: str
) -> str:
    script = render_import_script(cfg, image_ref, sqsh_path)
    logs = shlex.quote(f"{cfg.cache_dir}/logs")
    rc, out, err = await node.run(f"mkdir -p {logs} && sbatch --parsable", stdin=script)
    job_id = out.strip().split(";", 1)[0]
    if rc != 0 or not job_id.isdigit():
        raise RuntimeError(f"sbatch failed (rc={rc}): {(err or out).strip()}")
    return job_id


async def _wait_for_job(
    node: _LoginNode, job_id: str, cfg: SlurmImageCacheConfig, deadline: float
) -> None:
    """Poll until ``job_id`` leaves the queue; on timeout cancel it and raise.

    ``squeue -j`` of a job that has left the queue prints nothing, or (once
    slurmctld has purged it) exits 1 with "Invalid job id". Any other failure
    -- SSH, a slurmctld hiccup -- is logged and retried: the job runs on its
    own and the deadline still bounds the wait.
    """
    while True:
        try:
            rc, out, err = await node.run(f"squeue -h -j {job_id} -o %T")
            if rc == 0 and not out.strip():
                return
            if rc != 0 and "invalid job id" in err.lower():
                return
            if rc != 0:
                raise RuntimeError(f"squeue failed (rc={rc}): {err.strip()}")
        except RuntimeError as e:
            logger.warning("Polling image import job %s failed: %s", job_id, e)
        if time.monotonic() >= deadline:
            try:
                await node.run(f"scancel {job_id}")
            except RuntimeError:
                pass
            raise RuntimeError(
                f"image import job {job_id} did not finish in {cfg.timeout_seconds:.0f}s"
            )
        await asyncio.sleep(cfg.poll_interval_seconds)


@dataclass
class _Slot:
    """Per-(event loop, cache file) state: the lock that serializes resolves
    of one file, and the outcome of the last import attempt made under it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    attempts: int = 0
    last_failed: bool = False


# Keyed by event loop too: asyncio locks bind to the loop they first run on.
_SLOTS: Dict[Tuple[int, str], _Slot] = {}
_SLOTS_GUARD = threading.Lock()


def _slot_for(sqsh_path: str) -> _Slot:
    key = (id(asyncio.get_running_loop()), sqsh_path)
    with _SLOTS_GUARD:
        slot = _SLOTS.get(key)
        if slot is None:
            slot = _SLOTS[key] = _Slot()
        return slot


async def resolve_cached_image(
    image_id: Optional[str],
    cfg: SlurmImageCacheConfig,
    cluster_alias: str,
    ssh_config: Optional[Path] = None,
) -> Optional[str]:
    """Return the ``image_id`` to launch with: the cached ``.sqsh`` when
    available (importing it first on a miss), else ``image_id`` unchanged.

    Never raises: every failure is logged and falls back to ``image_id``.

    :param image_id: the step's resolved ``image_id`` (``docker:<ref>``, a
        bare ref, a ``.sqsh`` path, or ``None``).
    :param cfg: the parsed ``slurm_image_cache`` block.
    :param cluster_alias: the ``~/.slurm/config`` ``Host`` alias (the infra's
        cluster segment).
    :param ssh_config: the SSH config file (default ``~/.slurm/config``).
    """
    prefix, ref = split_image_ref(image_id)
    if ref is None:
        return image_id
    sqsh_path = f"{cfg.cache_dir}/{sqsh_name(ref)}"
    cached = f"{prefix or 'docker:'}{sqsh_path}"
    node = _LoginNode(
        cluster_alias,
        ssh_config or Path.home() / ".slurm" / "config",
        cfg.ssh_timeout_seconds,
    )
    started = time.monotonic()
    slot = _slot_for(sqsh_path)
    attempts_seen = slot.attempts
    try:
        async with slot.lock:
            if await _is_cached(node, sqsh_path):
                logger.info("SLURM image cache hit: %s -> %s", ref, sqsh_path)
                return cached
            # An import ran and failed while this launch waited for the lock:
            # fall back with it rather than each waiter submitting another
            # import of an image that just failed. The next launch retries.
            if slot.attempts != attempts_seen and slot.last_failed:
                raise RuntimeError(
                    "the import attempted for a concurrent launch failed"
                )
            slot.attempts += 1
            slot.last_failed = True
            job_name = import_job_name(sqsh_path)
            job_id = await _queued_job_id(node, job_name)
            if job_id:
                logger.info(
                    "SLURM image cache miss for %s: waiting on import job %s "
                    "already queued as %s",
                    ref,
                    job_id,
                    job_name,
                )
            else:
                job_id = await _submit(node, cfg, ref, sqsh_path)
                logger.info(
                    "SLURM image cache miss for %s: submitted import job %s on "
                    "partition %s (log: %s/logs/%s-%s.out)",
                    ref,
                    job_id,
                    cfg.partition,
                    cfg.cache_dir,
                    job_name,
                    job_id,
                )
            await _wait_for_job(node, job_id, cfg, started + cfg.timeout_seconds)
            if not await _is_cached(node, sqsh_path):
                raise RuntimeError(
                    f"import job {job_id} finished but {sqsh_path} is not a valid "
                    f"image; see {cfg.cache_dir}/logs/{job_name}-{job_id}.out"
                )
            slot.last_failed = False
            logger.info(
                "SLURM image cache filled: %s -> %s in %.0fs",
                ref,
                sqsh_path,
                time.monotonic() - started,
            )
            return cached
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "SLURM image cache unavailable for %s (%s); launching from the "
            "registry instead, which imports the image inside the job",
            ref,
            e,
        )
        return image_id
