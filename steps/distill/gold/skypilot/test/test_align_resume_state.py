"""Tests for the resume-state rewrite (../src/align_resume_state.py).

Run from the step directory with `make test`.

The template tests (test_step_template.py) run it inside the seeding block; these
pin its own rules: which checkpoint it rewrites, how a fraction resolves, and which
values it refuses.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "align_resume_state.py"


def _checkpoint(root, step, save_steps=1000, logging_steps=1, max_steps=8150):
    """checkpoint-<step> under root, holding only a trainer_state.json."""
    ckpt = root / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    state = dict(global_step=step, max_steps=max_steps, save_steps=save_steps)
    (ckpt / "trainer_state.json").write_text(
        json.dumps(dict(state, logging_steps=logging_steps))
    )
    return ckpt


def _state(ckpt):
    return json.loads((ckpt / "trainer_state.json").read_text())


def _run(root, save_steps="250", logging_steps="1", max_steps="0"):
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--checkpoint-dir",
            str(root),
            "--save-steps",
            str(save_steps),
            "--logging-steps",
            str(logging_steps),
            "--max-steps",
            str(max_steps),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_it_rewrites_the_checkpoint_the_trainer_resumes_from(tmp_path):
    """get_last_checkpoint's pick, by number, not by name and not by what was
    seeded: a checkpoint already in the dir outranks the source's highest."""
    seeded = _checkpoint(tmp_path, 2000)
    already_there = _checkpoint(tmp_path, 10000)
    (tmp_path / "checkpoint-99999.tmp").mkdir()  # not checkpoint-<N>
    (tmp_path / "checkpoint-50000").write_text("")  # not a directory

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert _state(already_there)["save_steps"] == 250
    assert _state(seeded)["save_steps"] == 1000
    assert "checkpoint-10000/trainer_state.json save_steps 1000 -> 250" in (
        result.stdout
    )


@pytest.mark.parametrize(
    "max_steps, expected",
    [("12000", 1200), ("0", 815), ("-1", 815)],
    ids=["this-run", "epoch-bounded", "hf-default"],
)
def test_a_fraction_is_a_share_of_max_steps(tmp_path, max_steps, expected):
    """As TrainerState.compute_steps: ceil(max_steps * fraction), with the
    checkpoint's 8150 when this run's max_steps is not positive."""
    ckpt = _checkpoint(tmp_path, 2000)
    result = _run(tmp_path, save_steps="0.1", max_steps=max_steps)
    assert result.returncode == 0, result.stderr
    assert _state(ckpt)["save_steps"] == expected


def test_the_file_is_replaced_not_edited(tmp_path):
    """The seeded state is a hardlink into the source run; the source keeps its grid."""
    ckpt = _checkpoint(tmp_path / "run", 2000)
    source = tmp_path / "source.json"
    source.hardlink_to(ckpt / "trainer_state.json")

    result = _run(tmp_path / "run")

    assert result.returncode == 0, result.stderr
    assert _state(ckpt)["save_steps"] == 250
    assert json.loads(source.read_text())["save_steps"] == 1000
    assert not (ckpt / "trainer_state.json.tmp").exists()


def test_a_matching_grid_is_left_alone(tmp_path):
    """Nothing to change: the file keeps its inode, so the hardlink survives."""
    ckpt = _checkpoint(tmp_path, 2000, save_steps=250)
    inode = (ckpt / "trainer_state.json").stat().st_ino
    result = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert (ckpt / "trainer_state.json").stat().st_ino == inode
    assert "already on this run's grid" in result.stdout


def test_a_read_only_checkpoint_dir_is_rewritten_and_keeps_its_mode(tmp_path):
    ckpt = _checkpoint(tmp_path, 2000)
    ckpt.chmod(0o555)
    try:
        result = _run(tmp_path)
        assert result.returncode == 0, result.stderr
        assert _state(ckpt)["save_steps"] == 250
        assert ckpt.stat().st_mode & 0o777 == 0o555
    finally:
        ckpt.chmod(0o755)


@pytest.mark.parametrize(
    "flag, value",
    [
        ("save_steps", "${GOLD_SAVE_STEPS}"),
        ("save_steps", "nan"),
        ("save_steps", "inf"),
        ("logging_steps", "0"),
        ("logging_steps", "-5"),
    ],
)
def test_a_bad_step_count_is_fatal_and_writes_nothing(tmp_path, flag, value):
    """Each one names itself in a FATAL line, never a bare traceback."""
    ckpt = _checkpoint(tmp_path, 2000)
    result = _run(tmp_path, **{flag: value})
    assert result.returncode == 1
    assert f"distill-gold: FATAL: {flag} must be" in result.stderr
    assert repr(value) in result.stderr
    assert "Traceback" not in result.stderr
    assert _state(ckpt)["save_steps"] == 1000


@pytest.mark.parametrize("value", ["${GOLD_MAX_STEPS}", "inf", "nan"])
def test_a_bad_max_steps_is_fatal(tmp_path, value):
    _checkpoint(tmp_path, 2000)
    result = _run(tmp_path, max_steps=value)
    assert result.returncode == 1
    assert "distill-gold: FATAL: max_steps must be an integer" in result.stderr
    assert "Traceback" not in result.stderr


def test_no_checkpoint_to_resume_from_is_fatal(tmp_path):
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "FATAL: no checkpoint-* directory" in result.stderr
