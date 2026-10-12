#!/usr/bin/env python3
"""Rewrite a seeded checkpoint's trainer_state.json to this run's save and logging grid.

HF Trainer computes ``save_steps`` and ``logging_steps`` from this run's arguments,
then, on resume, replaces the whole state with the checkpoint's ``trainer_state.json``.
The save callback reads the state, so a run seeded by ``resume_from_checkpoint_dir``
saves on the SOURCE run's grid; only a transformers warning that never reaches the
step's output says so. Build 4c758672 (A1) resumed a 250-step ladder from c2000's
1000-step grid and never wrote its 2250, 2500 or 3500 rungs.

The step's resume block runs this on rank 0, after seeding and before the trainer
starts. It rewrites the checkpoint the trainer will actually resume from: the
highest ``checkpoint-<N>`` directory in the run's checkpoint dir, by the rule
``transformers.trainer_utils.get_last_checkpoint`` applies (the trainer calls
``train(resume_from_checkpoint=True)``). That need not be the highest one seeded
from the source, when the dir already held checkpoints.

Values follow ``TrainerState.compute_steps``: a fraction below 1 is a share of this
run's ``max_steps``, or of the checkpoint's when ``max_steps`` is not positive (an
epoch-bounded run, whose schedule a resume must match anyway). ``eval_steps`` is left
alone: the step never enables evaluation.

The file is replaced, never edited in place: it is a hardlink into the source run's
checkpoint, which must keep its own grid. ``cp -a`` copies the source directory's
mode, so a read-only source leaves a read-only copy; the directory is made writable
for the replace and its mode restored after.
"""

import argparse
import json
import math
import os
import re
import stat
import sys

PREFIX = "distill-gold: resume:"
# transformers.trainer_utils._re_checkpoint
_CHECKPOINT = re.compile(r"^checkpoint\-(\d+)$")


def fatal(message):
    sys.exit(f"distill-gold: FATAL: {message}")


def step_count(name, raw):
    """A positive, finite number, as the trainer's argument parser would take it."""
    try:
        value = float(raw)
    except ValueError:
        fatal(f"{name} must be a number, got {raw!r}")
    if not math.isfinite(value) or value <= 0:
        fatal(f"{name} must be a positive, finite number, got {raw!r}")
    return value


def resumed_checkpoint(checkpoint_dir):
    """The checkpoint get_last_checkpoint picks: the highest-numbered directory."""
    steps = {}
    for name in os.listdir(checkpoint_dir):
        match = _CHECKPOINT.search(name)
        if match and os.path.isdir(os.path.join(checkpoint_dir, name)):
            steps[int(match.group(1))] = name
    if not steps:
        fatal(f"no checkpoint-* directory under {checkpoint_dir} to resume from")
    return os.path.join(checkpoint_dir, steps[max(steps)])


def replace_json(path, state):
    """Write state to path through a new inode, so a hardlinked original is untouched."""
    directory = os.path.dirname(path)
    mode = stat.S_IMODE(os.stat(directory).st_mode)
    try:
        if not mode & stat.S_IWUSR:
            os.chmod(directory, mode | stat.S_IWUSR)
        try:
            with open(path + ".tmp", "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            os.replace(path + ".tmp", path)
        finally:
            if not mode & stat.S_IWUSR:
                os.chmod(directory, mode)
    except OSError as e:
        fatal(f"cannot rewrite {path}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--save-steps", required=True)
    parser.add_argument("--logging-steps", required=True)
    parser.add_argument("--max-steps", required=True)
    args = parser.parse_args(argv)

    requested = {
        "save_steps": step_count("save_steps", args.save_steps),
        "logging_steps": step_count("logging_steps", args.logging_steps),
    }
    try:
        run_max_steps = int(float(args.max_steps))
    except (ValueError, OverflowError):
        fatal(f"max_steps must be an integer, got {args.max_steps!r}")

    checkpoint = resumed_checkpoint(args.checkpoint_dir)
    path = os.path.join(checkpoint, "trainer_state.json")
    with open(path) as f:
        state = json.load(f)
    total_steps = run_max_steps if run_max_steps > 0 else state["max_steps"]

    changed = []
    for key, value in requested.items():
        steps = math.ceil(total_steps * value) if value < 1 else int(value)
        if state.get(key) != steps:
            changed.append(f"{key} {state.get(key)} -> {steps}")
            state[key] = steps
    name = os.path.basename(checkpoint)
    if changed:
        replace_json(path, state)
        print(f"{PREFIX} {name}/trainer_state.json " + ", ".join(changed))
    else:
        print(f"{PREFIX} {name}/trainer_state.json already on this run's grid")


if __name__ == "__main__":
    main()
