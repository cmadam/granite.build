"""Unit tests for scripts/repair_export_rope_theta.py.

The script mutates PUBLISHED model directories from completed builds -- 38 of them in the
GOLD distillation epic -- so its walk, its dry-run default and its provenance trail are
worth testing before it is pointed at /proj.

The normalisation itself is NOT re-tested here. The script imports it from the
`distill-hf-export` step precisely so the repair and the export cannot disagree about what
a correct config is, and that logic has its own suite under
steps/distill-hf-export/skypilot/test/. These tests inject a stub normaliser so they run
without the upstream `gb_steps_post_training` package the step imports at module scope.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "repair_export_rope_theta.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("repair_export_rope_theta", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _hoist(raw):
    """Stub standing in for the step's normalise_model_config."""
    out = dict(raw)
    changes = []
    nested = (raw.get("rope_parameters") or {}).get("rope_theta")
    if nested is not None and "rope_theta" not in out:
        out["rope_theta"] = nested
        changes.append(f"hoisted rope_parameters.rope_theta -> top-level {nested}")
    if out.get("use_cache") is False:
        out["use_cache"] = True
        changes.append("use_cache False -> True")
    return out, changes


def _noop_assert(cfg):
    return None


def _model_dir(root: Path, name: str, cfg: dict, *, weights=True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if weights:
        (d / "model.safetensors").write_text("w")
    return d


BROKEN = {
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "use_cache": False,
}
CLEAN = {"rope_theta": 10000000, "use_cache": True}


# ------------------------------------------------------------------ the walk


def test_finds_plain_export_dirs_not_just_export_N(tmp_path):
    """THE SCOPE BUG IN THE ORIGINAL SCAN. A glob of `*/*/export-*/config.json` misses a
    directory named plain `export`, which is how the 8,150-step stage-1 run -- the one the
    whole v2 programme's founding conclusion came from -- was reported as 'not found'.
    """
    _model_dir(tmp_path, "run-a/build-1/export-25", BROKEN)
    _model_dir(tmp_path, "run-b/build-2/export", BROKEN)
    found = {p.name for p in mod.find_model_dirs(tmp_path)}
    assert found == {"export-25", "export"}


def test_finds_nested_align_dirs(tmp_path):
    """align/retagged_student sits one level deeper than an export dir, and it is the
    reference the repair is checked against, so the walk must reach it."""
    _model_dir(tmp_path, "run-a/build-1/align/retagged_student", CLEAN)
    assert [p.name for p in mod.find_model_dirs(tmp_path)] == ["retagged_student"]


def test_ignores_a_dir_with_config_but_no_weights(tmp_path):
    """A checkpoint's parent, a corpus dir, a stray config: not a published model."""
    d = tmp_path / "run-a/build-1/notamodel"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert mod.find_model_dirs(tmp_path) == []


def test_finds_sharded_weights(tmp_path):
    d = tmp_path / "run-a/build-1/export-1"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(BROKEN))
    (d / "model-00001-of-00002.safetensors").write_text("w")
    assert [p.name for p in mod.find_model_dirs(tmp_path)] == ["export-1"]


# ------------------------------------------------------------------ dry run


def test_dry_run_changes_nothing_on_disk(tmp_path):
    """The default. Pointing this at 38 published artifacts and having it report before it
    writes is the whole reason --apply is opt-in."""
    d = _model_dir(tmp_path, "run/build/export-1", BROKEN)
    before = (d / "config.json").read_bytes()
    rec = mod.repair_dir(d, _hoist, _noop_assert, apply=False)
    assert rec["action"] == "would-repair"
    assert (d / "config.json").read_bytes() == before
    assert not (d / "config.json.bak").exists()
    assert not (d / "config_repair.json").exists()


# ------------------------------------------------------------------ apply


def test_apply_repairs_and_keeps_provenance(tmp_path):
    d = _model_dir(tmp_path, "run/build/export-1", BROKEN)
    original = (d / "config.json").read_bytes()
    rec = mod.repair_dir(d, _hoist, _noop_assert, apply=True)

    assert rec["action"] == "repaired"
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["rope_theta"] == 10000000
    assert cfg["use_cache"] is True
    # The nested dict stays: transformers 5 reads it.
    assert cfg["rope_parameters"]["rope_theta"] == 10000000
    # Silently mutating a completed build's registered artifact destroys the audit trail.
    assert (d / "config.json.bak").read_bytes() == original
    side = json.loads((d / "config_repair.json").read_text())
    assert side["original"] == "config.json.bak"
    assert side["changed"]
    assert "rope_theta" in " ".join(side["changed"])
    assert side["repaired_at"]


def test_apply_is_idempotent(tmp_path):
    """Re-running over a repaired tree must report 'clean', not repair it twice -- a second
    backup would overwrite the real original with the already-repaired file."""
    d = _model_dir(tmp_path, "run/build/export-1", BROKEN)
    mod.repair_dir(d, _hoist, _noop_assert, apply=True)
    original = (d / "config.json.bak").read_bytes()
    rec = mod.repair_dir(d, _hoist, _noop_assert, apply=True)
    assert rec["action"] == "clean"
    assert (d / "config.json.bak").read_bytes() == original


def test_refuses_to_overwrite_an_existing_backup(tmp_path):
    """If a .bak is already there but the config still needs repair, something else touched
    this directory. Stop rather than destroy the only copy of the original."""
    d = _model_dir(tmp_path, "run/build/export-1", BROKEN)
    (d / "config.json.bak").write_text('{"someone": "else"}')
    rec = mod.repair_dir(d, _hoist, _noop_assert, apply=True)
    assert rec["action"] == "refused"
    assert "config.json.bak" in rec["detail"]
    assert json.loads((d / "config.json").read_text()) == BROKEN


def test_already_clean_dir_is_left_byte_identical(tmp_path):
    d = _model_dir(tmp_path, "run/build/align/retagged_student", CLEAN)
    before = (d / "config.json").read_bytes()
    rec = mod.repair_dir(d, _hoist, _noop_assert, apply=True)
    assert rec["action"] == "clean"
    assert (d / "config.json").read_bytes() == before


def test_apply_fails_loudly_when_the_result_is_not_portable(tmp_path):
    """The repair is checked with the same assertion the export step publishes under. If it
    does not hold, the original must still be on disk."""
    d = _model_dir(tmp_path, "run/build/export-1", BROKEN)
    original = (d / "config.json").read_bytes()

    def bad_assert(cfg):
        raise ValueError("still not portable")

    rec = mod.repair_dir(d, _hoist, bad_assert, apply=True)
    assert rec["action"] == "failed"
    assert (d / "config.json").read_bytes() == original
