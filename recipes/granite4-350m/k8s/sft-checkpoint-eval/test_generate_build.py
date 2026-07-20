"""Tests for the k8s sft-checkpoint-eval generator.

For now this file holds catalog-load assertions for eval-catalog.yaml (the
single source of truth for the 27 per-epoch checkpoint evals). Later tasks
extend it with generate_build.py assertions.
"""
import importlib.util
import re
from pathlib import Path

import pytest
import yaml


def _load_generator():
    path = Path(__file__).parent / "generate_build.py"
    spec = importlib.util.spec_from_file_location("gen_build_k8s", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gb = _load_generator()

_HERE = Path(__file__).parent
_CATALOG = _HERE / "eval-catalog.yaml"
# The RL catalog is the canonical source for eval names, categories, and named
# sets; the k8s catalog must stay in lockstep with it.
_RL_CATALOG = (
    _HERE.parents[1]
    / "lsf"
    / "rl-checkpoint-eval"
    / "eval-catalog.yaml"
)

_CATEGORIES = {"code", "general", "math", "safety", "multilingual", "bfcl"}
_IMAGES = {
    "SAGE_OLMES_IMAGE",
    "SAGE_CODE_IMAGE",
    "SAGE_SAFETY_IMAGE",
    "SAGE_MULTILINGUAL_IMAGE",
    "BFCL_IMAGE",
}
_SET_TO_CATEGORY = {
    "code-eval": "code",
    "general-eval": "general",
    "math-eval": "math",
    "safety-eval": "safety",
    "multilingual-eval": "multilingual",
}
# Categories with a single fixed image (code/general/math all share
# SAGE_OLMES_IMAGE for olmes evals but code also uses SAGE_CODE_IMAGE, so only
# these are 1:1).
_CATEGORY_TO_IMAGE = {
    "safety": "SAGE_SAFETY_IMAGE",
    "multilingual": "SAGE_MULTILINGUAL_IMAGE",
    "bfcl": "BFCL_IMAGE",
}
_SCRIPT_RE = re.compile(r"^[a-z0-9_]+\.sh$")


@pytest.fixture(scope="module")
def catalog():
    with open(_CATALOG) as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def rl_catalog():
    with open(_RL_CATALOG) as fh:
        return yaml.safe_load(fh)


def test_exactly_27_evals_with_valid_categories(catalog):
    evals = catalog["evals"]
    assert len(evals) == 27, f"expected 27 evals, got {len(evals)}"
    for name, meta in evals.items():
        assert meta.get("category") in _CATEGORIES, (
            f"{name} has invalid category {meta.get('category')!r}"
        )


def test_sage_evals_have_script_and_image_bfcl_does_not(catalog):
    evals = catalog["evals"]
    for name, meta in evals.items():
        if meta["category"] == "bfcl":
            assert "script" not in meta, f"{name} (bfcl) must not have a script"
        else:
            assert meta.get("script"), f"{name} missing non-empty script"
            assert meta.get("image"), f"{name} missing non-empty image"
    # bfcl is the sole bfcl-category eval.
    assert evals["bfcl"]["category"] == "bfcl"
    assert "script" not in evals["bfcl"]


def test_every_image_is_known(catalog):
    for name, meta in catalog["evals"].items():
        assert meta.get("image") in _IMAGES, (
            f"{name} has unknown image {meta.get('image')!r}"
        )


def test_sage_scripts_are_wellformed_and_not_slim(catalog):
    for name, meta in catalog["evals"].items():
        if meta["category"] == "bfcl":
            continue
        script = meta["script"]
        assert _SCRIPT_RE.match(script), f"{name} script {script!r} malformed"
        assert not script.endswith("_slim.sh"), (
            f"{name} uses a _slim script {script!r}; k8s needs the non-slim name"
        )


def test_non_bfcl_evals_have_log_suffix(catalog):
    for name, meta in catalog["evals"].items():
        if meta["category"] == "bfcl":
            continue
        assert meta.get("log_suffix"), f"{name} missing non-empty log_suffix"


def test_full_eval_set_contains_all_27(catalog):
    evals = catalog["evals"]
    full = catalog["sets"]["full-eval"]
    assert len(full) == 27
    assert set(full) == set(evals.keys())
    assert len(set(full)) == len(full), "full-eval has duplicates"


def test_set_members_exist_in_evals(catalog):
    evals = catalog["evals"]
    for set_name, members in catalog["sets"].items():
        for m in members:
            assert m in evals, f"set {set_name} references unknown eval {m}"


def test_category_sets_group_the_right_categories(catalog):
    evals = catalog["evals"]
    for set_name, category in _SET_TO_CATEGORY.items():
        for m in catalog["sets"][set_name]:
            assert evals[m]["category"] == category, (
                f"{m} in {set_name} has category {evals[m]['category']!r}, "
                f"expected {category!r}"
            )
    assert catalog["sets"]["bfcl"] == ["bfcl"]


def test_category_image_consistency(catalog):
    for name, meta in catalog["evals"].items():
        expected = _CATEGORY_TO_IMAGE.get(meta["category"])
        if expected is not None:
            assert meta["image"] == expected, (
                f"{name} ({meta['category']}) image {meta['image']!r} != "
                f"expected {expected!r}"
            )


def test_names_match_rl_catalog(catalog, rl_catalog):
    assert set(catalog["evals"].keys()) == set(rl_catalog["evals"].keys())


def test_categories_match_rl_catalog(catalog, rl_catalog):
    for name, meta in catalog["evals"].items():
        assert meta["category"] == rl_catalog["evals"][name]["category"], (
            f"{name} category {meta['category']!r} != "
            f"RL {rl_catalog['evals'][name]['category']!r}"
        )


def test_set_membership_matches_rl_catalog(catalog, rl_catalog):
    assert set(catalog["sets"].keys()) == set(rl_catalog["sets"].keys())
    for set_name, members in catalog["sets"].items():
        assert members == rl_catalog["sets"][set_name], (
            f"set {set_name} differs from RL catalog"
        )


# ─── generate_build.py: compute_eval_epochs ──────────────────────────────────
def test_eval_epochs_all_expands_to_full_range():
    assert gb.compute_eval_epochs({"NUM_EPOCHS": 4}) == [1, 2, 3, 4]
    assert gb.compute_eval_epochs({"NUM_EPOCHS": 4, "EVAL_EPOCHS": "all"}) == [
        1, 2, 3, 4
    ]


def test_eval_epochs_csv_string():
    assert gb.compute_eval_epochs({"NUM_EPOCHS": 5, "EVAL_EPOCHS": "2,4"}) == [2, 4]


def test_eval_epochs_list_dedups_and_sorts():
    assert gb.compute_eval_epochs({"NUM_EPOCHS": 5, "EVAL_EPOCHS": [3, 1, 3]}) == [
        1, 3
    ]


def test_eval_epochs_single_int():
    assert gb.compute_eval_epochs({"NUM_EPOCHS": 5, "EVAL_EPOCHS": 3}) == [3]


def test_eval_epochs_out_of_range_raises():
    with pytest.raises(ValueError):
        gb.compute_eval_epochs({"NUM_EPOCHS": 3, "EVAL_EPOCHS": [4]})
    with pytest.raises(ValueError):
        gb.compute_eval_epochs({"NUM_EPOCHS": 3, "EVAL_EPOCHS": [0]})


def test_eval_epochs_num_epochs_below_one_raises():
    with pytest.raises(ValueError):
        gb.compute_eval_epochs({"NUM_EPOCHS": 0})


# ─── generate_build.py: resolve_eval_names ───────────────────────────────────
def test_resolve_eval_names_set_expansion(catalog):
    names = gb.resolve_eval_names({"EVAL_SETS": ["multilingual-eval"]}, catalog)
    assert names == catalog["sets"]["multilingual-eval"]


def test_resolve_eval_names_individual_and_dedup(catalog):
    names = gb.resolve_eval_names(
        {"EVAL_SETS": ["bfcl", "mgsm", "bfcl", "multilingual-eval"]}, catalog
    )
    # First-seen order preserved, de-duped (mgsm already pulled in before the set).
    assert names[0] == "bfcl"
    assert names[1] == "mgsm"
    assert len(names) == len(set(names))
    assert "global-mmlu" in names


def test_resolve_eval_names_unknown_raises(catalog):
    with pytest.raises(ValueError):
        gb.resolve_eval_names({"EVAL_SETS": ["no-such-eval"]}, catalog)


def test_resolve_eval_names_empty_raises(catalog):
    with pytest.raises(ValueError):
        gb.resolve_eval_names({"EVAL_SETS": []}, catalog)


# ─── generate_build.py: training target ──────────────────────────────────────
def test_training_target_one_output_per_epoch():
    tgt = gb.build_training_target([1, 2, 3])
    outs = tgt["outputs"]
    assert set(outs) == {"epoch_1", "epoch_2", "epoch_3"}
    for name, spec in outs.items():
        assert spec["type"] == "model"
    # Per-epoch-unique HF uris.
    uris = {spec["uri"] for spec in outs.values()}
    assert len(uris) == 3
    # Step is the forked open-instruct on the gbspace-config-dev branch.
    assert "open-instruct" in tgt["steps"][0]["step_uri"]
    assert "gbspace-config-dev" in tgt["steps"][0]["step_uri"]


def test_tokenize_target_shape():
    tgt = gb.build_tokenize_target()
    cmd = tgt["steps"][0]["config"]["k8s"]["additional_files"]["/tmp/command.sh"]
    assert "run_data_prep_v2.sh" in cmd
    assert "LLMB_ARTIFACT_ID:tokenized_dataset" in cmd
    assert tgt["steps"][0]["config"]["compute_config"]["num_gpus_per_node"] == 0


# ─── generate_build.py: eval targets ─────────────────────────────────────────
def _small_catalog(catalog):
    return catalog


def test_eval_fanout_counts_and_bindings(catalog):
    params = {
        "NUM_EPOCHS": 2,
        "EVAL_EPOCHS": "all",
        "EVAL_SETS": ["multilingual-eval", "bfcl"],
    }
    build, eval_epochs, eval_names = gb.generate(params, catalog)
    targets = build["granite.build"]["targets"]
    assert eval_epochs == [1, 2]
    assert len(eval_names) == 6  # 5 multilingual + bfcl
    eval_targets = [
        n for n in targets if re.search(r"-ep\d+$", n) and not n.startswith("export-")
    ]
    assert len(eval_targets) == 12
    # Every eval target binds sft-training.epoch_<m>.
    for name in eval_targets:
        m = int(name.rsplit("-ep", 1)[1])
        binding = targets[name]["inputs"]["model"]["binding"]
        assert binding == f"sft-training.epoch_{m}"


def test_sage_target_command_references_script_and_epoch(catalog):
    tgt = gb.build_sage_eval_target(catalog["evals"]["mgsm"], 2)
    cmd = tgt["steps"][0]["config"]["k8s"]["additional_files"]["/tmp/command.sh"]
    assert "multilingual_mgsm.sh" in cmd
    assert "-ep_2" in cmd
    assert "$${EXPERIMENT}-ep_2" in cmd
    assert tgt["steps"][0]["config"]["k8s"]["image"] == "$${SAGE_MULTILINGUAL_IMAGE}"


def test_sage_extra_env_overrides_and_exports(catalog):
    # multiple-sh overrides MAX_LENGTH=512 and exports MULTIPLE_LANG=sh.
    cmd = gb.build_sage_eval_target(catalog["evals"]["multiple-sh"], 1)["steps"][0][
        "config"
    ]["k8s"]["additional_files"]["/tmp/command.sh"]
    assert "export MAX_LENGTH=512" in cmd
    assert "export MULTIPLE_LANG=sh" in cmd
    # ifeval hardcodes BATCH_SIZE=30.
    cmd2 = gb.build_sage_eval_target(catalog["evals"]["olmes-ifeval"], 1)["steps"][0][
        "config"
    ]["k8s"]["additional_files"]["/tmp/command.sh"]
    assert "export BATCH_SIZE=30" in cmd2


def test_bfcl_target_image_and_binding(catalog):
    tgt = gb.build_bfcl_eval_target(catalog["evals"]["bfcl"], 2)
    assert tgt["steps"][0]["config"]["k8s"]["image"] == "$${BFCL_IMAGE}"
    assert tgt["inputs"]["model"]["binding"] == "sft-training.epoch_2"
    cmd = tgt["steps"][0]["config"]["k8s"]["additional_files"]["/tmp/command.sh"]
    assert "run-bfcl.sh" in cmd
    assert "$${EXPERIMENT}-ep_2" in cmd
    assert "LLMB_ARTIFACT_ID:bfcl_results" in cmd


# ─── generate_build.py: export targets ───────────────────────────────────────
def test_epoch_exports_gate_on_their_evals(catalog):
    params = {
        "NUM_EPOCHS": 2,
        "EVAL_EPOCHS": "all",
        "EVAL_SETS": ["multilingual-eval", "bfcl"],
    }
    build, _, _ = gb.generate(params, catalog)
    targets = build["granite.build"]["targets"]
    for m in (1, 2):
        exp = targets[f"export-ep{m}"]
        # Gates on all 6 of this epoch's eval targets, each wait_for_push.
        gated = exp["inputs"]
        assert len(gated) == 6
        for spec in gated.values():
            assert spec["wait_for_push"] is True
            assert spec["binding"].endswith((f"eval_log", "bfcl_results"))
            assert f"-ep{m}." in spec["binding"]
        cmd = exp["steps"][0]["config"]["k8s"]["additional_files"]["/tmp/command.sh"]
        assert f"$${{EXPERIMENT}}-ep_{m}" in cmd
        assert "LLMB_ARTIFACT_ID:epoch_csv" in cmd


def test_combined_export_gates_and_lists_all_folders(catalog):
    params = {
        "NUM_EPOCHS": 2,
        "EVAL_EPOCHS": "all",
        "EVAL_SETS": ["multilingual-eval", "bfcl"],
    }
    build, _, _ = gb.generate(params, catalog)
    combined = build["granite.build"]["targets"]["export-combined"]
    gated = combined["inputs"]
    assert len(gated) == 2
    for spec in gated.values():
        assert spec["wait_for_push"] is True
        assert spec["binding"].endswith(".epoch_csv")
    cmd = combined["steps"][0]["config"]["k8s"]["additional_files"]["/tmp/command.sh"]
    assert "$${EXPERIMENT}-ep_1" in cmd
    assert "$${EXPERIMENT}-ep_2" in cmd
    assert "combined.csv" in cmd


def test_whole_build_roundtrips_and_target_count(catalog):
    params = {
        "NUM_EPOCHS": 2,
        "EVAL_EPOCHS": "all",
        "EVAL_SETS": ["multilingual-eval", "bfcl"],
    }
    build, eval_epochs, eval_names = gb.generate(params, catalog)
    dumped = yaml.safe_dump(build, sort_keys=False, default_flow_style=False)
    reloaded = yaml.safe_load(dumped)  # must not raise
    n_epochs, n_evals = len(eval_epochs), len(eval_names)
    expected = 2 + n_epochs * n_evals + n_epochs + 1
    assert len(reloaded["granite.build"]["targets"]) == expected
    # Command literals survive the round-trip byte-for-byte.
    orig = build["granite.build"]["targets"]["tokenize"]["steps"][0]["config"]["k8s"][
        "additional_files"
    ]["/tmp/command.sh"]
    rt = reloaded["granite.build"]["targets"]["tokenize"]["steps"][0]["config"]["k8s"][
        "additional_files"
    ]["/tmp/command.sh"]
    assert str(orig) == rt


# ─── generate_build.py: main() smoke ─────────────────────────────────────────
def test_main_smoke_stdout_and_keep_last_n(tmp_path, capsys):
    params_file = tmp_path / "parameters.yaml"
    params_file.write_text(
        yaml.safe_dump(
            {
                "NUM_EPOCHS": 2,
                "EXPERIMENT": "smoke-exp",
                "KEEP_LAST_N_CHECKPOINTS": 1,
            }
        )
    )
    rc = gb.main(
        [
            "--output", "-",
            "--num-epochs", "2",
            "--eval-sets", "bfcl",
            "--eval-epochs", "all",
            "--parameters-path", str(params_file),
            "--catalog-path", str(_CATALOG),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    doc = yaml.safe_load(out)  # valid YAML written to stdout
    assert "granite.build" in doc
    # 2 epochs x 1 eval (bfcl) + tokenize + training + 2 epoch-exports + combined.
    assert len(doc["granite.build"]["targets"]) == 2 + 2 * 1 + 2 + 1


def test_main_bumps_keep_last_n_in_resolved_params(tmp_path):
    params_file = tmp_path / "parameters.yaml"
    params_file.write_text(
        yaml.safe_dump({"NUM_EPOCHS": 3, "KEEP_LAST_N_CHECKPOINTS": 1})
    )
    out = tmp_path / "build.yaml"
    params_out = tmp_path / "resolved.yaml"
    rc = gb.main(
        [
            "--output", str(out),
            "--eval-sets", "bfcl",
            "--parameters-path", str(params_file),
            "--catalog-path", str(_CATALOG),
            "--params-out", str(params_out),
        ]
    )
    assert rc == 0
    resolved = yaml.safe_load(params_out.read_text())
    assert int(resolved["KEEP_LAST_N_CHECKPOINTS"]) >= int(resolved["NUM_EPOCHS"])
    assert int(resolved["KEEP_LAST_N_CHECKPOINTS"]) == 3
