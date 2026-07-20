"""Tests for the k8s sft-checkpoint-eval generator.

For now this file holds catalog-load assertions for eval-catalog.yaml (the
single source of truth for the 27 per-epoch checkpoint evals). Later tasks
extend it with generate_build.py assertions.
"""
import re
from pathlib import Path

import pytest
import yaml

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
