"""Unit tests for scripts/gen_rope_reeval_commands.py.

The one property worth real tests: EXPERIMENT must be unique per model. When two eval
targets share one, they register the same artifact URI, the second registration is refused,
the target still reports SUCCESS with an empty output list, and consumers wait forever --
build b5f030cd, which is the failure mode this generator exists to make impossible across
52 models.
"""

import importlib.util
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "gen_rope_reeval_commands.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("gen_rope_reeval_commands", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()

ROOT = "/proj/granite-build/g4os/distill"

# The real inventory, as the repair scan found it on /proj.
ARMS = {
    "distill-350m-s1v2-ce0/9dab9130-64d7-48a3-a7ef-269712bac7c2": [
        25,
        50,
        75,
        100,
        150,
        200,
        300,
    ],
    "distill-350m-s1v2-ce005-retry/d8ca8e05-7617-4fd5-af16-a216ab8626e7": [
        25,
        50,
        75,
        100,
        150,
        200,
        300,
    ],
    "distill-350m-s1v2-ce015/8c8ebd63-411e-49a2-817b-a813f527c0e9": [
        25,
        50,
        75,
        100,
        150,
        200,
        300,
    ],
    "distill-350m-s1v2-ce040/c20ed3c0-67ed-4135-8f82-518ec6878e96": [
        25,
        50,
        75,
        100,
        150,
        200,
        300,
    ],
    "distill-350m-s1v2-early/57d77ea2-302d-462e-b315-b1600eb75fa9": [1, 2, 5, 10],
    "distill-350m-s1v2-lr0/f9e01bf4-cf6e-4ed9-a678-caae6575e780": [1],
    "distill-350m-stage1-v2-ctrl/bb779f1f-1fe6-4c3c-b8f0-d06285f3fd40": [
        500,
        1000,
        1500,
        2000,
    ],
}


def _real_dirs():
    dirs = []
    for arm, rungs in ARMS.items():
        dirs.append(f"{ROOT}/{arm}/align/retagged_student")
        dirs += [f"{ROOT}/{arm}/export-{r}" for r in rungs]
    # The two plain `export` dirs the original scan's export-* glob missed.
    dirs.append(
        f"{ROOT}/distill-350m-stage1/df8512e0-be93-4762-85ad-813d9080d72f/export"
    )
    dirs.append(
        f"{ROOT}/distill-350m-stage1/3185ce8c-4644-46c8-95ac-9aec292cc3cc/export"
    )
    return dirs


def test_experiment_names_are_unique_across_the_whole_inventory():
    names = [mod.experiment_name(Path(d)) for d in _real_dirs()]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"EXPERIMENT collision would silently lose results: {dupes}"


def test_experiment_name_distinguishes_two_builds_of_one_arm():
    """lr0 has two build ids and both carry an align/retagged_student. Naming by arm+rung
    alone would collide, so the build id has to be in there."""
    a = mod.experiment_name(
        Path(
            f"{ROOT}/distill-350m-s1v2-lr0/6f043cfd-03df-445a-81a1-f751011abde0/align/retagged_student"
        )
    )
    b = mod.experiment_name(
        Path(
            f"{ROOT}/distill-350m-s1v2-lr0/f9e01bf4-cf6e-4ed9-a678-caae6575e780/align/retagged_student"
        )
    )
    assert a != b


def test_experiment_name_marks_a_baseline_as_such():
    n = mod.experiment_name(
        Path(
            f"{ROOT}/distill-350m-s1v2-ce0/9dab9130-64d7-48a3-a7ef-269712bac7c2/align/retagged_student"
        )
    )
    assert "baseline" in n


def test_experiment_name_handles_a_plain_export_dir():
    n = mod.experiment_name(
        Path(f"{ROOT}/distill-350m-stage1/df8512e0-be93-4762-85ad-813d9080d72f/export")
    )
    assert "df8512e0" in n and n


def test_experiment_names_are_safe_for_a_uri_path_segment():
    for d in _real_dirs():
        n = mod.experiment_name(Path(d))
        assert n == n.strip()
        assert all(c.isalnum() or c in "-_" for c in n), n
        assert len(n) <= 64, n


def test_bfcl_command_pins_the_model_and_the_experiment():
    d = Path(
        f"{ROOT}/distill-350m-s1v2-ce0/9dab9130-64d7-48a3-a7ef-269712bac7c2/export-25"
    )
    cmd = mod.bfcl_command(d, categories="simple")
    assert f"MODEL_PATH={d}" in cmd
    assert f"EXPERIMENT={mod.experiment_name(d)}" in cmd
    # 'simple' for comparability with the numbers already on record.
    assert "TEST_CATEGORIES=simple" in cmd
    assert "recipes/granite4-350m/lsf/bfcl-eval" in cmd


def test_full_eval_command_restricts_to_the_prioritised_targets():
    d = Path(
        f"{ROOT}/distill-350m-s1v2-ce040/c20ed3c0-67ed-4135-8f82-518ec6878e96/export-300"
    )
    cmd = mod.full_eval_command(d)
    for target in ("evalplus-humaneval", "multiple-java", "olmes-ifeval"):
        assert target in cmd
    assert f"MODEL_PATH={d}" in cmd


# ------------------------------------------------- full-eval's own bfcl target
#
# full-eval carries a 26th target, `bfcl`, whose output URI is
#   env://$BFCL_OUTPUT_DIR/$EXPERIMENT/$BFCL_EVAL_NAME
# and the standalone bfcl-eval recipe's is
#   env://$OUTPUT_DIR/$EXPERIMENT/$EVAL_NAME
# with BFCL_OUTPUT_DIR == OUTPUT_DIR == /proj/granite-build/g4os/bfcl and
# BFCL_EVAL_NAME == EVAL_NAME == bfclv3. Same EXPERIMENT therefore means the SAME artifact
# URI registered twice -- the b5f030cd mode again, on the 8 prioritised models.


def test_full_eval_includes_bfcl():
    d = Path(f"{ROOT}/distill-350m-s1v2-ce040/c20ed3c0-67ed-4135-8f82-518ec6878e96/export-300")
    assert "bfcl" in mod.full_eval_command(d).split()


def test_full_eval_bfcl_does_not_collide_with_the_standalone_sweep():
    """The two must not resolve to one artifact URI. EXPERIMENT stays the model's identity
    (that is what makes sage and bfcl results joinable), so the eval name is what differs."""
    d = Path(f"{ROOT}/distill-350m-s1v2-ce040/c20ed3c0-67ed-4135-8f82-518ec6878e96/export-300")
    full = mod.full_eval_command(d)
    assert f"BFCL_EVAL_NAME={mod.WIDE_EVAL_NAME}" in full
    assert mod.WIDE_EVAL_NAME != mod.STANDALONE_EVAL_NAME
    # Same EXPERIMENT on purpose; the URI is disambiguated by the eval name instead.
    assert f"EXPERIMENT={mod.experiment_name(d)}" in full
    assert f"EXPERIMENT={mod.experiment_name(d)}" in mod.bfcl_command(d)


def test_full_eval_bfcl_widens_the_categories():
    """The standalone sweep runs `simple` for comparability with the numbers on record.
    Re-running `simple` again here would buy nothing, so the full-eval rung is the widening
    the report left open."""
    d = Path(f"{ROOT}/distill-350m-s1v2-ce0/9dab9130-64d7-48a3-a7ef-269712bac7c2/export-25")
    assert "BFCL_TEST_CATEGORIES=all" in mod.full_eval_command(d)
    assert "TEST_CATEGORIES=simple" in mod.bfcl_command(d, categories="simple")
