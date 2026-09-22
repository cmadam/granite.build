"""The source-delivery contract is ONE contract, asserted mechanically.

Every ported distillation step delivers ``gb_steps_post_training.distillation`` the same
way: the same ``code_config`` block, and the same region at the top of its ``run:``. That
sameness is the whole point — six subtly different copies of credential handling and
PYTHONPATH resolution is exactly the failure this avoids — so it is asserted rather than
described.

THIS step is the reference. Its own test_step_template.py asserts the region is
*correct*; this file asserts every other ported step's copy is byte-identical to it. The
two together are what let each other step's suite skip re-asserting the twelve
properties: identical to a correct reference is correct.

Living here rather than in a shared conftest is deliberate — it is the reference step's
job to notice when a copy has drifted, and a test that scans its siblings finds a NEW
ported step automatically rather than waiting for someone to add it to a list.
"""

from pathlib import Path

import pytest

_STEPS_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE = "distill-tokenizer-align"

_CC_BEGIN = "  # ─── Source delivery"
_CC_END = '    setup_command: ""'
_SR_BEGIN = "            # --- distill source delivery: BEGIN"
_SR_END = "            # --- distill source delivery: END"


def _templates():
    """Every ported distillation step's template, reference first."""
    found = {}
    for path in sorted(_STEPS_ROOT.glob("distill-*/skypilot/step-template.yaml")):
        found[path.parts[-3]] = path
    return found


def _region(text, begin, end, *, inclusive_end):
    start = text.index(begin)
    stop = text.index(end, start) + (len(end) if inclusive_end else 0)
    return text[start:stop]


def _code_config(text):
    return _region(text, _CC_BEGIN, _CC_END, inclusive_end=True)


def _source_region(text):
    return _region(text, _SR_BEGIN, _SR_END, inclusive_end=True)


def test_the_reference_step_is_present():
    """Guards against this test passing vacuously if the reference is ever renamed."""
    assert _REFERENCE in _templates()


def test_at_least_one_other_step_is_compared():
    """A byte-identity test over a single file proves nothing. This fails while only the
    reference exists, so it turns into a real assertion the moment a second step lands —
    rather than sitting green and empty."""
    assert len(_templates()) >= 2, "only the reference step exists; nothing to compare"


@pytest.mark.parametrize("name", sorted(n for n in _templates() if n != _REFERENCE))
def test_code_config_block_is_byte_identical_to_the_reference(name):
    templates = _templates()
    expected = _code_config(templates[_REFERENCE].read_text())
    actual = _code_config(templates[name].read_text())
    assert actual == expected, (
        f"{name}'s code_config block has drifted from {_REFERENCE}'s. "
        "Splice it from the reference rather than editing it in place."
    )


@pytest.mark.parametrize("name", sorted(n for n in _templates() if n != _REFERENCE))
def test_source_delivery_region_is_byte_identical_to_the_reference(name):
    templates = _templates()
    expected = _source_region(templates[_REFERENCE].read_text())
    actual = _source_region(templates[name].read_text())
    assert actual == expected, (
        f"{name}'s source-delivery region has drifted from {_REFERENCE}'s. "
        "Splice it from the reference rather than editing it in place."
    )


@pytest.mark.parametrize("name", sorted(_templates()))
def test_every_ported_step_has_both_regions(name):
    """A step that grew a bespoke source path would otherwise fail with an obscure
    ValueError from .index() instead of saying what is wrong."""
    text = _templates()[name].read_text()
    for marker in (_CC_BEGIN, _CC_END, _SR_BEGIN, _SR_END):
        assert marker in text, f"{name} is missing the marker {marker!r}"


@pytest.mark.parametrize("name", sorted(_templates()))
def test_no_step_ships_a_dockerfile(name):
    """Every ported step is a non-image step: common.mk keys off the Dockerfile's
    ABSENCE, so adding one silently turns image/publish-image back on."""
    assert not (_templates()[name].parent / "Dockerfile").exists()


# ─── The pinned checkout ───────────────────────────────────────────────────────
# Asserted here rather than in test_step_template.py because it is a property of the
# CONTRACT: all six steps must pin the same tree at the same commit, and the value
# itself has to be one this project controls. The byte-identity tests above already
# guarantee sameness; these say what the shared value must BE.

_PINNED_DIR = "/proj/granite-build/g4os/gb-steps-collection-post-training-gb"
_PINNED_REF = "09bfcb1662529644bd09f620c24293aa43f3b807"
_SHARED_DIR = "/proj/granite-build/g4os/gb-steps-collection-post-training"


@pytest.mark.parametrize("name", sorted(_templates()))
def test_every_step_pins_the_project_controlled_checkout(name):
    """Not the shared clone. That tree is advanced by its upstream author, and every
    time it moves all six pins stop matching and the step exits 1 -- mid-build, in
    1820703f, between `align` and `corpus`. Reverting it is a standoff lost on the next
    fetch, and the failure reads as a recipe bug rather than a moved dependency."""
    text = _templates()[name].read_text(encoding="utf-8")
    assert f'code_dir: "{_PINNED_DIR}"' in text, f"{name} does not pin the -gb checkout"
    assert (
        f'code_dir: "{_SHARED_DIR}"' not in text
    ), f"{name} still points at the shared tree, which moves without warning"


@pytest.mark.parametrize("name", sorted(_templates()))
def test_every_step_pins_a_full_commit(name):
    """A branch head makes two runs a week apart different runs while reporting the
    same provenance, and a prefix is not a commit."""
    text = _templates()[name].read_text(encoding="utf-8")
    assert f'expect_ref: "{_PINNED_REF}"' in text, f"{name} pins a different commit"
    assert len(_PINNED_REF) == 40


def test_the_pin_is_reachable_from_the_patch_it_carries():
    """The pinned commit is the base commit plus this repo's patch. Stating the base in
    the patch file is what lets someone rebuild the checkout from scratch; without it
    the pin names a tree that cannot be reconstructed."""
    patch = (
        _STEPS_ROOT
        / _REFERENCE
        / "skypilot"
        / "patches"
        / "retag_student_identity_vocab.diff"
    )
    assert patch.is_file(), f"missing {patch}"
    text = patch.read_text(encoding="utf-8")
    assert "70c1550a171aa8e09a9ad9047a5bf763c39e8579" in text, "base commit unstated"
    assert "retag_student.py" in text
