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
