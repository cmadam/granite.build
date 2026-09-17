"""
PORTED, not authored here. Upstream source of truth:
  repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
  path   steps/distill-corpus-prep/test/test_prep_corpus.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Divergences, both about path resolution rather than behaviour:
  - upstream's sys.path.insert for the shared package is removed; conftest.py resolves it
    from GB_DISTILL_CODE_DIR, and upstream's relative parents[] do not point at a checkout
    from inside granite.build.
  - the step's own src/ is already on PYTHONPATH via `make test`, so prep_corpus imports
    directly.

Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
"""

"""Unit tests for distill-corpus-prep.

SCOPE, and what is deliberately NOT here. Everything below runs without transformers: the
tokenizer is a stub whose apply_chat_template returns ids and an assistant mask chosen by
the test. That covers the step's DECISIONS -- which records are dropped and why, how the
split is drawn, what the manifest records -- because those are the parts a mocked tokenizer
cannot lie about.

What a stub CANNOT test is the property the step exists to guarantee: that a real Granite
chat template produces a non-empty assistant mask, and that lengths measured here match the
lengths the trainer will measure. Asserting that against a fake would be self-congratulation.
It is tested by executing the RENDERED step-template command on BlueVela against real
tokenizers -- see README.md -- which is also the only thing that exercises the seam between
the template and the script.
"""
import hashlib
import json
import sys
from pathlib import Path

import prep_corpus as pc  # noqa: E402
import pytest
from gb_steps_post_training.distillation import tokenizer_identity  # noqa: E402

# The shared distillation package, for tokenizer_identity. Mirrors the image layout, where
# the Dockerfile vendors it onto PYTHONPATH.


def _conv(n_user=1, assistant="hi", last="assistant", think=False, system=False):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "sys"})
    for i in range(n_user):
        msgs.append({"role": "user", "content": f"q{i}"})
        body = f"<think>because</think>{assistant}" if think else assistant
        msgs.append({"role": "assistant", "content": body})
    if last == "user":
        msgs.append({"role": "user", "content": "trailing"})
    return {"messages": msgs}


def _norm(record, **kw):
    kw.setdefault("think_policy", "keep")
    kw.setdefault("boundary", "last_message")
    kw.setdefault("min_messages", 2)
    return pc.normalise(record, **kw)


# ------------------------------------------------------------------ normalise


def test_keeps_a_well_formed_conversation():
    rec, reason = _norm(_conv())
    assert reason == ""
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]


@pytest.mark.parametrize(
    "record,expected",
    [
        ({}, "no_messages"),
        ({"messages": []}, "no_messages"),
        ({"messages": [{"role": "user", "content": "x"}]}, "too_few_messages"),
        (
            {"messages": ["not a dict", {"role": "user", "content": "x"}]},
            "malformed_message",
        ),
    ],
)
def test_structural_rejections(record, expected):
    assert _norm(record)[1] == expected


def test_rejects_unknown_role_by_name_so_the_log_says_which():
    rec, reason = _norm(
        {
            "messages": [
                {"role": "narrator", "content": "x"},
                {"role": "assistant", "content": "y"},
            ]
        }
    )
    assert rec is None
    assert reason == "bad_role:narrator"


def test_accepts_the_tool_role():
    """The granite-4.2 template renders tool results as their own turn, so a record using
    `tool` is valid input. Rejecting it would silently discard every tool-use conversation.
    """
    rec, reason = _norm(
        {
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "tool", "content": '{"t": 12}'},
                {"role": "assistant", "content": "12 degrees"},
            ]
        }
    )
    assert reason == ""
    assert len(rec["messages"]) == 3


def test_refuses_non_string_content_rather_than_coercing():
    """A tool_calls-only assistant turn is legitimate data this step cannot length-check.
    Coercing it to "" would emit a conversation whose assistant turn is empty."""
    rec, reason = _norm(
        {
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": None},
            ]
        }
    )
    assert rec is None and reason == "non_string_content"


def test_requires_both_a_user_and_an_assistant_turn():
    assert (
        _norm(
            {
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ]
            }
        )[1]
        == "no_assistant_turn"
    )
    assert (
        _norm(
            {
                "messages": [
                    {"role": "system", "content": "a"},
                    {"role": "assistant", "content": "b"},
                ]
            }
        )[1]
        == "no_user_turn"
    )


# ------------------------------------------------- the completion-boundary contract


def test_last_message_boundary_drops_conversations_ending_on_a_user_turn():
    """Under GOLD's last_message_only such a record contributes no loss at all, so keeping
    it would inflate the example count with samples that train nothing."""
    assert _norm(_conv(last="user"))[1] == "last_message_not_assistant"


def test_all_assistant_boundary_keeps_the_same_conversation():
    rec, reason = _norm(_conv(last="user"), boundary="all_assistant")
    assert reason == "" and rec["messages"][-1]["role"] == "user"


# ------------------------------------------------------------------ think policy


def test_think_keep_leaves_the_trace_alone():
    rec, _ = _norm(_conv(think=True), think_policy="keep")
    assert "<think>" in rec["messages"][-1]["content"]


def test_think_strip_removes_the_trace_but_keeps_the_answer():
    rec, _ = _norm(_conv(think=True, assistant="42"), think_policy="strip")
    assert rec["messages"][-1]["content"] == "42"


def test_think_strip_is_multiline_because_real_traces_are():
    body = "<think>line one\nline two\n</think>the answer"
    rec, _ = _norm(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": body},
            ]
        },
        think_policy="strip",
    )
    assert rec["messages"][-1]["content"] == "the answer"


def test_think_strip_drops_a_record_that_was_nothing_but_a_trace():
    rec, reason = _norm(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<think>only</think>"},
            ]
        },
        think_policy="strip",
    )
    assert rec is None and reason == "empty_after_strip"


def test_think_require_drops_records_without_a_trace():
    assert _norm(_conv(think=False), think_policy="require")[1] == "no_think_block"
    assert _norm(_conv(think=True), think_policy="require")[1] == ""


def test_think_tags_in_user_content_do_not_count():
    """Confirmed on real data: in bespoke_stratos_17k_think.jsonl the trace is in the
    ASSISTANT turn. A <think> in a user turn is quoted text, not a trace to supervise.
    """
    rec = {
        "messages": [
            {"role": "user", "content": "what does <think>x</think> mean?"},
            {"role": "assistant", "content": "a tag"},
        ]
    }
    assert _norm(rec, think_policy="require")[1] == "no_think_block"


def test_passthrough_fields_survive_and_absent_ones_are_not_invented():
    rec, _ = _norm({**_conv(), "documents": [{"text": "d"}], "tools": [{"name": "t"}]})
    assert rec["documents"] == [{"text": "d"}] and rec["tools"] == [{"name": "t"}]
    assert set(_norm(_conv())[0]) == {"messages"}


# ------------------------------------------------------------------ measure


class _Tok:
    """Stub tokenizer. `plan` maps a rendered-conversation marker to (n_tokens, n_target)."""

    def __init__(self, n_tokens=10, n_target=4, raises=False):
        self.n_tokens, self.n_target, self.raises = n_tokens, n_target, raises
        self.calls = []

    def apply_chat_template(self, convs, **kw):
        self.calls.append(kw)
        if self.raises:
            raise ValueError("template blew up")
        mask = [1] * self.n_target + [0] * (self.n_tokens - self.n_target)
        return {"input_ids": [list(range(self.n_tokens))], "assistant_masks": [mask]}


def _measure(rec, tok, max_length=100, length_policy="drop"):
    return pc.measure(rec, tok, max_length=max_length, length_policy=length_policy)


def test_measure_requests_the_assistant_mask():
    """If this kwarg is ever dropped the empty-mask check silently stops checking."""
    tok = _Tok()
    _measure(_conv(), tok)
    assert tok.calls[0]["return_assistant_tokens_mask"] is True
    assert tok.calls[0]["add_generation_prompt"] is False


def test_measure_drops_a_record_with_an_empty_assistant_mask():
    rec, reason, n_tok, n_tgt = _measure(_conv(), _Tok(n_tokens=8, n_target=0))
    assert rec is None and reason == "empty_assistant_mask" and n_tgt == 0


def test_over_length_is_dropped_under_the_default_policy():
    rec, reason, n_tok, _ = _measure(_conv(), _Tok(n_tokens=50), max_length=20)
    assert rec is None and reason == "over_max_length" and n_tok == 50


def test_over_length_is_kept_and_flagged_under_truncate():
    rec, reason, _, _ = _measure(
        _conv(), _Tok(n_tokens=50), max_length=20, length_policy="truncate"
    )
    assert rec is not None and reason == "truncated"


def test_a_template_error_is_a_drop_reason_not_a_crash():
    """Template failures are data-dependent. One malformed record must not abort a corpus."""
    rec, reason, _, _ = _measure(_conv(), _Tok(raises=True))
    assert rec is None and reason.startswith("template_error:ValueError")


def test_tools_and_documents_are_forwarded_to_the_template():
    """They change the rendered prompt, so a length measured without them is the wrong
    length -- and the trainer WILL render with them."""
    tok = _Tok()
    _measure({**_conv(), "tools": [{"name": "t"}], "documents": [{"text": "d"}]}, tok)
    assert tok.calls[0]["tools"] == [{"name": "t"}]
    assert tok.calls[0]["documents"] == [{"text": "d"}]


# ------------------------------------------------------------------ load_records


def test_load_records_reads_jsonl_and_skips_blank_lines(tmp_path):
    fp = tmp_path / "d.jsonl"
    fp.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert [r["a"] for r in pc.load_records(str(fp), "train", "", "")] == [1, 2]


def test_load_records_names_the_line_number_of_bad_json(tmp_path):
    fp = tmp_path / "d.jsonl"
    fp.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(pc.PrepError, match=r"d\.jsonl:2"):
        list(pc.load_records(str(fp), "train", "", ""))


def test_a_directory_is_refused_before_it_can_look_like_a_hub_id(tmp_path):
    with pytest.raises(pc.PrepError, match="not a file"):
        list(pc.load_records(str(tmp_path), "train", "", ""))


# ------------------------------------------------------------------ build / manifest


@pytest.fixture
def stub_tok(monkeypatch):
    tok = _Tok()
    monkeypatch.setattr(
        pc, "load_tokenizer", lambda d: (tok, str(d / "chat_template.jinja"))
    )
    return tok


def _args(**kw):
    import argparse

    base = dict(
        dataset="",
        dataset_split="train",
        dataset_config="",
        tokenizer="",
        out_dir="",
        max_length=100,
        length_policy="drop",
        think_policy="keep",
        completion_boundary="last_message",
        min_messages=2,
        eval_fraction=0.0,
        seed=42,
        max_examples=0,
        hf_home="",
        # The CLI default is "drop", but this factory says "keep" deliberately: the
        # fixture conversations carry no `documents`, so the two settings are
        # identical for every pre-existing test, and "keep" keeps the factory the
        # IDENTITY behaviour. A default that filters would mean any future fixture
        # gaining a `documents` field silently changes unrelated assertions. The
        # drop path gets its own tests, which set it explicitly.
        documents_policy="keep",
        # Same reasoning: off is the identity shape for the emitted corpus, so the
        # pre-existing assertions about train.jsonl's contents stay true. The stamp
        # gets its own tests.
        emit_row_id=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _dataset(tmp_path, n, **conv_kw):
    fp = tmp_path / "in.jsonl"
    with fp.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps(_conv(assistant=f"a{i}", **conv_kw)) + "\n")
    return fp


def _tokdir(tmp_path, identity="granite-4.2-3b"):
    d = tmp_path / "tok"
    d.mkdir(exist_ok=True)
    (d / "tokenizer.json").write_text("{}")
    if identity:
        tokenizer_identity.write(d, identity)
    return d


def test_build_emits_manifest_and_train_split(tmp_path, stub_tok):
    out = tmp_path / "out"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 5)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
        )
    )
    assert m["counts"]["kept"] == 5
    assert m["tokenizer_identity"] == "granite-4.2-3b"
    assert m["tokenized"] is False, "the corpus is text; the manifest must say so"
    lines = (out / "train.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5
    assert json.loads(lines[0])["messages"][0]["role"] == "user"
    assert json.loads((out / pc.MANIFEST_NAME).read_text())["counts"]["kept"] == 5


def test_manifest_carries_the_identity_that_gold_train_compares(tmp_path, stub_tok):
    """The one key that couples this step to the trainer. If it moves or is renamed, the
    guard in render_gold_config goes quiet -- which is exactly how it was broken before.
    """
    out = tmp_path / "out"
    pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 2)),
            tokenizer=str(_tokdir(tmp_path, "some-tokenizer")),
            out_dir=str(out),
        )
    )
    manifest = json.loads((out / "corpus_manifest.json").read_text())
    assert manifest[tokenizer_identity.IDENTITY_KEY] == "some-tokenizer"


def test_identity_falls_back_to_a_content_hash_when_unrecorded(tmp_path, stub_tok):
    """A base tokenizer no step produced still yields a checkable identity -- returning None
    here is what made the trainer's guard skippable."""
    out = tmp_path / "out"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 2)),
            tokenizer=str(_tokdir(tmp_path, identity=None)),
            out_dir=str(out),
        )
    )
    assert m["tokenizer_identity"].startswith("sha256:")


def test_eval_split_is_deterministic_from_the_seed(tmp_path, stub_tok):
    def run(seed, out):
        pc.build(
            _args(
                dataset=str(_dataset(tmp_path, 20)),
                tokenizer=str(_tokdir(tmp_path)),
                out_dir=str(out),
                eval_fraction=0.25,
                seed=seed,
            )
        )
        return (out / "eval.jsonl").read_text()

    a = run(42, tmp_path / "a")
    b = run(42, tmp_path / "b")
    c = run(7, tmp_path / "c")
    assert a == b
    assert a != c, "a different seed must draw a different split"


def test_eval_fraction_zero_emits_no_eval_file(tmp_path, stub_tok):
    out = tmp_path / "out"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 4)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            eval_fraction=0.0,
        )
    )
    assert not (out / "eval.jsonl").exists()
    assert set(m["splits"]) == {"train"}


def test_a_tiny_eval_fraction_still_yields_one_example(tmp_path, stub_tok):
    """Rounding to zero would emit "eval requested, nothing produced", which reads as a bug
    downstream rather than as rounding."""
    out = tmp_path / "out"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 4)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            eval_fraction=0.01,
        )
    )
    assert m["splits"]["eval"]["examples"] == 1


def test_eval_fraction_that_would_consume_everything_is_refused(tmp_path, stub_tok):
    with pytest.raises(pc.PrepError, match="nothing to train on"):
        pc.build(
            _args(
                dataset=str(_dataset(tmp_path, 2)),
                tokenizer=str(_tokdir(tmp_path)),
                out_dir=str(tmp_path / "o"),
                eval_fraction=0.99,
            )
        )


def test_splits_are_disjoint_and_cover_everything(tmp_path, stub_tok):
    out = tmp_path / "out"
    pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 10)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            eval_fraction=0.3,
        )
    )

    def bodies(name):
        return [ln for ln in (out / name).read_text().strip().split("\n")]

    tr, ev = bodies("train.jsonl"), bodies("eval.jsonl")
    assert len(tr) + len(ev) == 10
    assert not (set(tr) & set(ev))


def test_max_examples_stops_early(tmp_path, stub_tok):
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 50)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(tmp_path / "o"),
            max_examples=3,
        )
    )
    assert m["counts"]["kept"] == 3
    assert m["counts"]["input"] == 3, "should not read past what it needs"


def test_an_all_empty_mask_corpus_blames_the_template_not_the_data(
    tmp_path, monkeypatch
):
    """The most likely misconfiguration of this step, and the one that otherwise surfaces as
    a GOLD crash at sft.py:909 minutes into a multi-node allocation."""
    monkeypatch.setattr(
        pc, "load_tokenizer", lambda d: (_Tok(n_target=0), "tmpl.jinja")
    )
    with pytest.raises(
        pc.PrepError, match=r"generation %\} markers|EMPTY assistant mask"
    ):
        pc.build(
            _args(
                dataset=str(_dataset(tmp_path, 3)),
                tokenizer=str(_tokdir(tmp_path)),
                out_dir=str(tmp_path / "o"),
            )
        )


def test_zero_survivors_reports_the_drop_reasons(tmp_path, stub_tok):
    fp = tmp_path / "in.jsonl"
    fp.write_text(json.dumps(_conv(last="user")) + "\n")
    with pytest.raises(pc.PrepError, match="last_message_not_assistant"):
        pc.build(
            _args(
                dataset=str(fp),
                tokenizer=str(_tokdir(tmp_path)),
                out_dir=str(tmp_path / "o"),
            )
        )


def test_manifest_records_every_policy_that_changed_the_output(tmp_path, stub_tok):
    """Reproducibility: the manifest must be enough to rebuild the same corpus."""
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 3)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(tmp_path / "o"),
            max_length=77,
            think_policy="strip",
            completion_boundary="all_assistant",
            length_policy="truncate",
        )
    )
    assert m["policies"] == {
        "max_length": 77,
        "length_policy": "truncate",
        "think_policy": "strip",
        "documents_policy": "keep",
        "emit_row_id": False,
        "completion_boundary": "all_assistant",
        "min_messages": 2,
    }
    assert m["token_stats"]["max_tokens"] == 10


# --------------------------------------------------------------- real-data shapes
#
# Every test below was written against a shape found in data/distillation/en_sft_4.1 by
# executing the step on it (LSF job 1137876), where 0 of 405,672 records survived. They are
# regression tests for a corpus the unit suite had no reason to imagine: the 39 tests above
# all passed while the step rejected an entire real dataset.


def test_tools_stored_as_a_json_string_is_parsed_not_forwarded():
    # The shape that killed 395,007 records: `tools` is the string "[]", and
    # apply_chat_template iterates it character by character.
    rec, reason = pc.normalise(
        {"messages": _conv()["messages"], "tools": '[{"name": "f"}]'},
        think_policy="keep",
        boundary="last_message",
        min_messages=2,
    )
    assert reason == ""
    assert rec["tools"] == [{"name": "f"}], "a JSON string must be decoded to a list"


def test_an_empty_tools_string_is_omitted_entirely():
    rec, reason = pc.normalise(
        {"messages": _conv()["messages"], "tools": "[]"},
        think_policy="keep",
        boundary="last_message",
        min_messages=2,
    )
    assert reason == ""
    assert "tools" not in rec, "an empty list must not be passed to the template at all"


def test_an_empty_documents_list_is_omitted():
    rec, _ = pc.normalise(
        {"messages": _conv()["messages"], "documents": []},
        think_policy="keep",
        boundary="last_message",
        min_messages=2,
    )
    assert "documents" not in rec


@pytest.mark.parametrize("bad", ["{not json", '{"a": 1}', "3", '["a string"]'])
def test_tools_that_is_not_a_list_of_dicts_is_a_drop_not_a_render(bad):
    # Refused here, with a reason, rather than forwarded to blow up inside the template --
    # where it arrives as a bare ValueError with no mention of the field.
    rec, reason = pc.normalise(
        {"messages": _conv()["messages"], "tools": bad},
        think_policy="keep",
        boundary="last_message",
        min_messages=2,
    )
    assert rec is None and reason == "bad_tools"


def test_tool_response_is_aliased_to_tool_and_counted():
    # granite-4.2's template accepts `tool`; en_sft_4.1 spells it `tool_response`. The
    # template's `tool` branch emits literal <tool_response> tags, so they are one concept.
    stats: dict = {}
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "tool_response", "content": "result"},
        {"role": "assistant", "content": "a"},
    ]
    rec, reason = pc.normalise(
        {"messages": msgs},
        think_policy="keep",
        boundary="last_message",
        min_messages=2,
        stats=stats,
    )
    assert reason == ""
    assert [m["role"] for m in rec["messages"]] == ["user", "tool", "assistant"]
    assert stats == {"role_alias:tool_response->tool": 1}, "the rename must be visible"


def test_an_unknown_role_is_still_refused():
    # The alias map is a two-entry allowlist, not a licence to guess.
    rec, reason = pc.normalise(
        {
            "messages": [
                {"role": "user", "content": "u"},
                {"role": "narrator", "content": "x"},
                {"role": "assistant", "content": "a"},
            ]
        },
        think_policy="keep",
        boundary="last_message",
        min_messages=2,
    )
    assert rec is None and reason == "bad_role:narrator"


def test_the_first_template_error_message_is_captured():
    class _Boom:
        def apply_chat_template(self, *a, **k):
            raise ValueError(
                "Tools should either be a JSON schema, or a callable function"
            )

    detail: dict = {}
    rec, reason, _, _ = pc.measure(
        {"messages": _conv()["messages"]},
        _Boom(),
        max_length=10,
        length_policy="drop",
        detail=detail,
    )
    assert rec is None and reason == "template_error:ValueError"
    assert (
        "JSON schema" in detail["template_error:ValueError"]
    ), "the type alone cost a job round-trip to diagnose; keep the message"


# ------------------------------------------------------- boundary-scoped token counting


@pytest.mark.parametrize(
    "mask,expected",
    [
        ([0, 0, 1, 1, 0, 0, 1, 1, 1], 3),  # two turns -> only the final one
        ([1, 1, 1], 3),  # one turn that is the whole sequence
        ([0, 0, 1], 1),  # mask ends at the last token
        ([1, 0, 0], 1),  # ...and does not have to
        ([0, 0, 0], 0),  # no generation markers at all
        ([], 0),
    ],
)
def test_last_mask_span(mask, expected):
    assert pc.last_mask_span(mask) == expected


def test_the_boundary_scopes_what_is_counted_not_just_what_is_dropped():
    # The template marks EVERY assistant turn as generation, but GOLD under
    # last_message_only supervises the final one. A multi-turn record must therefore report
    # fewer target tokens under last_message than under all_assistant. Single-turn data
    # (the whole of bespoke_stratos) cannot show this difference, which is why job 1137876
    # reported identical stats for both boundaries and looked fine.
    class _TwoTurns:
        def apply_chat_template(self, convs, **k):
            return {
                "input_ids": [[1] * 10],
                "assistant_masks": [[0, 1, 1, 0, 0, 1, 1, 1, 1, 0]],
            }

    rec = {"messages": _conv(n_user=2)["messages"]}
    tok = _TwoTurns()
    _, _, _, n_last = pc.measure(
        rec, tok, max_length=99, length_policy="drop", boundary="last_message"
    )
    _, _, _, n_all = pc.measure(
        rec, tok, max_length=99, length_policy="drop", boundary="all_assistant"
    )
    assert n_last == 4 and n_all == 6, "last_message must count only the final turn"


def test_totals_report_this_record_not_a_running_sum():
    # Overwritten per call so build() can accumulate over KEPT records only; accumulating
    # inside measure() would fold in the records dropped for length and make the
    # "unsupervised signal" figure incomparable to total_target_tokens.
    class _Tok3:
        def apply_chat_template(self, convs, **k):
            return {"input_ids": [[1] * 4], "assistant_masks": [[0, 1, 1, 1]]}

    totals: dict = {}
    rec = {"messages": _conv()["messages"]}
    for _ in range(3):
        pc.measure(rec, _Tok3(), max_length=99, length_policy="drop", totals=totals)
    assert totals["mask_tokens_all_assistant"] == 3


def test_an_all_zero_mask_is_a_template_fault_under_either_boundary():
    class _NoMarkers:
        def apply_chat_template(self, convs, **k):
            return {"input_ids": [[1, 2, 3]], "assistant_masks": [[0, 0, 0]]}

    for boundary in ("last_message", "all_assistant"):
        rec, reason, _, _ = pc.measure(
            {"messages": _conv()["messages"]},
            _NoMarkers(),
            max_length=99,
            length_policy="drop",
            boundary=boundary,
        )
        assert rec is None and reason == "empty_assistant_mask"


# ------------------------------------------------- per-row training manifest + documents
#
# These cover the two things hew asked for on 2026-08-26: drop the grounded records whose
# documents the template discards, and be able to say WHICH data point was used in training.
# The second is why row_id exists at all -- the source corpus has no id field, so identity
# has to be manufactured from content, and every property below is a property that a future
# refactor could break without any existing test noticing.


def test_row_id_is_stable_under_key_order_and_whitespace():
    """The id must be a function of the DATA, not of how it was serialised.

    Otherwise the same conversation read from two files gets two ids, and the manifest
    claims to have trained on a row it never saw. This is the property that makes a content
    hash usable as a key at all, so it is asserted directly rather than assumed from
    `sort_keys=True` being present in the source.
    """
    a = {"messages": [{"role": "user", "content": "hi"}], "tools": "[]"}
    b = {"tools": "[]", "messages": [{"content": "hi", "role": "user"}]}
    assert pc.row_id(a) == pc.row_id(b)
    assert pc.row_id(a) != pc.row_id(
        {"messages": [{"role": "user", "content": "hi!"}], "tools": "[]"}
    )
    assert len(pc.row_id(a)) == 32, "blake2b-128 is 16 bytes = 32 hex chars"


def test_row_id_distinguishes_unicode_that_escaping_would_collapse():
    """ensure_ascii=False means the bytes hashed are the text's own UTF-8.

    A future editor who drops that argument would still pass the test above, because
    escaping is deterministic -- so the risk is not collision but a SILENT change of every
    id in the corpus, which invalidates every manifest ever published. This pins the digest
    of a known non-ASCII record so that change cannot land unnoticed.
    """
    rec = {"messages": [{"role": "user", "content": "café 你好"}]}
    assert pc.row_id(rec) == pc.row_id(json.loads(json.dumps(rec)))
    # Same data, ASCII-escaped on the way in: must still be the same id.
    assert pc.row_id(rec) == pc.row_id(json.loads(json.dumps(rec, ensure_ascii=True)))


def test_sidecar_has_one_entry_per_input_record_kept_or_dropped(tmp_path, stub_tok):
    """The claim the whole file rests on: nothing is silently unaccounted for."""
    fp = tmp_path / "in.jsonl"
    with fp.open("w") as fh:
        fh.write(json.dumps(_conv(assistant="a0")) + "\n")  # kept
        fh.write(
            json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
        )  # dropped
        fh.write(json.dumps(_conv(assistant="a1")) + "\n")  # kept
    out = tmp_path / "o"
    m = pc.build(
        _args(dataset=str(fp), tokenizer=str(_tokdir(tmp_path)), out_dir=str(out))
    )
    rows = [json.loads(l) for l in (out / pc.ROWS_NAME).read_text().strip().split("\n")]
    assert len(rows) == 3 == m["rows"]["entries"]
    assert [r["disposition"] for r in rows] == ["kept", "dropped", "kept"]
    assert m["counts"]["kept"] == 2
    # A dropped row still carries an id and a reason -- "which data point was NOT used, and
    # why" is half of what a provenance manifest is for.
    assert rows[1]["src_id"] and rows[1]["reason"]
    assert (
        "out_id" not in rows[1]
    ), "a dropped record was never emitted, so it has no out_id"


def test_sidecar_src_and_out_ids_differ_exactly_when_the_record_was_transformed(
    tmp_path, stub_tok
):
    """Both ids are kept because normalise() rewrites records.

    A holder of train.jsonl can only hash what they have -- the EMITTED row -- so out_id is
    what makes a lookup possible from that end; src_id is what links back to the source
    dataset. If a refactor ever made out_id a copy of src_id, provenance would look complete
    while being wrong for every transformed row, which is the failure this pins.
    """
    fp = tmp_path / "in.jsonl"
    # `tools` as the JSON string "[]" is the real corpus's shape; normalise() parses and then
    # omits it, so the emitted record genuinely differs from the source.
    rec = _conv(assistant="a0")
    rec["tools"] = "[]"
    fp.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "o"
    pc.build(_args(dataset=str(fp), tokenizer=str(_tokdir(tmp_path)), out_dir=str(out)))
    row = json.loads((out / pc.ROWS_NAME).read_text().strip())
    assert row["src_id"] != row["out_id"], "this record WAS transformed"
    emitted = json.loads((out / "train.jsonl").read_text().strip())
    assert (
        pc.row_id(emitted) == row["out_id"]
    ), "hashing a row of train.jsonl must find it in the sidecar"


def test_sidecar_records_the_split_so_training_rows_are_identifiable(
    tmp_path, stub_tok
):
    """`kept` is NOT the training set once eval_fraction > 0."""
    out = tmp_path / "o"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 10)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            eval_fraction=0.2,
        )
    )
    rows = [json.loads(l) for l in (out / pc.ROWS_NAME).read_text().strip().split("\n")]
    kept = [r for r in rows if r["disposition"] == "kept"]
    assert {r["split"] for r in kept} == {"train", "eval"}
    n_train = sum(1 for r in kept if r["split"] == "train")
    assert n_train == m["rows"]["training_rows"] == m["splits"]["train"]["examples"]
    assert n_train < len(kept), "the eval rows must NOT count as trained on"


def test_manifest_sha_matches_the_sidecar_on_disk(tmp_path, stub_tok):
    """The pair has to be verifiable as a pair, not just present in the same directory."""
    out = tmp_path / "o"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 4)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
        )
    )
    got = hashlib.sha256((out / pc.ROWS_NAME).read_bytes()).hexdigest()
    assert m["rows"]["sha256"] == got
    assert m["rows"]["row_id_scheme"] == pc.ROW_ID_SCHEME


def _grounded(**kw):
    rec = _conv(**kw)
    rec["documents"] = [{"title": "t", "text": "some grounding text"}]
    return rec


def test_documents_policy_drop_removes_grounded_records_with_a_named_reason(
    tmp_path, stub_tok
):
    fp = tmp_path / "in.jsonl"
    with fp.open("w") as fh:
        fh.write(json.dumps(_conv(assistant="a0")) + "\n")
        fh.write(json.dumps(_grounded(assistant="a1")) + "\n")
    out = tmp_path / "o"
    m = pc.build(
        _args(
            dataset=str(fp),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            documents_policy="drop",
        )
    )
    assert m["counts"]["kept"] == 1
    # A NAMED reason, not a bare count: the manifest has to say why 0.86% of an epoch went
    # missing, or the next reader re-derives it from scratch.
    assert m["counts"]["drop_reasons"] == {"documents_not_renderable": 1}
    rows = [json.loads(l) for l in (out / pc.ROWS_NAME).read_text().strip().split("\n")]
    assert [r["disposition"] for r in rows] == ["kept", "dropped"]


def test_documents_policy_drop_keeps_records_whose_documents_are_empty(
    tmp_path, stub_tok
):
    """An empty `documents` is not a grounded record and must survive.

    This is the bug the policy would most easily have: the real corpus stores the field as
    the JSON STRING "[]", which is truthy, so a naive truthiness test would discard the
    entire corpus while reporting a plausible-looking drop reason.
    """
    fp = tmp_path / "in.jsonl"
    with fp.open("w") as fh:
        a = _conv(assistant="a0")
        a["documents"] = []
        b = _conv(assistant="a1")
        b["documents"] = "[]"
        c = _conv(assistant="a2")
        for r in (a, b, c):
            fh.write(json.dumps(r) + "\n")
    m = pc.build(
        _args(
            dataset=str(fp),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(tmp_path / "o"),
            documents_policy="drop",
        )
    )
    assert m["counts"]["kept"] == 3, "empty documents is not grounding"
    assert m["counts"]["drop_reasons"] == {}


def test_documents_policy_keep_is_the_identity_behaviour(tmp_path, stub_tok):
    fp = tmp_path / "in.jsonl"
    fp.write_text(json.dumps(_grounded(assistant="a1")) + "\n")
    m = pc.build(
        _args(
            dataset=str(fp),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(tmp_path / "o"),
            documents_policy="keep",
        )
    )
    assert m["counts"]["kept"] == 1


def test_template_renders_documents_is_measured_not_assumed():
    """The premise of the drop policy is re-checked per run; both answers must be reachable."""

    class Discards:
        def apply_chat_template(self, msgs, **kw):
            return "<|im_start|>user\nsummarise<|im_end|>"

    class Renders:
        def apply_chat_template(self, msgs, **kw):
            docs = kw.get("documents") or []
            return "".join(d.get("text", "") for d in docs) + "<|im_start|>user\n"

    class Raises:
        def apply_chat_template(self, msgs, **kw):
            raise TypeError("this template takes no documents")

    assert pc.template_renders_documents(Discards()) is False
    assert pc.template_renders_documents(Renders()) is True
    # A template that REFUSES documents also does not render them. Conservative on purpose:
    # the student sees no grounding either way, so the policy decision is the same.
    assert pc.template_renders_documents(Raises()) is False


def test_manifest_records_the_measured_renderability(tmp_path, stub_tok):
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 2)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(tmp_path / "o"),
        )
    )
    assert m["template_renders_documents"] is False
    assert (
        "template_renders_documents" not in m["policies"]
    ), "it is an observation about the template, not a policy knob"


def test_emit_row_id_stamps_a_verifiable_id_into_the_emitted_row(tmp_path, stub_tok):
    """The stamped id must be recomputable from the row that carries it.

    This is the whole point of excluding ROW_ID_FIELD from its own preimage: a consumer who
    holds only train.jsonl can CHECK provenance rather than take it on faith. If the field
    were ever folded back into the hash, every stamped row would fail to verify while the
    manifest still looked complete.
    """
    out = tmp_path / "o"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 5)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            emit_row_id=True,
        )
    )
    emitted = [
        json.loads(l) for l in (out / "train.jsonl").read_text().strip().split("\n")
    ]
    assert emitted, "nothing was emitted"
    for row in emitted:
        assert pc.row_id(row) == row[pc.ROW_ID_FIELD]
    assert m["rows"]["id_field"] == pc.ROW_ID_FIELD
    assert m["policies"]["emit_row_id"] is True
    # And the stamped value is the same out_id the sidecar recorded, so the two records of
    # the same fact cannot disagree.
    side = {
        r["out_id"]
        for r in (
            json.loads(l) for l in (out / pc.ROWS_NAME).read_text().strip().split("\n")
        )
        if r["disposition"] == "kept"
    }
    assert {r[pc.ROW_ID_FIELD] for r in emitted} <= side


def test_row_id_is_idempotent_under_stamping():
    """Hashing a stamped record must give back the stamp, not a new value.

    Called out separately from the end-to-end test because this is the property a refactor
    would break most cheaply -- dropping the exclusion is a one-line change that no other
    assertion notices until a real corpus has already been published with unverifiable ids.
    """
    rec = {"messages": [{"role": "user", "content": "hi"}]}
    first = pc.row_id(rec)
    stamped = {**rec, pc.ROW_ID_FIELD: first}
    assert pc.row_id(stamped) == first
    # Idempotent a second time, and unaffected by a WRONG stamp -- the id is a function of
    # the content, so a corrupted field is detectable rather than self-confirming.
    assert pc.row_id({**rec, pc.ROW_ID_FIELD: "deadbeef"}) == first


def test_emit_row_id_off_leaves_the_emitted_corpus_unchanged(tmp_path, stub_tok):
    """Off must mean absent, not present-and-empty: the flag is opt-in until a real run has
    proven the trainer tolerates the extra column."""
    out = tmp_path / "o"
    m = pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 3)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
        )
    )
    for line in (out / "train.jsonl").read_text().strip().split("\n"):
        assert pc.ROW_ID_FIELD not in json.loads(line)
    assert m["rows"]["id_field"] is None
    assert m["policies"]["emit_row_id"] is False


# ------------------------------------------------- completion marker (resume across steps)
#
# The behaviour under test is what lets a recipe restarted after a preemption in a LATER step walk
# past this one. Before the marker, prep-corpus.sh:108 refused on the mere presence of
# train.jsonl, which gave the same answer to "identical corpus already built" (nothing to do) and
# "different policies were requested" (do not overwrite) -- and made a restart need a human to
# delete 3.1 GB of correct output.


def _build_twice(tmp_path, stub_tok, second_kw=None):
    out = tmp_path / "out"
    a1 = _args(
        dataset=str(_dataset(tmp_path, 5)),
        tokenizer=str(_tokdir(tmp_path)),
        out_dir=str(out),
    )
    pc.build(a1)
    a2 = _args(**{**vars(a1), **(second_kw or {})})
    return out, a2


def test_a_marker_is_written_after_the_manifest(tmp_path, stub_tok):
    out, _ = _build_twice(tmp_path, stub_tok)
    from gb_steps_post_training.distillation import step_state

    doc = json.loads((out / step_state.MARKER_NAME).read_text())
    assert doc["step"] == pc.STEP_NAME
    # Every real output is recorded, so a corpus whose sidecar was deleted does not read as done.
    assert {r["path"] for r in doc["outputs"]} == {
        "train.jsonl",
        pc.ROWS_NAME,
        pc.MANIFEST_NAME,
    }


def test_rebuilding_the_same_corpus_reports_already_done_instead_of_redoing_the_work(
    tmp_path, stub_tok
):
    out, a2 = _build_twice(tmp_path, stub_tok)
    before = (out / "train.jsonl").read_bytes()
    with pytest.raises(pc.AlreadyDone) as exc:
        pc.build(a2)
    # The manifest comes back so a caller can print the counts without re-reading the file, and
    # nothing was rewritten.
    assert exc.value.manifest["counts"]["kept"] == 5
    assert (out / "train.jsonl").read_bytes() == before


def test_already_done_is_exit_zero_from_main_because_a_restarted_recipe_must_get_past_it(
    tmp_path, stub_tok, capsys
):
    out, a2 = _build_twice(tmp_path, stub_tok)
    argv = [
        "--dataset",
        a2.dataset,
        "--tokenizer",
        a2.tokenizer,
        "--out-dir",
        str(out),
        "--max-length",
        "100",
        "--documents-policy",
        "keep",
    ]
    assert pc.main(argv) == 0
    assert "already built" in capsys.readouterr().out


@pytest.mark.parametrize(
    "changed,names_key",
    [
        ({"max_length": 64}, "max_length"),
        ({"think_policy": "strip"}, "think_policy"),
        ({"completion_boundary": "all_assistant"}, "completion_boundary"),
        ({"documents_policy": "drop"}, "documents_policy"),
        ({"emit_row_id": True}, "emit_row_id"),
        ({"min_messages": 3}, "min_messages"),
        ({"seed": 7}, "seed"),
        ({"max_examples": 3}, "max_examples"),
    ],
)
def test_a_changed_policy_refuses_and_names_the_key(
    tmp_path, stub_tok, changed, names_key
):
    out, a2 = _build_twice(tmp_path, stub_tok, changed)
    with pytest.raises(pc.PrepError) as exc:
        pc.build(a2)
    assert names_key in str(exc.value)
    assert "DIFFERENT expectation" in str(exc.value)


def test_a_different_dataset_at_the_same_out_dir_refuses(tmp_path, stub_tok):
    out, a2 = _build_twice(tmp_path, stub_tok)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(_conv(assistant="z")) + "\n")
    a2.dataset = str(other)
    with pytest.raises(pc.PrepError) as exc:
        pc.build(a2)
    assert "dataset" in str(exc.value)


def test_a_retagged_tokenizer_with_a_new_identity_refuses_even_at_the_same_path(
    tmp_path, stub_tok
):
    """The case tokenizer_identity exists for, arriving through the completion marker.

    A tokenizer directory can be rebuilt in place -- the retag step does exactly that -- so the
    PATH is not the input. The identity is, which is why the expectation records it rather than
    args.tokenizer.
    """
    out, a2 = _build_twice(tmp_path, stub_tok)
    tokenizer_identity.write(Path(a2.tokenizer), "granite-4.1-3b-retagged")
    with pytest.raises(pc.PrepError) as exc:
        pc.build(a2)
    assert "tokenizer_identity" in str(exc.value)


def test_a_deleted_output_refuses_rather_than_silently_rebuilding(tmp_path, stub_tok):
    """0.7 h on the full corpus. A missing declared output means something unexplained happened
    to the tree, and spending that on a guess is worse than stopping with the filename.
    """
    out, a2 = _build_twice(tmp_path, stub_tok)
    (out / pc.ROWS_NAME).unlink()
    with pytest.raises(pc.PrepError) as exc:
        pc.build(a2)
    assert pc.ROWS_NAME in str(exc.value)


def test_an_eval_split_is_recorded_only_when_it_exists(tmp_path, stub_tok):
    """eval.jsonl is conditional on --eval-fraction, so listing it unconditionally would make
    every zero-eval run refuse on the absence of a file it never writes."""
    from gb_steps_post_training.distillation import step_state

    out = tmp_path / "out"
    pc.build(
        _args(
            dataset=str(_dataset(tmp_path, 10)),
            tokenizer=str(_tokdir(tmp_path)),
            out_dir=str(out),
            eval_fraction=0.2,
        )
    )
    doc = json.loads((out / step_state.MARKER_NAME).read_text())
    assert "eval.jsonl" in {r["path"] for r in doc["outputs"]}


def test_two_shards_of_one_corpus_do_not_collide_because_the_shard_is_in_the_expectation(
    tmp_path, stub_tok
):
    """Each shard writes its own out_dir, so this is really a check that the shard index is part
    of what "already built" means -- a shard 1 marker must not make shard 0 look done.
    """
    ds = str(_dataset(tmp_path, 10))
    tok = str(_tokdir(tmp_path))
    out0 = tmp_path / "s0"
    pc.build(
        _args(
            dataset=ds, tokenizer=tok, out_dir=str(out0), shard_index=0, shard_count=2
        )
    )
    with pytest.raises(pc.PrepError) as exc:
        pc.build(
            _args(
                dataset=ds,
                tokenizer=tok,
                out_dir=str(out0),
                shard_index=1,
                shard_count=2,
            )
        )
    assert "shard" in str(exc.value)
