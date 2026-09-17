#!/usr/bin/env python3
#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-corpus-prep/src/prep_corpus.py
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# Verbatim apart from `black`/`isort` reflow, which CI requires repo-wide. Keep it that
# way so re-syncing upstream stays a three-way merge; behaviour changes belong upstream.
#
# It imports gb_steps_post_training.distillation at module scope, which is delivered at
# RUN time from the checkout named by code_config (see step-template.yaml). That is why
# the tests for this file are gated on GB_DISTILL_CODE_DIR — see test/conftest.py.
#
"""Build a GOLD training corpus: filter, normalise and split a conversation dataset.

WHAT THE OUTPUT IS, precisely, because the plan doc got this wrong and the mistake is easy
to repeat. The corpus is a TEXT-level JSONL of `messages` conversations. It contains no
token ids. gold.py detects a `.json`/`.jsonl` dataset_name, reads it with pandas, wraps it
in a Dataset (gold.py:389-433) and the trainer tokenizes at train time.

SO WHY IS THE CORPUS TOKENIZER-SPECIFIC, and why does this step demand a --tokenizer at
all? Because every decision it makes about WHICH conversations survive is measured in
tokens:
  - a conversation is kept or dropped by its rendered length against --max-length;
  - the completion boundary is checked by rendering with return_assistant_tokens_mask=True
    and requiring a non-empty mask.
Both answers move when the tokenizer moves. The two families here differ in pre_tokenizer
(Sequence[Split(regex), ByteLevel] on granite-4.1 vs plain ByteLevel on granite-4.2), which
is easily enough to push examples across a length boundary. Nothing downstream re-checks
lengths, so a corpus built with the wrong tokenizer is not a crash: it is a training set
that quietly contains examples the trainer will truncate. That is what corpus_manifest.json's
`tokenizer_identity` exists to prevent -- distill-gold-train refuses a mismatch against the
student it is about to train.

WHY THE TOKENIZER IS LOADED WITHOUT AutoTokenizer. Same reason retag_student.py does not use
it: a Granite directory's tokenizer_config.json declares `tokenizer_class: "GPT2Tokenizer"`,
so AutoTokenizer constructs that class, which imposes its own plain ByteLevel pre_tokenizer
over the one stored in tokenizer.json. It does not error -- it silently mis-segments
(26.1 vs 3.29 PPL/token, measured; docs/tokenizer_mismatch.md). Since this step's whole
purpose is to measure lengths with the tokenizer that will actually train, being wrong here
would be self-defeating in a way no test downstream would catch. So the tokenizer is built
directly as PreTrainedTokenizerFast(tokenizer_file=...) -- immune by construction, exactly
like the retag -- and the chat template is read from chat_template.jinja by hand.

THE CHECK THAT EARNS ITS KEEP. --completion-boundary is verified, not merely recorded. A
chat template without `{% generation %}` markers yields an all-zero assistant mask, and
GOLD does not notice until sft.py:909 -- after model load, after the vLLM server is up,
i.e. after minutes of an expensive multi-node allocation. Rendering one conversation here
turns that into an error in seconds. If EVERY conversation has an empty mask the step fails
rather than emitting an empty corpus, because "0 examples kept" and "your template has no
generation markers" deserve different exit messages.

Example:
    python prep_corpus.py \
      --dataset data/distillation/bespoke_stratos_17k_think.jsonl \
      --tokenizer output/retagged_student \
      --out-dir output --max-length 4096 \
      --think-policy keep --completion-boundary last_message \
      --eval-fraction 0.02 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from gb_steps_post_training.distillation import step_state, tokenizer_identity

# Roles a conversation may use. `tool` is included because the granite-4.2 template renders
# tool results as their own turn; a record using it is valid input, not a malformed one.
VALID_ROLES = ("system", "user", "assistant", "tool")

# Role names that mean a VALID_ROLES role under a different spelling, mapped rather than
# dropped -- but COUNTED in the manifest, because renaming roles is a transformation of the
# data and an operator is entitled to see how much of it happened.
#
# FOUND ON REAL DATA (LSF job 1137876): `data/distillation/en_sft_4.1` spells tool results
# `tool_response`, and 10,665 of its messages use it. granite-4.2's chat template accepts
# only system/user/assistant/tool -- and its `tool` branch renders the content wrapped in
# literal `<tool_response>` tags (chat_template.jinja:176-178). So the two spellings are the
# same concept and the dataset simply predates the template's naming; dropping those
# conversations would discard every tool-result conversation in the corpus for a spelling.
#
# This map is deliberately tiny and explicit. An unrecognised role still fails as
# `bad_role:<r>`, because guessing at an unknown role is how a corpus ends up rendering
# something nobody intended.
ROLE_ALIASES = {"tool_response": "tool"}

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

THINK_POLICIES = ("keep", "strip", "require")
LENGTH_POLICIES = ("drop", "truncate")
BOUNDARIES = ("last_message", "all_assistant")

DOCUMENTS_POLICIES = ("drop", "keep")

MANIFEST_NAME = "corpus_manifest.json"
STEP_NAME = "distill-corpus-prep"


def _declared_outputs(out_dir: Path) -> list[str]:
    """What this step promises to leave behind, as filenames relative to out_dir.

    eval.jsonl is conditional -- it is written only when eval_fraction > 0 -- so it is included
    only if it exists. Listing it unconditionally would make every zero-eval run REFUSE on the
    absence of a file it never produces, which is the sort of check that gets deleted rather than
    fixed. The manifest and the row sidecar are both small enough to be digested, so a corpus
    whose manifest was edited by hand does NOT read as complete.
    """
    names = ["train.jsonl", ROWS_NAME, MANIFEST_NAME]
    if (out_dir / "eval.jsonl").exists():
        names.insert(1, "eval.jsonl")
    return names


ROWS_NAME = "corpus_rows.jsonl"
ROW_ID_SCHEME = "blake2b-128/canonical-json"
# The key the id is written under when --emit-row-id is set. Named once, because it has to
# agree in three places that are edited at different times: what prep writes, what row_id()
# excludes from its own preimage, and what the trainer-side manifest reads back.
ROW_ID_FIELD = "row_id"


def template_renders_documents(tok) -> bool:
    """Does THIS tokenizer's chat template actually put `documents` into the prompt?

    The whole case for dropping grounded records rests on the template silently discarding
    them (audit-corpus-renderability.py: 7,013 of 811,172 on the deliverable corpus), which
    makes the assistant's answer reference text the student will never see -- measured at a
    45.1% floor for verbatim >=12-word lifts from the discarded document. That is training
    hallucination on purpose.

    But that is a fact about a TEMPLATE, and templates are the thing this project patches
    most often. If a future retag renders documents properly, dropping those records would
    throw away 0.86% of the corpus to fix a defect that no longer exists. So the premise is
    re-measured on every run instead of being trusted: render one synthetic record whose
    document contains a sentinel that cannot occur by chance, and look for it.

    Returns True if the sentinel survives into the rendered text. A template that raises on
    `documents` counts as NOT rendering them, which is the conservative answer -- it is the
    same outcome for the student either way.
    """
    sentinel = "ZQX-DOCUMENT-SENTINEL-8f21"
    rec = [
        {"role": "user", "content": "summarise"},
        {"role": "assistant", "content": "ok"},
    ]
    try:
        out = tok.apply_chat_template(
            rec,
            tokenize=False,
            add_generation_prompt=False,
            documents=[{"title": "t", "text": sentinel}],
        )
    except Exception:
        return False
    if isinstance(out, list):
        out = out[0] if out else ""
    return sentinel in str(out)


def row_id(record: dict) -> str:
    """A stable content-addressed id for one conversation record.

    WHY THIS IS NEEDED AT ALL: the source corpus has NO id field. Its rows are exactly
    `messages` / `tools` / `documents` (measured over all 2,000 probe rows and the
    deliverable corpus's schema), so "which data point was used during training" has no
    answer until we manufacture one. A row's own content is the only identity available,
    which makes a content hash not a workaround but the only correct key: it is stable
    across reruns, across machines, and across a reshuffle, and it needs no counter that
    a resumed or sharded run could desynchronise.

    Canonical JSON (sorted keys, no incidental whitespace) so that two records differing
    only in key order or serialisation hash the same -- otherwise the same conversation
    read from two files would get two ids and the manifest would claim to have trained on
    something it did not. `ensure_ascii=False` so the bytes hashed are the text's own
    UTF-8, not an escaping artefact that a future writer could change without changing
    the data.

    blake2b truncated to 128 bits: 16 bytes is ~2e-29 collision probability over a
    million rows, which is far below the rate at which any other part of this pipeline is
    wrong, and it keeps the sidecar readable. Not sha256 only because the digest_size
    parameter makes the truncation explicit rather than a slice someone later "tidies".

    ROW_ID_FIELD IS EXCLUDED FROM THE HASH, and that is what makes the id usable rather
    than merely present. With --emit-row-id the id is written INTO the emitted record so it
    travels with the data all the way to the trainer; if the id were part of its own preimage
    that stamp would be impossible (the value would have to be known before it was computed),
    and hashing a row of train.jsonl would give an answer that matched nothing. Excluding the
    field instead makes two useful things true: stamping is idempotent, and anyone holding
    train.jsonl can recompute a row's id and check it against the one stamped there. That is
    a verifiable claim about provenance instead of a number to be trusted.
    """
    if ROW_ID_FIELD in record:
        record = {k: v for k, v in record.items() if k != ROW_ID_FIELD}
    canon = json.dumps(
        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(canon, digest_size=16).hexdigest()


class PrepError(Exception):
    """A corpus the step refuses to build. Message is the operator-facing explanation."""


class AlreadyDone(Exception):
    """This exact corpus is already in out_dir. Carries the manifest that was found.

    Distinct from PrepError because the two mean opposite things to a restarted recipe: PrepError
    is "stop", this is "walk past step 1 and get on with step 2". Before this existed the shell
    wrapper refused on the mere presence of train.jsonl (prep-corpus.sh:108), which is the same
    answer for "identical corpus already built" and "different policies, do not overwrite" -- and
    a preemption during training could not restart the recipe without a human deleting 3.1 GB of
    correct output.
    """

    def __init__(self, manifest: dict, lines: list[str]) -> None:
        super().__init__("; ".join(lines))
        self.manifest = manifest
        self.lines = lines


def expectation(args: argparse.Namespace, identity: str) -> dict:
    """Everything that changes the bytes this step writes, and nothing that does not.

    The contract is the same one `policies` has in the manifest, and it is deliberately the
    SUPERSET of it: `policies` answers "how was this corpus shaped", while an expectation must also
    pin WHICH dataset and WHICH tokenizer, because the same policies over a different input are a
    different corpus. --hf-home is excluded on purpose: it moves a cache, not an output.

    tokenizer_identity rather than the tokenizer PATH, for the reason the manifest already uses it:
    a path can be repointed at a retagged directory with the same name, and it is the vocabulary
    that decides the token counts and the drop set.
    """
    return {
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_config": args.dataset_config or None,
        "tokenizer_identity": identity,
        "policies": {
            "max_length": args.max_length,
            "length_policy": args.length_policy,
            "think_policy": args.think_policy,
            "completion_boundary": args.completion_boundary,
            "min_messages": args.min_messages,
            "documents_policy": args.documents_policy,
            "emit_row_id": args.emit_row_id,
        },
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "max_examples": args.max_examples,
        "shard": {
            "index": int(getattr(args, "shard_index", 0) or 0),
            "count": int(getattr(args, "shard_count", 1) or 1),
        },
    }


# ------------------------------------------------------------------ loading


def load_records(dataset: str, split: str, config: str, hf_home: str) -> Iterator[dict]:
    """Yield raw records from a JSONL file or an HF dataset id.

    A local path wins over a hub id when both could match, and it is checked FIRST rather
    than by catching a hub error: a typo'd path that happens to look like `org/name` would
    otherwise become an outbound network call, which on this cluster means a long timeout
    instead of an immediate "no such file".
    """
    path = Path(dataset)
    if path.is_file():
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise PrepError(
                        f"{dataset}:{lineno} is not valid JSON: {exc}"
                    ) from exc
        return
    if path.exists():
        raise PrepError(
            f"--dataset {dataset} exists but is not a file. "
            "Pass the .jsonl itself, or an HF dataset id."
        )
    # Deferred import: the hub path is the rarer one and datasets is a heavy import that a
    # local-JSONL run should not pay for (and, offline, should not need installed).
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PrepError(
            f"--dataset {dataset} is not a local file, so it is treated as an HF dataset "
            f"id, but `datasets` is not importable: {exc}"
        ) from exc
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    ds = load_dataset(dataset, config or None, split=split)
    for row in ds:
        yield dict(row)


def load_tokenizer(tokenizer_dir: Path):
    """PreTrainedTokenizerFast built straight from tokenizer.json, with the .jinja template.

    Never AutoTokenizer -- see the module docstring. Returns (tokenizer, template_source).
    """
    from transformers import PreTrainedTokenizerFast

    tok_file = tokenizer_dir / "tokenizer.json"
    if not tok_file.is_file():
        raise PrepError(
            f"{tok_file} is missing. --tokenizer must be a directory holding a "
            "fast tokenizer (a model dir or distill-tokenizer-align's "
            "retagged_student / *_overlay)."
        )
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))

    jinja = tokenizer_dir / "chat_template.jinja"
    if jinja.is_file():
        tok.chat_template = jinja.read_text()
        return tok, str(jinja)
    # transformers 5.x treats chat_template.jinja as canonical, but a tokenizer_config from
    # an older export may still carry the template inline; accept it rather than failing on
    # a corpus that could legitimately be built.
    cfg = tokenizer_dir / "tokenizer_config.json"
    if cfg.is_file():
        inline = json.loads(cfg.read_text()).get("chat_template")
        if inline:
            tok.chat_template = inline
            return tok, f"{cfg} (inline chat_template)"
    raise PrepError(
        f"no chat template in {tokenizer_dir} (looked for chat_template.jinja and an inline "
        "chat_template in tokenizer_config.json). Lengths and the completion boundary are "
        "both measured on the RENDERED conversation, so this step cannot proceed without "
        "one. distill-tokenizer-align installs it via --chat-template."
    )


# ------------------------------------------------------------------ normalising


def normalise(
    record: dict,
    *,
    think_policy: str,
    boundary: str,
    min_messages: int,
    stats: dict | None = None,
    documents_policy: str = "keep",
) -> tuple[dict | None, str]:
    """Return (record, "") or (None, reason). Pure: does not touch the tokenizer.

    Kept separate from the length check so the cheap structural rejections happen before
    any rendering -- and so the reasons are testable without a tokenizer at all.
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "no_messages"
    if len(messages) < min_messages:
        return None, "too_few_messages"

    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None, "malformed_message"
        role, content = msg.get("role"), msg.get("content")
        if role in ROLE_ALIASES:
            if stats is not None:
                stats[f"role_alias:{role}->{ROLE_ALIASES[role]}"] = (
                    stats.get(f"role_alias:{role}->{ROLE_ALIASES[role]}", 0) + 1
                )
            role = ROLE_ALIASES[role]
        if role not in VALID_ROLES:
            return None, f"bad_role:{role}"
        if not isinstance(content, str):
            # tool_calls-only assistant turns exist in some corpora; they are legitimate
            # data but not something this step knows how to length-check or strip think
            # tags from, so they are refused loudly rather than silently coerced to "".
            return None, "non_string_content"
        out.append({"role": role, "content": content})

    assistants = [m for m in out if m["role"] == "assistant"]
    if not assistants:
        return None, "no_assistant_turn"
    if not any(m["role"] == "user" for m in out):
        return None, "no_user_turn"

    # THE COMPLETION-BOUNDARY CONTRACT, enforced here rather than assumed. Under
    # last_message_only GOLD supervises exactly the final turn, so a conversation whose
    # final turn is NOT the assistant's contributes no loss at all -- it would be trained
    # on as a no-op, wasting the sample and skewing the reported example count.
    if boundary == "last_message" and out[-1]["role"] != "assistant":
        return None, "last_message_not_assistant"

    has_think = any("<think>" in m["content"] for m in assistants)
    if think_policy == "require" and not has_think:
        return None, "no_think_block"
    if think_policy == "strip":
        for m in out:
            if m["role"] == "assistant":
                m["content"] = THINK_RE.sub("", m["content"]).strip()
        if any(not m["content"] for m in out if m["role"] == "assistant"):
            return None, "empty_after_strip"

    kept = {"messages": out}
    # Pass through the two optional fields the granite chat templates consume. Dropping
    # them would silently change what the rendered prompt looks like relative to the source
    # dataset, which is the kind of difference that shows up only as a worse eval.
    #
    # COERCED, NOT FORWARDED VERBATIM, and that distinction cost a job (LSF 1137876, where
    # 395,007 of 405,672 en_sft_4.1 records died as `template_error:ValueError`). That
    # dataset stores `tools` as the JSON *string* `"[]"`, not as a list.
    # `apply_chat_template` iterates `tools` and demands each element be a dict or a
    # callable, so a string is iterated CHARACTER BY CHARACTER -- `'['` is neither, and it
    # raises. Nothing about the message says "your tools field is a string", which is why
    # the error is now surfaced verbatim (see `measure`).
    #
    # An empty tools/documents list is OMITTED rather than passed as `[]`. On this template
    # the two are equivalent (the tool-calling preamble is gated on truthiness), but a
    # record whose only difference from a plain conversation is an empty list should not
    # depend on a template's falsiness handling to render identically.
    for extra in ("documents", "tools"):
        raw = record.get(extra)
        if raw is None:
            continue
        # `documents_policy` is checked BEFORE the parse below, but only for a value that
        # parses to a non-empty list -- a record with `documents: []` or `"[]"` is not a
        # grounded record and must not be counted as one. That is why this cannot simply
        # test truthiness of the raw field: the string "[]" is truthy.
        if extra == "documents" and documents_policy == "drop":
            probe = raw
            if isinstance(probe, str):
                try:
                    probe = json.loads(probe)
                except ValueError:
                    probe = None
            if isinstance(probe, list) and probe:
                return None, "documents_not_renderable"
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, f"bad_{extra}"
        if not isinstance(raw, list):
            return None, f"bad_{extra}"
        if not raw:
            continue
        if not all(isinstance(item, dict) for item in raw):
            return None, f"bad_{extra}"
        kept[extra] = raw
    return kept, ""


# ------------------------------------------------------------------ measuring


def last_mask_span(mask) -> int:
    """Length of the FINAL contiguous run of 1s in an assistant mask.

    WHY THIS EXISTS. granite-4.2's chat template wraps EVERY assistant turn in
    `{% generation %}` (chat_template.jinja:99,141,152,157), so `assistant_masks` covers all
    of them. GOLD under `last_message_only=True` -- the setting this step's default
    `completion_boundary: last_message` corresponds to -- computes loss on only the FINAL
    assistant turn. Summing the whole mask therefore reports more supervised tokens than
    the trainer will actually supervise, by exactly the earlier assistant turns.

    The error is invisible on single-turn data, which is why it survived the first scale run
    (LSF job 1137876): `bespoke_stratos_17k` is one user turn and one assistant turn, so
    last_message and all_assistant produced byte-identical corpora and identical token
    stats. On a multi-turn corpus the same code would have overstated the supervised token
    count without any signal that it had.
    """
    end = next((i for i in range(len(mask) - 1, -1, -1) if mask[i]), None)
    if end is None:
        return 0
    start = end
    while start > 0 and mask[start - 1]:
        start -= 1
    return end - start + 1


def measure(
    record: dict,
    tok,
    *,
    max_length: int,
    length_policy: str,
    boundary: str = "all_assistant",
    detail: dict | None = None,
    totals: dict | None = None,
) -> tuple[dict | None, str, int, int]:
    """Render + tokenize one record. Returns (record|None, reason, n_tokens, n_target).

    n_target is the number of tokens THE TRAINER WILL SUPERVISE under `boundary`: the final
    assistant turn for `last_message`, every assistant turn for `all_assistant`. A record
    with none of them is dropped -- it is a sample the trainer would compute no loss on.

    `totals`, when given, gets THIS record's all-assistant mask count under
    `mask_tokens_all_assistant` -- overwritten per call, not accumulated, so the caller can
    add it up over KEPT records only. Accumulating here would silently mix in the records
    that were then dropped for length, and the resulting "unused signal" figure would be
    compared against a kept-only total and be nonsense. The figure matters because the gap
    between it and total_target_tokens is the argument for `all_assistant` on multi-turn
    data, so it is measured rather than left to guesswork.
    """
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if record.get("tools") is not None:
        kwargs["tools"] = record["tools"]
    if record.get("documents") is not None:
        kwargs["documents"] = record["documents"]
    try:
        enc = tok.apply_chat_template([record["messages"]], **kwargs)
    except (
        Exception
    ) as exc:  # template errors are data-dependent, not programmer errors
        # The TYPE goes in the drop histogram (a bounded set, so the histogram stays
        # readable) and the first MESSAGE goes in `detail`, which build() reports and puts
        # in the manifest. Without the message, `template_error:ValueError` on 395,007
        # records says only "the template refused nearly everything" -- diagnosing it meant
        # a second job. One line of text would have been enough; now it is kept.
        if detail is not None:
            detail.setdefault(f"template_error:{type(exc).__name__}", str(exc)[:600])
        return None, f"template_error:{type(exc).__name__}", 0, 0

    ids = enc["input_ids"][0]
    mask = (enc.get("assistant_masks") or [[]])[0]
    n_tokens = len(ids)
    n_all = sum(mask)
    # The boundary scopes what is COUNTED, exactly as it scopes what the trainer optimises.
    n_target = n_all if boundary == "all_assistant" else last_mask_span(mask)
    if totals is not None:
        totals["mask_tokens_all_assistant"] = n_all

    # Tested on n_all, not n_target: an all-zero mask means the TEMPLATE has no generation
    # markers, which is a different fault from a boundary that happens to select nothing,
    # and build() reports the two differently.
    if n_all == 0:
        return None, "empty_assistant_mask", n_tokens, 0
    if n_tokens > max_length:
        if length_policy == "drop":
            return None, "over_max_length", n_tokens, n_target
        # truncate keeps the record but the caller is told, because truncation at the token
        # level cannot be reflected back into `messages` without re-detokenising -- so the
        # emitted record is the untruncated text and the trainer truncates it identically.
        # Recorded in the manifest as `truncated` so the count is not invisible.
        return record, "truncated", n_tokens, n_target
    return record, "", n_tokens, n_target


# ------------------------------------------------------------------ driver


def build(args: argparse.Namespace) -> dict[str, Any]:
    tok_dir = Path(args.tokenizer)
    tok, template_src = load_tokenizer(tok_dir)
    try:
        identity = tokenizer_identity.read(tok_dir)
    except tokenizer_identity.IdentityError as exc:
        raise PrepError(str(exc)) from exc
    if identity is None:
        raise PrepError(f"cannot determine a tokenizer identity for {tok_dir}")

    renders_documents = template_renders_documents(tok)
    if args.documents_policy == "drop" and renders_documents:
        print(
            f"WARNING: --documents-policy drop, but {tok_dir}'s chat template DOES render "
            "`documents`. The reason for dropping grounded records was that the template "
            "discarded them; on this template it does not, so dropping them throws away "
            "usable grounding. Re-read the policy before trusting this corpus.",
            file=sys.stderr,
        )
    if args.documents_policy == "keep" and not renders_documents:
        print(
            "WARNING: --documents-policy keep, and this template DISCARDS `documents`. "
            "Grounded records will be rendered without their grounding, so the assistant "
            "answer references text the student never sees -- measured at a 45.1% floor "
            "for verbatim >=12-word lifts. This is the hallucination-training case.",
            file=sys.stderr,
        )

    shard_count = int(getattr(args, "shard_count", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    if shard_count < 1:
        raise PrepError(f"--shard-count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise PrepError(f"--shard-index {shard_index} is outside 0..{shard_count - 1}")
    if shard_count > 1:
        # Both of these are refused rather than approximated, and for the same reason: they are
        # GLOBAL operations over the kept set, and a shard cannot see the kept set.
        #
        # --eval-fraction draws its split from a shuffle of this process's kept indices. Sharded,
        # that is K independent draws over K disjoint subsets -- still a valid split of the
        # corpus, but not the split (seed, n) names, so it would silently stop being
        # reproducible from the manifest. Prep the corpus sharded with no eval split and draw
        # the split once afterwards, or prep unsharded.
        if args.eval_fraction > 0:
            raise PrepError(
                "--eval-fraction with --shard-count > 1 would draw K independent splits over K "
                "disjoint subsets, so the manifest's (seed, eval_fraction) would no longer "
                "reproduce it. Prep sharded with --eval-fraction 0 and split afterwards."
            )
        # --max-examples means "stop after N kept". Per shard that is N*K, and which rows they
        # are depends on each shard's own drop rate -- so the same command yields a different
        # corpus at a different shard count, which is exactly what a cap is used to avoid.
        if args.max_examples:
            raise PrepError(
                f"--max-examples {args.max_examples} with --shard-count {shard_count} would keep "
                f"up to {args.max_examples * shard_count} rows, chosen differently at every "
                "shard count. Cap the input before prep, or prep unsharded."
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Is this already built? Asked HERE -- after the tokenizer identity and the shard arguments
    # are known, since both are part of what "already built" means, and before the loop that
    # costs 0.7 h on the full corpus.
    want = expectation(args, identity)
    verdict = step_state.decide(out_dir, STEP_NAME, want, _declared_outputs(out_dir))
    if verdict.kind == step_state.SKIP:
        manifest_path = out_dir / MANIFEST_NAME
        raise AlreadyDone(json.loads(manifest_path.read_text()), verdict.lines)
    if verdict.kind == step_state.REFUSE:
        raise PrepError("\n  ".join(verdict.lines))

    drops: dict[str, int] = {}
    # Two side-channels out of the per-record helpers, both optional so neither changes the
    # helpers' contracts: `notes` counts transformations applied (role aliases), `detail`
    # keeps the first message behind each template-error type.
    notes: dict[str, int] = {}
    detail: dict[str, str] = {}
    totals: dict[str, int] = {}
    n_mask_all = 0
    kept: list[dict] = []
    # One entry per INPUT record, kept or dropped. This is the per-datapoint training
    # manifest; corpus_manifest.json stays aggregate. Two files rather than one because a
    # 811,172-row array inside the manifest would make the file that every consumer reads
    # for tokenizer_identity unreadable, and the aggregate counts are what most consumers
    # want. The sidecar is ~90 bytes/row, so ~73 MB at full corpus scale.
    rows: list[dict] = []
    lengths: list[int] = []
    targets: list[int] = []
    n_in = 0
    n_truncated = 0
    n_rendered = 0

    # Counts EVERY record read, including other shards'. n_in counts only this shard's, so
    # that the sidecar's one-entry-per-assigned-record invariant still means what it says.
    g_in = 0
    for raw in load_records(
        args.dataset, args.dataset_split, args.dataset_config, args.hf_home
    ):
        if shard_count > 1:
            mine = g_in % shard_count == shard_index
            g_in += 1
            if not mine:
                continue
        n_in += 1
        rid = row_id(raw)
        rec, reason = normalise(
            raw,
            think_policy=args.think_policy,
            boundary=args.completion_boundary,
            min_messages=args.min_messages,
            stats=notes,
            documents_policy=args.documents_policy,
        )
        if rec is None:
            drops[reason] = drops.get(reason, 0) + 1
            rows.append({"src_id": rid, "disposition": "dropped", "reason": reason})
            continue
        rec, reason, n_tok, n_tgt = measure(
            rec,
            tok,
            max_length=args.max_length,
            length_policy=args.length_policy,
            boundary=args.completion_boundary,
            detail=detail,
            totals=totals,
        )
        n_rendered += 1
        if rec is None:
            drops[reason] = drops.get(reason, 0) + 1
            rows.append({"src_id": rid, "disposition": "dropped", "reason": reason})
            continue
        if reason == "truncated":
            n_truncated += 1
        # Accumulated here, over kept records only -- see measure()'s docstring.
        n_mask_all += totals.get("mask_tokens_all_assistant", 0)
        # `tools` GOES BACK TO A STRING BEFORE IT IS EMITTED, and `documents` does not.
        # normalise() parses both so measure() can render them (a string tools field is
        # iterated character by character by the template -- LSF 1137876), but the TRAINER
        # reads the two fields differently: it json.loads `tools`
        # (custom_gold_trainer.py:1745, again at :3075) and forwards `documents` to the
        # template as-is (:1750). Emitting the parsed list therefore made the deliverable
        # corpus crash the trainer's own preprocessing on the first tools-bearing row --
        # 17.5% of en-sft-4.1-0.2-16K, none of them in the 2,000-row probe head, so every
        # run so far was green. Found by checks/collator-masking.py on job 1162592.
        # Serialised HERE, before row_id(), so the id still hashes exactly the bytes that
        # reach train.jsonl and a holder of that file can still recompute it.
        # Only when present: an absent column reads as null and the trainer defaults it.
        if isinstance(rec.get("tools"), list):
            rec["tools"] = json.dumps(rec["tools"], ensure_ascii=False)
        # `out_id` hashes the EMITTED record, not the source one, and both are kept.
        # normalise() TRANSFORMS records (role aliases, think stripping, tools parsed from
        # their JSON string), so the two ids genuinely differ for any transformed row.
        # Recording only src_id would leave a holder of train.jsonl unable to look a row up;
        # recording only out_id would break the link back to the source dataset. Provenance
        # has to be traceable from BOTH ends or it answers only half the question.
        oid = row_id(rec)
        if args.emit_row_id:
            # THE ID TRAVELS WITH THE ROW. This is the difference between a manifest that
            # says what prep emitted and one that can say what the TRAINER consumed.
            #
            # The trainer applies its own row filter, and an arm-dependent one:
            # custom_gold_trainer.py:1824 additionally drops prompts over
            # `max_length - max_completion_length`, but only when lmbda != 0.0. Predicting
            # that from here would mean re-implementing the trainer's prompt render
            # (add_generation_prompt=True, enable_thinking=False, its tokenizer copy, its
            # template) in a second place that can drift from the first -- and a provenance
            # record that has silently drifted is worse than none, because it still looks
            # authoritative. Carrying the id instead lets the trainer answer from the dataset
            # it actually built, which is the only place the question has a true answer.
            #
            # Safe as an extra column: the tokenize map (:1814) sets no remove_columns, and
            # select_columns runs only under packing, which these configs do not use -- so
            # the field survives to trainer.train_dataset. It is also inert for rendering,
            # which reads only messages/tools/documents/chat_template_kwargs.
            rec[ROW_ID_FIELD] = oid
        rows.append(
            {
                "src_id": rid,
                "out_id": oid,
                "disposition": "kept",
                "reason": reason or "",
                "tokens": n_tok,
                "target_tokens": n_tgt,
                "kept_index": len(kept),
            }
        )
        kept.append(rec)
        lengths.append(n_tok)
        targets.append(n_tgt)
        if args.max_examples and len(kept) >= args.max_examples:
            break

    # An all-zero assistant mask across the board is a TEMPLATE fault, not a data fault, and
    # it is the single most likely way this step is misconfigured -- so it gets its own
    # error. Reported before the "kept 0" check because it explains it.
    if n_rendered and drops.get("empty_assistant_mask", 0) == n_rendered:
        raise PrepError(
            f"every one of {n_rendered} rendered conversations produced an EMPTY assistant "
            f"mask. That is a chat-template fault, not a data fault: the template at "
            f"{template_src} has no {{% generation %}} markers, so nothing marks the "
            "completion span. GOLD would fail on this at sft.py:909 after loading the model "
            "and starting vLLM. Use a template with generation markers (see "
            "templates/chatml_granite_42_generation.jinja)."
        )
    if not kept:
        raise PrepError(
            f"0 of {n_in} records survived filtering. Drop reasons: {drops or 'none'}. "
            + "".join(f"First {k}: {v} " for k, v in detail.items())
            + "Check --max-length, --think-policy and --completion-boundary against the "
            "dataset's actual shape."
        )

    # Deterministic split from an explicit seed. Shuffling INDICES rather than the records
    # keeps the operation identical whether the corpus fits in memory comfortably or not,
    # and makes the split reproducible from (seed, n) alone.
    order = list(range(len(kept)))
    random.Random(args.seed).shuffle(order)
    n_eval = int(round(len(order) * args.eval_fraction))
    if args.eval_fraction > 0 and n_eval == 0:
        # Silently emitting an empty eval split would look like "eval was requested and
        # produced nothing", which reads as a bug downstream. Round up to one instead.
        n_eval = 1
    if n_eval >= len(order):
        raise PrepError(
            f"--eval-fraction {args.eval_fraction} would put all {len(order)} kept examples "
            "in the eval split, leaving nothing to train on."
        )
    eval_idx, train_idx = order[:n_eval], order[n_eval:]

    splits: dict[str, dict] = {}
    for name, idx in (("train", train_idx), ("eval", eval_idx)):
        if name == "eval" and not idx:
            continue
        dest = out_dir / f"{name}.jsonl"
        with dest.open("w") as fh:
            for i in sorted(idx):  # sorted: stable file order for a given index set
                fh.write(json.dumps(kept[i], ensure_ascii=False) + "\n")
        splits[name] = {"path": str(dest.resolve()), "examples": len(idx)}

    # The split is drawn AFTER the loop, so the sidecar's kept rows learn their split here.
    # Keyed by kept_index rather than by position in `rows`, because `rows` also holds the
    # dropped records and the two lists are deliberately different lengths.
    by_kept = {r["kept_index"]: r for r in rows if r["disposition"] == "kept"}
    for name, idx in (("train", train_idx), ("eval", eval_idx)):
        for i in idx:
            by_kept[i]["split"] = name

    manifest = {
        # THE KEY distill-gold-train COMPARES. Same name, same scheme, one function.
        "tokenizer_identity": identity,
        "tokenizer_path": str(tok_dir.resolve()),
        "chat_template_source": template_src,
        # An OBSERVATION about the template, not a policy -- which is why it sits here and
        # not in `policies`, whose contract is "every knob that changed the output" and is
        # asserted exactly by the tests. Measured on this run's tokenizer rather than
        # assumed (template_renders_documents()); it is what makes documents_policy
        # interpretable, since "drop" is only the right call while this is false.
        "template_renders_documents": renders_documents,
        "format": "messages-jsonl",
        "tokenized": False,  # spelled out: the corpus holds text, not ids
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_config": args.dataset_config or None,
        "policies": {
            "max_length": args.max_length,
            "length_policy": args.length_policy,
            "think_policy": args.think_policy,
            "completion_boundary": args.completion_boundary,
            "min_messages": args.min_messages,
            "documents_policy": args.documents_policy,
            "emit_row_id": args.emit_row_id,
        },
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        # Always present, even for an unsharded run: a consumer must be able to tell "this is
        # the whole corpus" from "this is a piece of one" without inferring it from a missing
        # key. merge_shards.py refuses a set that does not agree on `count` or that is missing
        # an `index`.
        "shard": {"index": shard_index, "count": shard_count},
        "counts": {
            "input": n_in,
            "rendered": n_rendered,
            "kept": len(kept),
            "truncated": n_truncated,
            "dropped": sum(drops.values()),
            "drop_reasons": dict(sorted(drops.items())),
            # Present only when something fired, so an empty-clean run keeps a clean
            # manifest. `transformations` is what was CHANGED (as opposed to dropped), and
            # it belongs in the manifest for the same reason drop_reasons does: a consumer
            # comparing this corpus to its source dataset needs to know.
            **({"transformations": dict(sorted(notes.items()))} if notes else {}),
            **({"template_errors": dict(sorted(detail.items()))} if detail else {}),
        },
        "token_stats": {
            "total_tokens": sum(lengths),
            "total_target_tokens": sum(targets),
            "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 1),
            # Scoped by completion_boundary -- these are the tokens the trainer will
            # actually compute loss on, not every generation-marked token. See
            # last_mask_span().
            "mean_target_tokens": round(sum(targets) / len(targets), 1),
            # Every assistant turn, regardless of boundary. Equal to the two above under
            # `all_assistant`; under `last_message` the gap is the supervised signal the
            # boundary discards, which is the number a recipe needs to choose between them.
            "total_mask_tokens_all_assistant": n_mask_all,
        },
        "splits": splits,
    }
    # Written BEFORE the manifest, and the manifest carries its sha256. So a manifest that
    # exists is a manifest whose sidecar is already complete and whose digest was taken over
    # the finished file -- a consumer that finds both can verify it has the pair that were
    # produced together, rather than a sidecar from one run beside a manifest from another.
    rows_path = out_dir / ROWS_NAME
    with rows_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    h = hashlib.sha256()
    with rows_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)

    manifest["rows"] = {
        "path": str(rows_path.resolve()),
        "sha256": h.hexdigest(),
        "entries": len(rows),
        "row_id_scheme": ROW_ID_SCHEME,
        # Stated positively rather than left to be inferred from the corpus: a consumer
        # asking "can I trace what was trained on?" gets a yes/no here instead of having to
        # open train.jsonl and look for a field.
        "id_field": ROW_ID_FIELD if args.emit_row_id else None,
        # A row is in the training set iff disposition == "kept" AND split == "train".
        # Spelled out because "kept" alone is the wrong answer whenever eval_fraction > 0,
        # and that is a mistake a consumer makes once and never notices.
        "training_rows": sum(
            1 for r in rows if r["disposition"] == "kept" and r.get("split") == "train"
        ),
    }
    # An entry per input record, or the sidecar does not mean what the key above says it
    # means. Cheap to assert, and it is the invariant that the whole file rests on.
    if len(rows) != n_in and not args.max_examples:
        raise PrepError(
            f"sidecar has {len(rows)} entries for {n_in} input records assigned to shard "
            f"{shard_index}/{shard_count} -- a record was "
            "neither kept nor recorded as dropped, so the per-row manifest is incomplete "
            "and must not be published as one"
        )

    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    # LAST, and atomically. The marker is what a restarted recipe reads to decide whether to walk
    # past this step, so it must not be able to exist beside a half-written corpus -- which is why
    # it goes after the manifest, which itself goes after train.jsonl and the sidecar.
    step_state.write_marker(out_dir, STEP_NAME, want, _declared_outputs(out_dir))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="path to a conversations .jsonl, or an HF dataset id",
    )
    p.add_argument(
        "--dataset-split", default="train", help="HF split (ignored for a file)"
    )
    p.add_argument("--dataset-config", default="", help="HF config name, if any")
    p.add_argument(
        "--tokenizer",
        required=True,
        help="tokenizer directory whose lengths define this corpus -- normally "
        "distill-tokenizer-align's retagged_student",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="must match distill-gold-train's max_length; a corpus filtered at a "
        "different length silently contains examples the trainer truncates",
    )
    p.add_argument(
        "--length-policy",
        choices=LENGTH_POLICIES,
        default="drop",
        help="drop: refuse over-length conversations (default -- a truncated "
        "assistant turn teaches an unterminated answer). truncate: keep "
        "them and let the trainer truncate, counted in the manifest.",
    )
    p.add_argument(
        "--think-policy",
        choices=THINK_POLICIES,
        default="keep",
        help="keep: leave <think> blocks in assistant turns. strip: remove them. "
        "require: drop conversations that have none (reasoning-only corpus).",
    )
    p.add_argument(
        "--completion-boundary",
        choices=BOUNDARIES,
        default="last_message",
        help="last_message: the final turn must be the assistant's, matching "
        "GOLD's last_message_only. all_assistant: every assistant turn is "
        "supervised.",
    )
    p.add_argument("--eval-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="stop after N KEPT examples (0 = no limit). For smoke runs.",
    )
    p.add_argument("--min-messages", type=int, default=2)
    # Default "drop", which is NOT the identity behaviour, and deliberately so: on every
    # template in this project the grounding is silently discarded, so "keep" is the option
    # that quietly corrupts the corpus while looking like the safe choice. A default that
    # has to be overridden to do the harmful thing is the right way round. Both settings are
    # recorded in the manifest and both warn when they disagree with the measured template.
    # Default OFF, and deliberately not yet the default even though the manifest wants it.
    # Adding a column to the emitted corpus changes what reaches the collator, and no job has
    # yet proven the trainer tolerates it end to end. The evidence is encouraging rather than
    # conclusive -- custom_gold_trainer.py:1771 already carries `tools` through as a string
    # column for exactly this Arrow-compatibility reason -- so this flips to default-on once a
    # real run has trained with it, not before.
    # BooleanOptionalAction and not store_true, so steps/distill-corpus-prep/step-template.yaml
    # can pass the flag UNCONDITIONALLY -- same reason as build_overlay.py's --verify. A
    # store_true has no negative form, so the template's only option is a conditional that
    # renders to nothing, which leaves the preceding line's `\` continuing into a blank line.
    # That parses today and is one edit away from swallowing the next argument. Nothing calls
    # this with a value, so `--emit-row-id` keeps meaning exactly what it meant.
    p.add_argument(
        "--emit-row-id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write each emitted row's content id into the row itself, under "
        f"{ROW_ID_FIELD!r}, so the trainer can report which rows it actually "
        "consumed after its own arm-dependent filtering. Excluded from its own "
        "hash, so the stamp is idempotent and any holder of train.jsonl can "
        "recompute and verify it.",
    )
    p.add_argument(
        "--documents-policy",
        choices=DOCUMENTS_POLICIES,
        default="drop",
        help="drop: discard records with non-empty `documents`, because the "
        "chat template discards the grounding and the answer depends on it "
        "(7,013/811,172 = 0.86% on the deliverable corpus, 45.1% floor on "
        "verbatim dependence). keep: render them anyway.",
    )
    # ---------------------------------------------------------------- sharding
    # A full-corpus prep was measured at 303 input rows/s single-process -- ~0.7 h for
    # 811,172 rows -- so sharding is NOT needed at this corpus size and was deliberately not
    # built when that measurement came in. It is built now for the size AFTER this one: the
    # cost is linear in rows and entirely CPU-bound in the tokenizer, so a 5x corpus is a 3.5 h
    # serial job on a preemptable queue, which is a job that gets killed rather than a job that
    # finishes.
    #
    # ASSIGNMENT IS BY MODULO OVER THE INPUT STREAM, not by byte range. Every shard reads the
    # whole file and tokenizes only its own rows; reading 3.4 GB is IO-bound and cheap next to
    # the tokenization, and modulo needs no index, no line-offset table, and no assumption that
    # the input is seekable -- so the same code path works for an HF dataset as for a jsonl.
    #
    # The property worth having, and the reason for the round-robin merge in merge_shards.py:
    # K shards merged are BYTE-IDENTICAL to one unsharded run. Shard s holds input rows
    # s, s+K, s+2K, ..., each in input order, so a K-way lockstep merge reconstructs the global
    # input order exactly. That turns "is the sharded path correct?" into a diff, which is a
    # question a test can answer, instead of a distribution argument.
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="split the input across this many independent processes (default 1, "
        "i.e. no sharding). Merge with merge_shards.py.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="which shard THIS process handles, 0-based. Takes input records where "
        "index %% shard-count == shard-index.",
    )
    p.add_argument("--hf-home", default="", help="HF_HOME for the hub path")
    return p


def out_dir_marker(out_dir: str) -> Path:
    return Path(out_dir) / step_state.MARKER_NAME


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.eval_fraction < 1.0:
        print("ERROR: --eval-fraction must be in [0.0, 1.0)", file=sys.stderr)
        return 2
    try:
        m = build(args)
    except AlreadyDone as exc:
        # rc 0. A recipe restarted after a preemption in a later step must be able to walk past
        # this one, and "the work you asked for is already here" is a success, not a failure.
        print(f"=== corpus already built -> {args.out_dir}")
        for line in exc.lines:
            print(f"  {line}")
        c = exc.manifest["counts"]
        print(
            f"  kept {c['kept']} of {c['input']} "
            f"(dropped {c['dropped']}, truncated {c['truncated']})"
        )
        print(f"  tokenizer identity : {exc.manifest['tokenizer_identity']}")
        print(f"  delete {out_dir_marker(args.out_dir)} to rebuild deliberately")
        return 0
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    c, t, pol = m["counts"], m["token_stats"], m["policies"]
    print(f"\n=== corpus built -> {args.out_dir}")
    print(f"  tokenizer identity : {m['tokenizer_identity']}")
    print(
        f"  kept {c['kept']} of {c['input']} "
        f"(dropped {c['dropped']}, truncated {c['truncated']})"
    )
    for reason, n in c["drop_reasons"].items():
        print(f"    drop {reason}: {n}")
    for label, n in c.get("transformations", {}).items():
        print(f"    applied {label}: {n} messages")
    for kind, msg in c.get("template_errors", {}).items():
        print(f"    first {kind}: {msg}")
    print(
        f"  tokens: mean {t['mean_tokens']} max {t['max_tokens']} "
        f"target-mean {t['mean_target_tokens']}"
    )
    unused = t["total_mask_tokens_all_assistant"] - t["total_target_tokens"]
    if unused > 0:
        pct = 100.0 * unused / t["total_mask_tokens_all_assistant"]
        print(
            f"  NOTE: completion_boundary={pol['completion_boundary']} leaves "
            f"{unused} assistant tokens ({pct:.1f}%) unsupervised. "
            f"completion_boundary=all_assistant would use them."
        )
    for name, s in m["splits"].items():
        print(f"  {name}: {s['examples']} -> {s['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
