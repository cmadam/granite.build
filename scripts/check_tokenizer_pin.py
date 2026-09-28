#!/usr/bin/env python3
"""Verify a checkpoint's tokenizer_class pin actually took effect.

granite 4.x checkpoints declare tokenizer_class "GPT2Tokenizer" over a trained
Sequence[Split, ByteLevel] pre_tokenizer. AutoTokenizer honours that declaration,
instantiates the slow GPT2 converter, and discards the trained Split -- so the
model is evaluated on a segmentation it was never trained on. Repinning
tokenizer_class to PreTrainedTokenizerFast makes AutoTokenizer load
tokenizer.json verbatim instead.

This script checks the pin from the consumer's side: it loads the checkpoint the
way an eval harness does and asserts the resulting class is not a GPT2Tokenizer,
then proves the segmentation matches tokenizer.json read directly.

Usage:
    python3 check_tokenizer_pin.py <checkpoint_dir> [<reference_dir> ...]

Exit status is 0 only if every directory passes.
"""

import json
import pathlib
import sys

PROBES = [
    "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    "The quick brown fox jumps over 42 lazy dogs.",
    "  leading and   internal   whitespace\t\tand a tab",
    "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>",
    "Ceci n'est pas une pipe — naïve café, 日本語テキスト, 🚀",
    "1234567890 0x1F 3.14159e-7",
]


def check(path: pathlib.Path) -> bool:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    print(f"\n=== {path}")
    ok = True

    cfg_path = path / "tokenizer_config.json"
    declared = None
    if cfg_path.is_file():
        declared = json.loads(cfg_path.read_text()).get("tokenizer_class")
    print(f"  tokenizer_config.json tokenizer_class : {declared!r}")

    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False)
    loaded = type(tok).__name__
    print(f"  AutoTokenizer resolved class          : {loaded}")
    print(f"  is_fast                               : {tok.is_fast}")

    # The actual gate the user asked for.
    if loaded.startswith("GPT2Tokenizer"):
        print(f"  FAIL: AutoTokenizer still returns {loaded}; the pin did not take.")
        ok = False
    elif not tok.is_fast:
        print(
            f"  FAIL: {loaded} is a slow tokenizer; it cannot be reading tokenizer.json."
        )
        ok = False
    else:
        print(f"  PASS: not a GPT2Tokenizer.")

    # A class name is a label. This is the behavioural proof: compare the ids
    # AutoTokenizer produces against tokenizer.json loaded with no transformers
    # in the path at all.
    tj = path / "tokenizer.json"
    if not tj.is_file():
        print("  WARN: no tokenizer.json; skipping segmentation comparison.")
        return ok

    raw = json.loads(tj.read_text())
    pre = raw.get("pre_tokenizer") or {}
    kinds = (
        [p.get("type") for p in pre["pretokenizers"]]
        if pre.get("type") == "Sequence"
        else [pre.get("type")]
    )
    print(f"  tokenizer.json pre_tokenizer          : {pre.get('type')} {kinds}")
    if "Split" not in kinds:
        print("  NOTE: no trained Split in this tokenizer; the pin is a no-op here.")

    ground = Tokenizer.from_file(str(tj))
    mismatches = 0
    for probe in PROBES:
        want = ground.encode(probe, add_special_tokens=False).ids
        got = tok(probe, add_special_tokens=False)["input_ids"]
        if want != got:
            mismatches += 1
            print(f"  MISMATCH on {probe[:48]!r}")
            print(
                f"    tokenizer.json : {want[:24]}{' ...' if len(want) > 24 else ''}  (len {len(want)})"
            )
            print(
                f"    AutoTokenizer  : {got[:24]}{' ...' if len(got) > 24 else ''}  (len {len(got)})"
            )
    if mismatches:
        print(
            f"  FAIL: {mismatches}/{len(PROBES)} probes segment differently than tokenizer.json."
        )
        ok = False
    else:
        print(f"  PASS: all {len(PROBES)} probes match tokenizer.json exactly.")

    # Not pass/fail, but worth seeing: a truncation block baked into
    # tokenizer.json is a training-time leak, and a post_processor that injects
    # special tokens would change every eval prompt.
    trunc = raw.get("truncation")
    if trunc:
        print(f"  NOTE: tokenizer.json carries truncation {trunc}")
    pp = raw.get("post_processor")
    if pp:
        specials = pp.get("special_tokens") or {}
        print(
            f"  NOTE: post_processor {pp.get('type')}, special_tokens={list(specials)}"
        )

    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    results = {}
    for arg in sys.argv[1:]:
        p = pathlib.Path(arg)
        if not p.is_dir():
            print(f"\n=== {p}\n  FAIL: not a directory")
            results[arg] = False
            continue
        try:
            results[arg] = check(p)
        except Exception as exc:  # a load failure is itself a failed check
            print(f"  FAIL: {type(exc).__name__}: {exc}")
            results[arg] = False

    print("\n=== summary")
    for arg, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {arg}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
