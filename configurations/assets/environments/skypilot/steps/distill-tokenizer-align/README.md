# distill-tokenizer-align

Makes a raw Granite base student able to speak the teacher's ChatML, and emits the three
tokenizer artifacts every later distillation step reads.

| | |
|---|---|
| **Type** | `data_processing` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; this step builds none) |
| **GPU** | none needed — two tokenizers and one embedding matrix, on CPU |
| **Emits** | `retagged_student`, `teacher_overlay`, `student_overlay` (all `type: model`) |

## Why this step exists

A Granite directory's `tokenizer_config.json` declares `tokenizer_class: "GPT2Tokenizer"`.
`AutoTokenizer` honours that, constructs that class, and **the class imposes its own plain
`ByteLevel` pre_tokenizer over the one stored in `tokenizer.json`**. Nothing errors — the text
simply segments differently (26.1 vs 3.29 PPL/token, measured upstream).

The 4.1 base **student** is where this bites: its `pre_tokenizer` is
`Sequence[Split(regex), ByteLevel]`, and the override discards that `Split`. Two fixes that
look obvious and are both wrong:

- **It is not the legacy `vocab.json` / `merges.txt` sidecars.** Under transformers 5.8 they are
  inert. The overlay excludes them for hygiene, not protection.
- **Deleting `tokenizer_class` is not sufficient.** With a `config.json` present,
  `model_type: granite` resolves through `TOKENIZER_MAPPING_NAMES` to `GPT2Tokenizer` anyway.
  So the overlay builder **pins** the key to `PreTrainedTokenizerFast`.

## The three outputs, and who consumes them

| Artifact | What it is | Consumer |
|---|---|---|
| `retagged_student` | the base student re-embedded onto the teacher's tokenizer, with a chat template installed | `gold-distill`'s `model_name_or_path`; the tokenizer `distill-corpus-prep` tokenizes with |
| `teacher_overlay` | the teacher's tokenizer files only, `tokenizer_class` pinned | the teacher tokenizer for the eval steps — kept separate from the teacher MODEL path on purpose |
| `student_overlay` | the *pre-retag* student's tokenizer files, pinned the same way | `distill-corpus-prep`, as the trustworthy comparison point when it asserts one-tokenizer-per-run |

**All three must be declared in the consuming target's `outputs:`.** An undeclared output makes
the buildrun resolver drop the `NEWARTIFACT` event, and the target then completes with no
output *and* no error — a silent failure that costs a full training run.

## Source delivery

This step ships **no image and no Python of its own** beyond `src/run-align.sh`. The work is done
by `gb_steps_post_training.distillation`, which is delivered at run time from a checkout on the
shared filesystem — the same arrangement `gold-distill` uses for the kd-sandbox trainer.

**No credential reaches the container.** That is deliberate: every other step in this repo either
clones nothing or clones a public repo unauthenticated, and granite.build's own private-repo
access clones server-side on the gbserver host.

```yaml
code_config:
  code_dir: "/proj/granite-build/g4os/gb-steps-collection-post-training"
  expect_ref: "70c1550a171aa8e09a9ad9047a5bf763c39e8579"
```

`expect_ref` is checked against the checkout's actual `HEAD` and the step **fails loudly** on a
mismatch, because a silently-moved shared checkout is how two runs that report the same pin end
up having trained on different code. Uncommitted changes are not fatal but are warned about and
recorded as `distill_code_dirty` step metadata.

### Refreshing the checkout

The cluster account has its own GHE key, so this needs no secret:

```bash
D=/proj/granite-build/g4os/gb-steps-collection-post-training
git -C "$D" fetch --all
git -C "$D" checkout <new-sha>
```

Then bump `expect_ref` in the recipe (or the step default) to the same SHA.

### The clone fallback

For a host with no `/proj`, set `repo` + `ref` and leave `code_dir` empty; the step clones at run
time. That path needs a credential, named by `token_secret` (a secret NAME, never a value —
gbserver merges the space's secrets into the task environment). It is **empty by default** and the
repo is private, so the step refuses with a message naming the missing secret rather than hanging
on a credential prompt. A read-only **deploy key** via gitstore's `GIT_SSH_KEY` convention would be
the tightest option if this ever has to be automated.

## Config

| Key | Default | Notes |
|---|---|---|
| `align_config.teacher_model` | `""` | The 4.2 30B teacher **directory**, not an HF repo id. Its tokenizer is what the student is retagged onto. |
| `align_config.student_model` | `""` | The **raw, pre-retag** 4.1 3B base student — the one directory that genuinely mis-segments. |
| `align_config.out_dir` | `align` | Relative values resolve against `$GB_BUILD_WORKDIR`. The three outputs are derived from it and are not separately configurable. |
| `align_config.chat_template` | `templates/chatml_granite_42_generation.jinja` | **Relative resolves against the checkout.** Empty ⇒ `--no-chat-template` plus a loud warning: the student then has no template and GOLD cannot build assistant masks. A template *without* `{% generation %}` markers is worse than none — it makes the masks all-zero, which raises inside the trainer instead of here. |
| `align_config.copy_mode` | `copy` | `copy` \| `symlink` \| `hardlink`. Only `copy` is safe across filesystems. Never symlink an overlay a later step may write to. |
| `align_config.verify` | `true` | Keep it true: `verify()` is what catches a resolved backend disagreeing with `tokenizer.json`, and it is four cheap encoding probes. |
| `align_config.dry_run` | `false` | Resolve and report what would be written, without writing it. Skips the resume marker entirely, in both directions. |

## Wiring it

```yaml
targets:
  align:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    outputs:
      retagged_student:
        uri: "env://{{ binding.path }}"
        type: model
      teacher_overlay:
        uri: "env://{{ binding.path }}"
        type: model
      student_overlay:
        uri: "env://{{ binding.path }}"
        type: model
    steps:
      - step_uri: space://steps/distill-tokenizer-align
        config:
          align_config:
            teacher_model: "$${TEACHER_MODEL}"
            student_model: "$${STUDENT_MODEL}"
            out_dir: "$${RUN_NAME}/align"
          launcher_config:
            resources:
              cluster: "bluevela"
              zone: "normal"
```

A consumer then binds one of them:

```yaml
  corpus:
    inputs:
      tokenizer:
        binding: align.retagged_student
    steps:
      - step_uri: space://steps/distill-corpus-prep
        config:
          corpus_config:
            tokenizer: "{{ bindings.tokenizer.binding.path }}"
```

## Resume

A recipe on a preemptable queue is **restarted, not resumed**, so every step must answer "has
this already been done, and with what?" for itself. `align_state` is this step's answer:

| Exit | Meaning |
|---|---|
| `0` | nothing recorded; do the work |
| `64` | recorded under this **exact** expectation; publish the paths and exit |
| `65` | recorded under a **different** expectation, or a declared output is gone — refuse |

`65` refuses rather than rebuilding because this step's output is what four later steps compare
their tokenizers against, so quietly replacing it with something built from different inputs is
the expensive mistake. The completion marker is written **last**, after the ChatML post-condition,
so a marker can never describe an unverified student.

## Tests

```bash
make test                                                   # 33 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 39 ported upstream tests
```

The contract tests read the template and `src/run-align.sh` directly and assert, among other
things, that every declared artifact id is actually printed (and vice versa), that booleans are
rendered as `--flag`/`--no-flag` pairs rather than `--flag {{ value }}`, and that every flag the
template passes is one the script parses. A green run with no checkout means the **step contract**
holds — not that the upstream code does.
