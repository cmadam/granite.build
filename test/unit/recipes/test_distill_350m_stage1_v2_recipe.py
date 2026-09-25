#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the distill-stage1-v2 recipe.

v2 is stage 1 retried after build df8512e0 produced a model worse on every one of the
30+ benchmarks measured. The run was mechanically perfect -- every step exit 0, no NaN,
no OOM -- and the objective is what failed: GOLD's loss IS the divergence, the student
was already SFT'd on this corpus family and therefore nearly satisfied it at
initialisation, and with no ground-truth term the only descent direction left for 7,640
steps was to become more certain. Entropy fell 42%, the generations became repetition
loops, and the train loss moved 2.7% and said nothing.

So what this file guards is different in kind from distill-stage1's tests. Those assert
the PLUMBING still works. These assert the four things that make v2 a different
experiment rather than a re-run, each of which is one edit away from silently reverting:

* the objective is anchored, and the anchor is REACHABLE (a patched trainer, pinned);
* the collapse is instrumented and the guard is armed;
* the horizon is bounded by steps, sized to the headroom that actually existed;
* the whole checkpoint curve survives, and every rung on the ladder is a checkpoint
  that will exist.

The last one is the subtlest. The ladder drives a Jinja loop, so CKPT_LADDER IS the
target graph: a rung that is not a multiple of GOLD_SAVE_STEPS names a checkpoint-N
directory the trainer never wrote, and the export target fails AFTER the training has
been paid for.
"""

import json
import pathlib
import re
import subprocess
import textwrap

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "distill-stage1-v2"
)
_STAGE1 = _RECIPE.parent / "distill-stage1"

_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_.,\"\[\]()\- ]+)\}")
_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")
_HEREDOC = re.compile(r"<<'PYSRC'\n(.*?)\n\s*PYSRC(?:\n|$)", re.S)

# The loop variable is bound by the <% for %> block, not by parameters.yaml.
_LOOP_VARS = {"N"}

_LADDER = ["25", "50", "75", "100", "150", "200", "300"]
_TARGETS = (
    ["sources", "align", "corpus", "train-gold"]
    + [f"export-{n}" for n in _LADDER]
    + ["eval-transfer-baseline"]
    + [f"eval-transfer-{n}" for n in _LADDER]
    + ["gen-smoke", "eval-bfcl-baseline"]
    + [f"eval-bfcl-{n}" for n in _LADDER]
)


def _params(**overrides):
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    return _targets(rendered)[target]["steps"][0]["config"]


@pytest.fixture(name="off")
def fixture_off(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False)


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    names = set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS
    assert names - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    referenced = set(_PLAIN_MARKER.findall(contents))
    # CKPT_LADDER (.split(...)) and CORPUS_DIR (.rstrip('/'), so a copied path with a
    # trailing slash cannot reach an artifact URI) are reached through an expression,
    # which the plain marker pattern does not match, so allow them explicitly.
    referenced |= {"CKPT_LADDER", "CORPUS_DIR"}
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_no_comment_uses_a_loop_variable_outside_its_loop():
    """The CLI templates this file IN FULL, comments included, with StrictUndefined.
    A loop variable named in a comment is an undefined-variable error that fails the
    build before submission -- which is exactly how this file first failed to render."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    for line in contents.splitlines():
        if line.lstrip().startswith("#"):
            assert not (set(_PLAIN_MARKER.findall(line)) & _LOOP_VARS), line


def test_the_recipe_renders_the_v2_graph(off):
    assert list(_targets(off)) == _TARGETS


def test_results_do_not_land_under_gbtest(off):
    assert "gbtest" not in _params()["WORKDIR_ROOT"]


def test_the_build_is_not_still_named_after_the_smoke_recipe(off):
    """distill-stage1 shipped as `name: distill-350m-smoke`, so three different
    builds share one name in `gb build list` and stage-1 runs are identifiable only
    by their artifact URI. v2 does not inherit that."""
    assert off["granite.build"]["name"] == "distill-350m-stage1-v2"
    assert "smoke" not in off["granite.build"]["name"]


# ─── The objective ─────────────────────────────────────────────────────────────


class TestTheAnchoredObjective:
    def test_the_ce_anchor_is_on(self, off):
        """The single change the post-mortem ranks first. Without it the loss has no
        ground-truth term, so on an already-SFT'd student 'become more confident' is
        a free way to score better -- which is what 7,640 steps of df8512e0 did."""
        assert float(_config(off, "train-gold")["gold_config"]["ce_coef"]) > 0

    def test_the_anchor_is_the_same_order_as_the_divergence(self, off):
        """Measured, not guessed: df8512e0's JSD ran 0.068 and CE on this student runs
        order 1. Below ~0.01 the anchor cannot restrain anything; above ~0.5 it is SFT
        with a divergence garnish, and either way the run answers a question nobody
        asked."""
        assert 0.01 <= float(_config(off, "train-gold")["gold_config"]["ce_coef"]) <= 0.5

    def test_entropy_is_logged_every_step(self, off):
        gold = _config(off, "train-gold")["gold_config"]
        assert gold["log_student_entropy"] is True
        assert int(gold["logging_steps"]) == 1

    def test_the_collapse_guard_is_armed(self, off):
        """15% because df8512e0 lost 42% monotonically, so a threshold in that range
        fires during the descent rather than confirming the arrival."""
        gold = _config(off, "train-gold")["gold_config"]
        assert 0 < float(gold["entropy_guard_drop_frac"]) < 0.42
        assert int(gold["entropy_guard_baseline_steps"]) >= 1
        assert int(gold["entropy_guard_patience"]) >= 1

    def test_a_stopping_guard_and_a_fixed_ladder_cannot_both_be_asked_for(self, off):
        """THE failure of build d1acf1c0, as an invariant.

        The guard tripped at step 77 of 2,000 and stopped the run, exactly as armed. But
        every rung of this recipe's ladder is a literal step number, and the trainer had
        written none of them: all four export targets failed with `requested checkpoint
        does not exist`, the build failed, and the run's question went unanswered after
        the training was paid for. The two features are individually right and jointly
        contradictory -- a guard that may stop at an arbitrary step, and consumers that
        demand specific steps -- so one of them has to give, and for a run whose PURPOSE
        is the collapse curve it is the stop.

        Either the guard only warns, or no rung is allowed to outlive a possible stop.
        This recipe takes the first branch; the assertion admits both so that a future
        recipe choosing the second is not forced to lie."""
        gold = _config(off, "train-gold")["gold_config"]
        if float(gold["entropy_guard_drop_frac"]) <= 0:
            return  # guard disarmed: the ladder is bounded by max_steps alone
        rungs = [int(n) for n in _params()["CKPT_LADDER"].split(",")]
        if gold["entropy_guard_action"] == "stop":
            raise AssertionError(
                "the guard may stop at any step past "
                f"{gold['entropy_guard_baseline_steps']}, but the ladder demands "
                f"checkpoints at {rungs}. Set ENTROPY_GUARD_ACTION=warn, or stop naming "
                "fixed rungs."
            )
        assert gold["entropy_guard_action"] == "warn"

    def test_the_guard_still_reports_and_checkpoints_when_it_only_warns(self, off):
        """warn must not become off. The trip step is the single most interesting
        checkpoint in the run, and the reading is the whole reason the guard is armed
        in an arm that is expected to collapse."""
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold["entropy_guard_drop_frac"]) > 0
        assert gold["log_student_entropy"] is True
        # A trip adds one checkpoint beyond the scheduled ones, so the no-eviction
        # budget has to cover it.
        produced = int(gold["max_steps"]) // int(gold["save_steps"]) + 1
        assert int(gold["save_total_limit"]) >= produced

    def test_the_guard_cannot_be_armed_without_its_metric(self, off):
        """Belt and braces: the renderer refuses this combination too, but a recipe
        that sets it has already wasted a submission."""
        gold = _config(off, "train-gold")["gold_config"]
        if float(gold["entropy_guard_drop_frac"]) > 0:
            assert gold["log_student_entropy"] is True

    def test_the_control_arm_renders_df8512e0s_exact_objective(self, tmp_path):
        """The control is `--param CE_COEF=0 --param ENTROPY_GUARD_DROP_FRAC=0`, and
        it is only a control if that reproduces the failed run's loss rather than
        approximating it. The renderer emits the new keys only when non-default, so
        this asserts the recipe does not defeat that by setting them some other way."""
        control = _render(tmp_path, INCLUDE_SFT=False, CE_COEF=0, ENTROPY_GUARD_DROP_FRAC=0)
        gold = _config(control, "train-gold")["gold_config"]
        assert float(gold["ce_coef"]) == 0.0
        assert float(gold["entropy_guard_drop_frac"]) == 0.0

    def test_beta_and_the_policy_are_unchanged_from_stage1(self, off):
        """One variable at a time. The anchor already changes the loss; moving beta or
        lmbda as well would make a v2-vs-df8512e0 comparison unattributable."""
        stage1 = get_params_from_file(str(_STAGE1 / "parameters.yaml"))
        mine = _params()
        for key in ("BETA", "LMBDA", "TEMPERATURE", "USE_LIGER_FUSED_JSD"):
            assert mine[key] == stage1[key], key
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold["lmbda"]) == 0.0
        assert int(gold["vllm_num_servers"]) == 0

    def test_the_corpus_and_geometry_are_unchanged_from_stage1(self):
        """Same reason. If these drift, v2 measures a different experiment and the
        recorded baseline row stops being the right comparison."""
        stage1 = get_params_from_file(str(_STAGE1 / "parameters.yaml"))
        mine = _params()
        for key in (
            "TARGET_ROWS",
            "SHUFFLE_SEED",
            "SEED",
            "MAX_LENGTH",
            "EVAL_FRACTION",
            "STUDENT_MODEL",
            "TEACHER_MODEL",
            "GOLD_LEARNING_RATE",
            "GOLD_MIN_LR",
            "WARMUP_RATIO",
            "LR_SCHEDULER_TYPE",
            "GOLD_PER_DEVICE_TRAIN_BATCH_SIZE",
            "GOLD_GRADIENT_ACCUMULATION_STEPS",
            "GOLD_NUM_NODES",
            "GOLD_NUM_GPUS",
        ):
            assert mine[key] == stage1[key], key


# ─── The trainer pin ───────────────────────────────────────────────────────────


class TestTheTrainerPin:
    """train-gold is the one target that does not read the pinned steps checkout, and
    until v2 nothing pinned what it did read.

    df8512e0 recorded kd_sandbox_commit fc7d66e from a tree carrying 159 uncommitted
    files including gold/, so its single record of what trained described a commit
    whose code did not run. An unanchored objective was the scientific failure; an
    unrecorded trainer would have made even the diagnosis unreproducible.
    """

    def test_the_trainer_is_a_checkout_this_project_controls(self):
        code_dir = _params()["KD_CODE_DIR"]
        assert code_dir.endswith("-gb"), code_dir
        assert code_dir != "/proj/granite-build/g4os/kd-sandbox"

    def test_the_pin_is_a_full_commit_not_a_branch(self):
        """A branch head moves, which is the whole failure being fixed."""
        ref = _params()["KD_EXPECT_REF"]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), ref

    def test_the_pin_reaches_the_step(self, off):
        assert _config(off, "train-gold")["gold_config"]["kd_expect_ref"] == _params()[
            "KD_EXPECT_REF"
        ]

    def test_the_patch_that_builds_that_checkout_ships_with_the_repo(self):
        """Otherwise the pin names a tree nobody can rebuild, which is a different
        way of having no provenance at all."""
        patch = (
            _RECIPE.parents[3]
            / "steps"
            / "gold-distill"
            / "skypilot"
            / "patches"
            / "ce_anchor_and_entropy_guard.diff"
        )
        assert patch.is_file()
        text = patch.read_text(encoding="utf-8")
        assert "Base commit:" in text
        assert _params()["KD_EXPECT_REF"] in text, "the patch must name the pinned result"
        for key in ("ce_coef", "log_student_entropy", "entropy_guard_drop_frac"):
            assert key in text, f"{key} is pinned but not in the patch"

    def test_no_target_overrides_the_shared_step_code_config(self, off):
        """The six ported steps' pin belongs to the step, once, for all six. A
        recipe-level override is what left align reading a patched tree while corpus
        read the shared one -- build 1820703f."""
        for name, target in _targets(off).items():
            for step in target["steps"]:
                assert "code_config" not in step.get("config", {}), name


# ─── The horizon and the ladder ────────────────────────────────────────────────


class TestTheHorizonAndTheLadder:
    def test_the_run_is_bounded_by_steps_not_by_an_epoch(self, off):
        """The inverse of distill-stage1's assertion, and deliberately so. An epoch
        was 8,150 steps against ~12% of available headroom, two thirds of which was
        closed by step 509."""
        gold = _config(off, "train-gold")["gold_config"]
        assert int(gold["max_steps"]) > 0

    def test_the_horizon_covers_the_descent_and_not_much_more(self, off):
        """Bounded by the control's OWN measured curve, not by df8512e0's step-509 knee.

        Build bb779f1f ran the unanchored objective 2,000 steps with the guard off:
        entropy -19% by step 80, -29.7% by 120, -34.1% by 250, and -35.5% at 2,000. The
        last 1,750 steps bought 1.4%. JSD and forward KL were flat after 500 and reverse
        KL never moved. So the horizon has to clear the shoulder (~250) and has no reason
        to run far past it -- and a short horizon is what makes a CE_COEF sweep
        affordable, which is the experiment this recipe is now for.

        The upper bound is the load-bearing half. Nothing stops someone restoring 2,000
        'to be safe', and that silently triples the cost of every arm."""
        max_steps = int(_config(off, "train-gold")["gold_config"]["max_steps"])
        assert 250 <= max_steps <= 600, max_steps

    def test_every_rung_is_a_checkpoint_that_will_exist(self, off):
        """The failure this prevents: a rung that is not a multiple of save_steps names
        a checkpoint-N the trainer never wrote, and the export fails after the
        training has been paid for."""
        gold = _config(off, "train-gold")["gold_config"]
        save = int(gold["save_steps"])
        for rung in _params()["CKPT_LADDER"].split(","):
            assert int(rung) % save == 0, f"checkpoint-{rung} is not a multiple of {save}"
            assert int(rung) <= int(gold["max_steps"])

    def test_the_last_rung_is_the_end_of_the_run(self, off):
        rungs = [int(n) for n in _params()["CKPT_LADDER"].split(",")]
        assert rungs == sorted(rungs)
        assert rungs[-1] == int(_config(off, "train-gold")["gold_config"]["max_steps"])

    def test_no_checkpoint_is_evicted(self, off):
        """save_total_limit 3 at save_steps 250 is what made df8512e0 unsalvageable:
        it kept the last 750 steps of an 8,150-step run and deleted every checkpoint
        from before the collapse. For an exploratory run the valuable ones are EARLY."""
        gold = _config(off, "train-gold")["gold_config"]
        produced = int(gold["max_steps"]) // int(gold["save_steps"])
        assert int(gold["save_total_limit"]) >= produced

    def test_each_rung_exports_the_checkpoint_it_names(self, off):
        for rung in _params()["CKPT_LADDER"].split(","):
            export = _config(off, f"export-{rung}")["export_config"]
            assert export["checkpoint"] == f"checkpoint-{rung}"
            assert export["dest"].endswith(f"/export-{rung}")

    def test_no_export_is_left_to_pick_the_highest_step(self, off):
        """Empty means 'highest step number', which for a guard-stopped run is
        whatever step the guard fired on -- a different checkpoint in each arm."""
        for rung in _params()["CKPT_LADDER"].split(","):
            assert _config(off, f"export-{rung}")["export_config"]["checkpoint"] != ""

    def test_each_rung_is_transfer_evaluated_against_its_own_export(self, off):
        for rung in _params()["CKPT_LADDER"].split(","):
            target = _targets(off)[f"eval-transfer-{rung}"]
            assert target["inputs"]["student"]["binding"] == f"export-{rung}.hf_model"
            cfg = _config(off, f"eval-transfer-{rung}")["eval_config"]
            assert cfg["output_dir"].endswith(f"/eval-transfer-{rung}")

    def test_entropy_is_among_the_transfer_metrics(self, off):
        """The metric that caught this after the fact, and the only one of the four
        that can see a collapsed student whose divergence happens to look fine."""
        for rung in list(_params()["CKPT_LADDER"].split(",")) + ["baseline"]:
            metrics = _config(off, f"eval-transfer-{rung}")["eval_config"]["metrics"]
            assert "entropy" in metrics
            assert "rkld" in metrics

    def test_the_baseline_read_is_still_the_aligned_student(self, off):
        """A divergence with no baseline is a number without a direction."""
        target = _targets(off)["eval-transfer-baseline"]
        assert target["inputs"]["student"]["binding"] == "align.retagged_student"

    def test_every_rung_shares_the_length_budget(self, off):
        budget = _params()["MAX_LENGTH"]
        assert budget == 8192
        assert _config(off, "corpus")["corpus_config"]["max_length"] == budget
        assert _config(off, "train-gold")["gold_config"]["max_length"] == budget
        for rung in list(_params()["CKPT_LADDER"].split(",")) + ["baseline"]:
            assert _config(off, f"eval-transfer-{rung}")["eval_config"]["max_length"] == budget


# ─── The generation smoke test ─────────────────────────────────────────────────


class TestGenSmoke:
    @pytest.fixture(name="cmd")
    def fixture_cmd(self, off):
        return _config(off, "gen-smoke")["command_config"]["command"]

    def test_the_command_survived_yaml(self, cmd):
        assert cmd.count("\n") > 20

    def test_the_shell_parses(self, cmd):
        result = subprocess.run(
            ["bash", "-n"], input=cmd, text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr

    def test_the_interpreter_actually_receives_every_rung(self, tmp_path):
        """Executes the assembly instead of parsing it. `bash -n` cannot see this
        failure and neither can compile(): the <% %> block tags are not trimmed, so
        each leaves a blank line where it stood, a blank line after a `\\` ENDS the
        command, and bash then runs the next rung as a program name -- exit 127 with
        `500:/proj/.../export-500: No such file or directory`, which is build
        bb779f1f's gen-smoke. The interpreter is reached with no rungs and no heredoc
        while every static check stays green, so the only test that can catch it is
        one that looks at what the interpreter was handed."""
        rendered = _render(tmp_path, INCLUDE_SFT=False, WORKDIR_ROOT=str(tmp_path))
        cmd = _config(rendered, "gen-smoke")["command_config"]["command"]

        argv_log = tmp_path / "argv.json"
        stdin_log = tmp_path / "stdin.txt"
        stub = tmp_path / "python-stub"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f"open({str(stdin_log)!r}, 'w').write(sys.stdin.read())\n"
            f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

        # Only the interpreter is swapped. Every argument, the array, the loop and the
        # heredoc are the recipe's own rendered text.
        script = cmd.replace("/stage/.venv/bin/python", str(stub))
        result = subprocess.run(["bash"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr

        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        rungs = _params()["CKPT_LADDER"].split(",")
        # The stub sees the `-` that real python consumes as "read the program from
        # stdin"; the recipe's own arguments start after it.
        assert argv[0] == "-", f"interpreter got {argv}"
        argv = argv[1:]
        assert len(argv) == 3 + len(rungs), f"interpreter got {argv}"
        assert argv[0].endswith("/repetition.json")
        assert argv[1] == str(_params()["GEN_SMOKE_MAX_REPETITION"])
        assert argv[2] == str(_params()["GEN_SMOKE_NEW_TOKENS"])
        for rung, got in zip(rungs, argv[3:]):
            step, _, path = got.partition(":")
            assert step == rung, f"expected rung {rung}, got {got}"
            assert path.endswith(f"/export-{rung}"), got

        # The heredoc has to land on the interpreter's stdin, not be orphaned by a
        # broken continuation.
        assert "def looped_fraction" in stdin_log.read_text(encoding="utf-8")

    def test_no_rung_is_appended_through_a_line_continuation(self):
        """The shape that broke, guarded at the source. A `\\`-continued line
        followed by a <% %> tag renders to a continuation followed by a blank line,
        which terminates the command."""
        lines = (_RECIPE / "build.yaml").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith("\\") and lines[i + 1].lstrip().startswith("<%"):
                raise AssertionError(
                    f"{_RECIPE.name}/build.yaml:{i + 1}: continuation followed by a "
                    f"block tag renders to a blank line and ends the command:\n"
                    f"  {line}\n  {lines[i + 1]}"
                )

    def test_the_embedded_python_compiles(self, cmd):
        bodies = _HEREDOC.findall(cmd)
        assert bodies, "no Python heredoc found; has the target changed shape?"
        for i, body in enumerate(bodies):
            compile(textwrap.dedent(body), f"gen-smoke-{i}", "exec")

    def test_it_reads_every_rung(self, cmd, off):
        for rung in _params()["CKPT_LADDER"].split(","):
            assert f"/export-{rung}" in cmd
            assert f"export_{rung}" in _targets(off)["gen-smoke"]["inputs"]

    def test_decoding_is_greedy(self, cmd):
        """Sampling would hide the failure. Temperature is what lets a narrowed
        distribution still look varied, and df8512e0's MultiPL-E ran at temperature
        0.2 on a model with an effective branching factor of 1.49 -- effectively
        greedy already, which is why it degenerated there and not elsewhere."""
        assert "do_sample=False" in cmd

    def test_the_prompts_are_raw_completions(self, cmd):
        """MultiPL-E is where the collapse showed worst and it applies no chat
        template, so a templated prompt set would test a different thing from the
        benchmark that caught this."""
        body = textwrap.dedent(_HEREDOC.findall(cmd)[0])
        prompts = body[body.index("PROMPTS = ["):body.index("]", body.index("PROMPTS = ["))]
        assert "<|start_of_role|>" not in prompts
        assert "<|im_start|>" not in prompts

    def test_only_the_final_rung_can_fail_the_target(self, cmd):
        """An early rung above threshold is a finding to read in the table. Failing
        the build on it would discard later checkpoints that may be fine."""
        assert 'if rows[-1]["degenerate"]:' in cmd
        assert cmd.count("sys.exit(1)") == 1

    def test_the_gate_is_the_run_metric_not_the_adjacent_pair_rate(self, cmd):
        """Measured: a CORRECT 7-line Java function scores 0.167 on the adjacent-pair
        rate, because two dedented closing braces are identical and one collision out
        of six pairs is 17%. Gating on that fails healthy checkpoints. The run-based
        metric gives 1.00 for the collapsed sample and 0.00 for the correct one."""
        assert 'degenerate": mean_looped > threshold' in cmd
        assert "def looped_fraction" in cmd
        assert "def adjacent_rate" in cmd, "keep reporting it; the post-mortem quotes it"

    def test_the_detector_separates_the_post_mortems_two_samples(self, cmd):
        """Executes the shipped detector rather than trusting its docstring. These are
        the actual generations recorded in the post-mortem."""
        body = textwrap.dedent(_HEREDOC.findall(cmd)[0])
        # Just the three pure functions: everything above them reads sys.argv.
        detector = body[body.index("def adjacent_rate") : body.index("import torch")]
        namespace = {}
        exec(compile(detector, "detector", "exec"), namespace)
        measure = namespace["measure"]

        collapsed = "// (true)\n" * 50
        correct = (
            "int count = 0;\n"
            "for (int i = 0; i < numbers.size(); i++) {\n"
            "    for (int j = i + 1; j < numbers.size(); j++) {\n"
            "        if (Math.abs(a - b) < threshold) { count++; }\n"
            "    }\n"
            "}\n"
            "return count >= 2;"
        )
        threshold = float(_params()["GEN_SMOKE_MAX_REPETITION"])
        assert measure(collapsed)[0] > threshold
        assert measure(correct)[0] <= threshold
        for degenerate_input in ("", "one line", "\n\n  \n"):
            assert measure(degenerate_input)[0] == 0.0

    def test_capability_is_measured_at_every_rung_and_at_the_baseline(self, off):
        """bb779f1f measured capability ONCE, at step 2,000, and got 0.000 (0/400) with
        no baseline and no earlier reading -- so it could not say whether the capability
        went at step 25 or step 1,999, nor whether the starting student had it. Every
        divergence metric in that build improved while this one sat at zero, which is
        exactly what eval-transfer cannot see. A ladder of divergence readings beside a
        single capability reading measures the wrong thing carefully."""
        targets = _targets(off)
        assert (
            targets["eval-bfcl-baseline"]["inputs"]["model"]["binding"]
            == "align.retagged_student"
        )
        for rung in _params()["CKPT_LADDER"].split(","):
            binding = targets[f"eval-bfcl-{rung}"]["inputs"]["model"]["binding"]
            assert binding == f"export-{rung}.hf_model"

    def test_no_two_bfcl_targets_share_an_output_directory(self, off):
        """Each writes under output_dir/experiment/eval_name, so a shared pair would
        have two targets registering one artifact URI -- the b5f030cd mode, where the
        second reports SUCCESS with an empty output list and consumers wait forever."""
        seen = {}
        names = ["eval-bfcl-baseline"] + [
            f"eval-bfcl-{n}" for n in _params()["CKPT_LADDER"].split(",")
        ]
        for name in names:
            cfg = _config(off, name)["bfcl_config"]
            key = (cfg["output_dir"], cfg["experiment"])
            assert key not in seen, f"{name} collides with {seen[key]}: {key}"
            seen[key] = name

    def test_the_descent_is_where_the_rungs_are(self, off):
        """The old ladder was 500/1000/1500/2000 -- every rung on the plateau, which is
        how two builds produced no reading anywhere in the region that moves. The
        control loses 19% of its entropy by step 80; at least half the rungs belong at
        or below that shoulder."""
        rungs = [int(n) for n in _params()["CKPT_LADDER"].split(",")]
        early = [n for n in rungs if n <= 120]
        assert len(early) >= len(rungs) / 2, f"only {early} of {rungs} are in the descent"


# ─── Diagnosability ────────────────────────────────────────────────────────────


def test_nccl_debug_is_on_by_default(off):
    """Build 8f02b739 hung at step 219 in the ZeRO-3 parameter all-gather across two
    racks and left no INIT,NET trace, because NCCL_DEBUG was empty. It had to be
    cancelled and relaunched with the flag, paying 45-60 minutes of redone sources +
    align + corpus. A 2-node run is exactly the shape that hang needs."""
    gold = _config(off, "train-gold")["gold_config"]
    assert gold["nccl_debug"] == "INFO"
    assert gold["nccl_debug_subsys"], "a trace with no subsystem filter is unbounded"


def test_the_stale_smoke_comments_did_not_come_along(off):
    """distill-stage1's parameters.yaml still describes a 48-row smoke run in its
    header, one node with two GPUs at GOLD_NUM_NODES, and 'no cross-node collectives'
    at NCCL_DEBUG -- all three contradicting its own live values. Copying a recipe
    copies its comments, and a comment that lies is worse than none."""
    text = (_RECIPE / "parameters.yaml").read_text(encoding="utf-8")
    assert "48 rows" not in text
    assert "Nothing it produces is a publishable measurement" not in text
    assert "no cross-node collectives" not in text
    assert "One node, two GPUs for GOLD" not in text


# ─── Reusing an existing corpus ────────────────────────────────────────────────


_PIN = "/proj/granite-build/g4os/distill/distill-350m-s1v2-ce040/c20ed3c0/corpus"


def _corpus_inputs(rendered):
    """Every input named `corpus`, by target. These are the four consumers whose
    wiring has to change together: miss one and it either waits on a target that
    was not rendered, or reads a corpus nobody checked."""
    out = {}
    for name, target in _targets(rendered).items():
        spec = (target.get("inputs") or {}).get("corpus")
        if spec is not None:
            out[name] = spec
    return out


@pytest.fixture(name="pinned")
def fixture_pinned(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False, CORPUS_DIR=_PIN)


class TestTheCorpusPin:
    """`sources` + `corpus` are ~50 minutes on the critical path before train-gold can
    start, and they are deterministic: the three arms of the CE_COEF sweep that
    completed (9dab9130, 8c8ebd63, c20ed3c0) produced byte-identical
    sources/train.jsonl, corpus/train.jsonl and corpus/eval.jsonl. So a sweep pays for
    the same file once per arm.

    Pinning has to be done as a DIRECT input, not by fixing BUILD_SUBDIR. gbserver's
    target reuse is scoped to one build id (docs/builds/target-reuse.md), and a shared
    output URI is the b5f030cd mode: the second build's registration is refused, the
    target still reports SUCCESS with an empty output list, and every consumer waits
    forever on a binding that will never resolve."""

    def test_the_corpus_is_built_in_the_build_by_default(self, off):
        """The default must not change. Every run before this parameter existed built
        its own corpus, and an unpinned build still has to."""
        assert "sources" in _targets(off)
        assert "corpus" in _targets(off)
        assert _corpus_inputs(off)
        for name, spec in _corpus_inputs(off).items():
            assert spec.get("binding") == "corpus.corpus", name
            assert "uri" not in spec, name

    def test_a_pin_removes_the_targets_that_would_rebuild_it(self, pinned):
        assert "sources" not in _targets(pinned)
        assert "corpus" not in _targets(pinned)

    def test_a_pin_is_read_directly_and_not_through_a_binding(self, pinned):
        """The load-bearing assertion. A `binding` names another target's output in
        THIS build; with `corpus` not rendered, any surviving binding is a consumer
        blocked on a producer that does not exist."""
        consumers = _corpus_inputs(pinned)
        assert consumers, "no target reads the corpus at all"
        for name, spec in consumers.items():
            assert "binding" not in spec, name
            assert spec["uri"] == f"env://{_PIN}/train.jsonl", name

    def test_a_pin_is_never_re_registered_as_an_output(self, pinned):
        """b5f030cd. Reading a URI another build produced is ordinary; declaring it as
        your own output is what gbserver refuses."""
        for name, target in _targets(pinned).items():
            for out_name, spec in (target.get("outputs") or {}).items():
                assert _PIN not in str(spec.get("uri", "")), f"{name}.{out_name}"

    def test_the_transfer_evals_read_the_pinned_eval_split(self, pinned):
        """These two compose the eval.jsonl path by hand rather than through the
        binding, so they are the sites a parameter switch is most likely to miss."""
        evals = [n for n in _targets(pinned) if n.startswith("eval-transfer-")]
        assert evals
        for name in evals:
            cfg = _config(pinned, name)["eval_config"]
            assert cfg["corpus"] == f"{_PIN}/eval.jsonl", name

    def test_an_unpinned_build_still_reads_its_own_eval_split(self, off):
        for name in [n for n in _targets(off) if n.startswith("eval-transfer-")]:
            cfg = _config(off, name)["eval_config"]
            assert cfg["corpus"].endswith("/corpus/eval.jsonl"), name
            assert _PIN not in cfg["corpus"], name

    def test_a_pin_is_checked_before_any_allocation_is_held(self, pinned, off):
        """The corpus is DEFINED by the retagged tokenizer and by the prep policies.
        A pin taken from a run with a different teacher or a different max_length
        trains on a corpus this build does not describe, and nothing downstream would
        say so -- which is the one new silent failure mode pinning introduces."""
        assert "corpus-pin-check" in _targets(pinned)
        assert "corpus-pin-check" not in _targets(off)
        gate = _targets(pinned)["corpus-pin-check"]
        assert (gate["inputs"]["tokenizer"]["binding"]) == "align.retagged_student"
        res = _config(pinned, "corpus-pin-check")["launcher_config"]["resources"]
        assert "accelerators" not in res, "the gate must not hold a GPU"
        for name in _corpus_inputs(pinned):
            bindings = {
                s.get("binding") for s in _targets(pinned)[name]["inputs"].values()
            }
            assert "corpus-pin-check.pin_check" in bindings, name


    @pytest.mark.parametrize("include_sft", [False, True])
    @pytest.mark.parametrize("pin", ["", _PIN])
    def test_both_switches_render_together(self, tmp_path, include_sft, pin):
        """Two independent conditionals now gate the same graph, and INCLUDE_SFT's
        train-sft is itself a corpus consumer. The combinations are cheap to render
        and the failure mode -- one arm of one switch leaving unbalanced YAML -- is
        only visible when both are exercised."""
        rendered = _render(tmp_path, INCLUDE_SFT=include_sft, CORPUS_DIR=pin)
        targets = _targets(rendered)
        assert ("train-sft" in targets) is include_sft
        assert ("sources" in targets) is not bool(pin)
        assert ("corpus-pin-check" in targets) is bool(pin)
        for spec in _corpus_inputs(rendered).values():
            assert bool(spec.get("uri")) is bool(pin)

    def test_a_trailing_slash_does_not_reach_the_uri(self, tmp_path):
        """`ls -d .../*/` prints a trailing slash, which is how the path gets copied in
        practice. Unnormalised it renders `env://<dir>//train.jsonl` -- which resolves on
        POSIX but records an artifact URI that does not match the canonical one."""
        contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
        plain, slashed = (
            apply_parameters(contents, [], _params(INCLUDE_SFT=False, CORPUS_DIR=d), str(tmp_path))
            for d in (_PIN, _PIN + "/")
        )
        assert plain == slashed


class TestThePinCheckScript:
    @pytest.fixture(name="cmd")
    def fixture_cmd(self, pinned):
        return _config(pinned, "corpus-pin-check")["command_config"]["command"]

    @pytest.fixture(name="src")
    def fixture_src(self, cmd):
        found = _HEREDOC.search(cmd)
        assert found, "no PYSRC heredoc survived"
        return textwrap.dedent(found.group(1))

    def test_the_shell_parses(self, cmd):
        proc = subprocess.run(
            ["bash", "-n"], input=cmd, text=True, capture_output=True, check=False
        )
        assert proc.returncode == 0, proc.stderr

    def test_the_embedded_python_compiles(self, src):
        compile(src, "corpus-pin-check", "exec")

    def test_it_checks_everything_that_defines_the_corpus(self, src):
        for key in (
            "tokenizer_identity",
            "max_length",
            "think_policy",
            "completion_boundary",
            "documents_policy",
            "eval_fraction",
        ):
            assert key in src, key

    def _run(self, src, tmp_path, manifest, **kw):
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        (corpus / "corpus_manifest.json").write_text(json.dumps(manifest))
        (corpus / "train.jsonl").write_text("{}\n")
        (corpus / "eval.jsonl").write_text("{}\n")
        tok = tmp_path / "retagged_student"
        tok.mkdir(exist_ok=True)
        script = tmp_path / "pin_check.py"
        script.write_text(src)
        argv = [
            str(corpus),
            kw.get("teacher", "/models/granite-4.1-3b-pinned"),
            str(kw.get("max_length", 8192)),
            kw.get("think_policy", "keep"),
            kw.get("documents_policy", "keep"),
            str(kw.get("eval_fraction", 0.005)),
            str(tok),
            str(tmp_path / "out.json"),
        ]
        return subprocess.run(
            ["python3", str(script), *argv],
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _manifest(**over):
        m = {
            "tokenizer_identity": "granite-4.1-3b-pinned",
            "tokenizer_path": "/gone/align/retagged_student",
            "seed": 42,
            "eval_fraction": 0.005,
            "policies": {
                "max_length": 8192,
                "think_policy": "keep",
                "completion_boundary": "last_message",
                "documents_policy": "keep",
            },
        }
        m.update(over)
        return m

    def test_a_matching_manifest_is_accepted(self, src, tmp_path):
        proc = self._run(src, tmp_path, self._manifest())
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (tmp_path / "out.json").is_file()

    def test_a_different_max_length_is_rejected(self, src, tmp_path):
        m = self._manifest()
        m["policies"]["max_length"] = 4096
        proc = self._run(src, tmp_path, m)
        assert proc.returncode != 0
        assert "max_length" in proc.stdout + proc.stderr

    def test_a_corpus_prepared_for_a_different_teacher_is_rejected(self, src, tmp_path):
        proc = self._run(
            src, tmp_path, self._manifest(tokenizer_identity="granite-4.0-1b-something")
        )
        assert proc.returncode != 0
        assert "tokenizer_identity" in proc.stdout + proc.stderr

    def test_a_missing_split_is_rejected(self, src, tmp_path):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "corpus_manifest.json").write_text(json.dumps(self._manifest()))
        (corpus / "train.jsonl").write_text("{}\n")
        tok = tmp_path / "retagged_student"
        tok.mkdir()
        script = tmp_path / "pin_check.py"
        script.write_text(src)
        proc = subprocess.run(
            [
                "python3", str(script), str(corpus),
                "/models/granite-4.1-3b-pinned", "8192", "keep", "keep", "0.005",
                str(tok), str(tmp_path / "out.json"),
            ],
            text=True, capture_output=True, check=False,
        )
        assert proc.returncode != 0
        assert "eval.jsonl" in proc.stdout + proc.stderr

    def test_every_mismatch_is_reported_not_just_the_first(self, src, tmp_path):
        m = self._manifest(tokenizer_identity="wrong", eval_fraction=0.5)
        m["policies"]["max_length"] = 4096
        proc = self._run(src, tmp_path, m)
        blob = proc.stdout + proc.stderr
        assert proc.returncode != 0
        for key in ("tokenizer_identity", "eval_fraction", "max_length"):
            assert key in blob, key
