# exp2 — Results

GRPO on all three pretrained models from [exp1](../exp1-pretrain-mixture/), one
episode each (116 rollout steps, 934 optimizer steps), identical configuration.
`tg15` still finishing at time of writing; its cells are marked _pending_.

All evaluation is GSM8K **test** (1,319 problems). RL trains on GSM8K **train**
(7,473), so there is no overlap.

---

## The dose-response is clean

Everything RL does scales with how much TinyGSM the model saw in pretraining.

| | `tg00` (0%) | `tg15` (15%) | `tg30` (30%) |
|---|---|---|---|
| **reward, start → end** | 0.022 → **0.014** | 0.023 → _pending_ | 0.037 → **0.152** |
| **KL, end** | **0.000328** | **0.00835** | **0.0145** |
| response_length, end | **895** | 239 | 216 |
| pass@1, before → after | 0.68% → **0.68%** | 3.87% → _pending_ | 6.44% → **9.40%** |
| `tinygsm-code_count`, before → after | 0.00% → **0.00%** | 89.31% → _pending_ | 97.12% → **98.18%** |

**KL rises monotonically with the pretraining TinyGSM share: 0.0003 → 0.0084 →
0.0145, a 44x spread.** Nothing constrained tg00 to stay close to its reference
— the KL coefficient was identical for all three. Its policy barely moved
because **the gradient signal was near zero and there was nowhere to go**.

That is the cleanest quantitative statement of the paper's thesis available
here: *the amount RL moves the policy is proportional to how much of the target
behaviour pretraining had already installed.*

---

## tg00: RL cannot bootstrap from nothing

A full episode — 116 rollout steps, 2h54m — changed nothing.

| RL step | pass@1 | `tinygsm-code_count` | `text_count` |
|---|---|---|---|
| before | 0.68% | 0.00% | 100.00% |
| 1 | 0.61% | 0.00% | 100.00% |
| 20 | 0.45% | 0.00% | 100.00% |
| 58 | 0.76% | 0.00% | 100.00% |
| **116** | **0.68%** | **0.00%** | **100.00%** |

pass@1 oscillates in 0.45–0.76% with no trend; `code_count` never leaves zero.
Reward over the run:

```
step:      1      9     17     25     33     41     49     57
tg00:   0.022  0.008  0.010  0.012  0.012  0.016  0.018  0.006
tg30:   0.037  0.076  0.049  0.053  0.061  0.070  0.057  0.039
```

tg00 fluctuates around 0.01 with no slope while tg30 climbs to 0.152.

Two reasons, and only the first is about sample efficiency:

1. **Too few informative groups.** GRPO normalises advantage within a group of
   8, so a group whose generations all score alike contributes no gradient. At
   0.68% pass@1, `1 − 0.9932⁸ = 5.3%` of groups carry signal — against 41% for
   tg30. Raising `n_samples_per_prompt` to 32 would lift that to 19.6%.
2. **The reward is not measuring what its name says.** tg00's "correct" answers
   are the answer parser recovering a number from unbounded repetition (see
   exp1 § tg00). Optimising it would reinforce repetition that happens to
   contain the right number — reward hacking, not learning.

More episodes do not fix either. **This is a result, not a failed run:** RL
amplifies what pretraining installed, and when pretraining installed nothing
usable, there is nothing to amplify. The paper has no 0% condition, so this
direction of the claim is untested there.

---

## tg30: both effects, at a very small policy cost

| RL step | pass@1 | `code_count` | `text_count` |
|---|---|---|---|
| **before** | **6.44%** | **97.12%** | 2.88% |
| 1 | 5.99% | 97.35% | 2.65% |
| 5 | 5.99% | 97.27% | 2.73% |
| 10 | 6.37% | 97.27% | 2.73% |
| 14 | 6.75% | 97.27% | 2.73% |
| 20 | 6.75% | 97.35% | 2.65% |
| 29 | 7.58% | 97.57% | 2.43% |
| 41 | 8.64% | 97.73% | 2.27% |
| 58 | **9.78%** ← peak | 98.10% | 1.90% |
| 82 | 9.48% | 98.10% | 1.90% |
| **116** | **9.40%** | **98.18%** | **1.82%** |

### Format collapse — the paper's claim

`tinygsm-code_count` rises monotonically from an already-saturated 97.12% to
98.18%. Stated the other way it is clearer: **`text_count` falls 2.88% → 1.82%,
so 37% of the model's remaining off-format output is eliminated.** RL pushes the
model further onto the single distribution pretraining favoured.

### Accuracy, and the peak

pass@1 rises 46% (6.44% → 9.40%), peaking at **9.78% at step 58** and then
declining. This non-monotonicity is why the paper reports "**Top** pass@1,
pass@64 and majority@64 **across epochs**" — accuracy has a peak and they select
it post hoc.

Note that `code_count` keeps climbing (98.10% → 98.18%) through the window where
pass@1 falls. **The behavioural collapse does not stop when the capability gain
does.** The two curves separate late in training.

### Sampled metrics

| | pass@1 | majority@64 | pass@64 | pass@64 − pass@1 | `code_count`@64 |
|---|---|---|---|---|---|
| before | 6.44% | 6.75% | 50.57% | +44.1 | 94.63% |
| step 58 | 9.78% | 13.87% | 59.44% | +49.7 | 95.55% |
| **step 116** | **9.40%** | **13.95%** | **58.15%** | +48.7 | **95.98%** |

**majority@64 more than doubles (6.75% → 13.95%), the largest relative gain of
the three.** Before RL the correct answer existed in the distribution
(pass@64 = 50.57%) but was almost never the mode (majority@64 = 6.75%). RL's
main effect is turning occasionally-sampled correct answers into the mode.

### KL stays tiny

`kl` ends at **0.0145**. A 46% accuracy gain and a 37% reduction in off-format
output, for a policy that barely moved from its pretrained reference. If RL were
teaching new capability rather than reweighting existing behaviour, the KL could
not be this small.

---

## Predictions vs. outcomes

Written in [PLAN.md](PLAN.md) before running.

**① "tg30's `tinygsm-code_count` rises from 97.12% toward ~100%; the effect may
be visible mainly as the sampled count converging on the greedy one."**
→ **Right on the direction.** Greedy 97.12% → 98.18%, sampled 94.63% → 95.98%.
The gap between them barely narrowed (2.49 → 2.20), so the second half of the
prediction was not what happened — both rose roughly in parallel.

**② "tg00's behaviour is the interesting case. If RL still drives it toward code
format, the source is Algebraic-Stack and the paper's story needs qualifying."**
→ **Interesting, but not in the predicted way.** RL drove it nowhere at all:
`code_count` stayed at exactly 0.00% for 116 steps. The paper's story does not
need qualifying; it gains a negative control it did not have.

**③ "pass@1 rises substantially for tg30."**
→ **Right.** +46% (6.44% → 9.40%), +52% at the peak.

**④ "pass@64 stays flat or falls. RL sharpens rather than extends. If pass@64
climbs meaningfully, RL is doing something more than redistribution."**
→ **Wrong.** pass@64 rose 50.57% → 58.15%, **+15%**.

By the criterion written down in advance, this means **RL did more than
redistribute probability mass over answers the model could already produce.**
Roughly 100 problems that 64 samples could not solve before became solvable.
That is a stronger claim than "amplification" alone, and it is the most
counter-intuitive number in this study.

Caveat: it is one run, one seed, one KL setting. The natural follow-up is 2–3
seeds to establish whether +7.6 points on pass@64 is outside run-to-run
variance.

---

## Method note: test-set selection

The paper reports the maximum of pass@1, pass@64 and majority@64 **across
epochs** (Figure 5 caption). Each maximum is taken independently, so the three
reported numbers may come from different checkpoints, and all are selected on
the test set — which inflates them relative to a pre-committed stopping rule.

This report therefore uses **step 116, the pre-committed endpoint**, as the
headline number everywhere. The step-58 peak is shown alongside so the two
conventions can be compared, and labelled as a peak. With 13 checkpoints from
one episode our selection surface is much smaller than the paper's, but the
distinction is worth stating rather than inheriting.

---

## Run metadata

| | |
|---|---|
| Algorithm | GRPO — `group_norm`, `use_kl_loss`, `kl_estimator k3`, `init_kl_coef 1e-3` |
| Prompts | GSM8K train, 7,473 |
| Reward | rule-based, `openrlhf/utils/math_verifier.py` — no reward model |
| Rollout | 64 prompts x 8 samples, temperature 0.7 |
| Steps | 116 rollout / 934 optimizer, one episode |
| Checkpoints | 13, log-spaced (1, 2, 3, 5, 7, 10, 14, 20, 29, 41, 58, 82, 116) |
| tg30 wall clock | 1h20m (41.7 s/step) |
| tg00 wall clock | **2h54m** (90.0 s/step) — its generations run to the 1024 cap |
| Hardware | same rented RTX 4090; actor + ref + vLLM colocated, no critic |

Two upstream issues had to be worked around; both are recorded in
[../../scripts/run_grpo_single.sh](../../scripts/run_grpo_single.sh) and
`openrlhf/models/utils.py`:

- openrlhf's built-in eval forces temperature 0, and greedy decoding from these
  models produces short degenerate outputs that hit an index mismatch in
  `compute_reward` and kill the run before training starts. Skipped with
  `--eval_steps 0`; nothing is lost, since all numbers here come from the
  paper's own `run_inference_all.py`.
- `compute_reward`'s packed-samples branch assumes the KL segment length equals
  `num_actions`. It now clamps rather than raising, warning once per process.
  It did not fire during training in any of the three runs.
