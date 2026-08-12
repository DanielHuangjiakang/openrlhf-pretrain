# exp2 — Results

GRPO on all three pretrained models from [exp1](../exp1-pretrain-mixture/),
identical configuration throughout. The number of episodes became a second axis
during the study, so results are organised as a dose–response along it.

| model | 0 episodes | 1 episode (116 rollout steps) | 2 episodes (232) | 3 episodes (348) |
|---|---|---|---|---|
| `tg00` (0% TinyGSM) | exp1 baseline | complete | not run — see § tg00 | — |
| `tg15` (15%) | exp1 baseline | complete | complete | — |
| `tg30` (30%) | exp1 baseline | complete | complete | complete |

All runs finished. Evaluation is GSM8K **test** (1,319 problems), greedy
decoding unless stated. RL trains on GSM8K **train** (7,473), no overlap.

---

## Headline

Against the pretrained starting point, at each group's longest run:

| | TinyGSM | episodes | pass@1 | `tinygsm-code_count` | `text_count` |
|---|---|---|---|---|---|
| `tg00` | 0% | 1 | 0.68% → **0.68%** (no gain) | 0.00% → **0.00%** | 100% → **100%** |
| `tg15` | 15% | 2 | 3.87% → **7.73%** (+100%) | 89.31% → **95.07%** | 10.69% → **4.93%** (−54%) |
| `tg30` | 30% | 3 | 6.44% → **13.87%** (+115%) | 97.12% → **98.71%** | 2.88% → **1.29%** (−55%) |

Both effects the paper claims reproduce, and `tg00` supplies a negative control
the paper does not have: **with nothing installed by pretraining, RL amplifies
nothing.**

The two models that do learn behave differently in a way worth stating up front:
`tg15` spends its two episodes mostly buying *format*, `tg30` mostly buying
*accuracy*. Over steps 58 → 232,

```
tg15    pass@1  7.13% -> 7.73%  (+0.6 pts)     code_count  90.45% -> 95.07%  (+4.6 pts)
tg30    pass@1 10.69% -> 13.34% (+2.7 pts)     code_count  98.18% -> 98.64%  (+0.5 pts)
```

Each model moves along whichever axis it still has room on. `tg15` started with
9 points of format left to collapse and little accuracy headroom it could reach;
`tg30` started already saturated on format. This is the same "the mixture fixes
format early, capability keeps paying" separation exp1 found in pretraining,
appearing again inside RL.

---

## What the second episode changed — three corrections

The one-episode results supported three claims that **two episodes overturned.**
They are recorded here rather than quietly edited out, because two of them are
the kind of error the paper itself is exposed to.

### ① There is no peak-and-decline. The "peak" was the LR schedule ending.

At one episode `tg30` read as peaking at step 58 and declining:

```
1 episode    step  58: 9.78%    82: 9.48%    116: 9.40%      <- read as over-optimisation
2 episodes   step  58: 10.69%   82: 11.37%   116: 11.45%     <- monotone at the same steps
```

Same model, same data, same hyperparameters — only the horizon differs. The
decline is 0.38 points, **5 problems out of 1,319**, and it lands exactly where
the one-episode cosine reaches `min_lr` (`actor_lr` = 1e-7 by step 116 of a
934-optimizer-step schedule). Stretch the same schedule over two episodes and
the dip is gone.

So the earlier reading — that accuracy peaks and RL then degrades it, which was
offered as the reason the paper reports *"Top pass@1 ... across epochs"* — is
**withdrawn**. Nothing here shows over-optimisation. See § Method note for what
survives of the selection argument.

### ② The paper's Figure 5 ordering reproduces, but only after enough RL.

The paper claims models with the highest TinyGSM share *"exhibit the largest
performance gain from fine-tuning"*. At one episode our data appeared to
contradict it; at two it agrees:

| | 1 episode | 2 episodes |
|---|---|---|
| `tg15` absolute gain | **+3.26 pts** | +3.86 pts |
| `tg30` absolute gain | +2.96 pts | **+6.90 pts** |
| `tg15` relative gain | **+84%** | +100% |
| `tg30` relative gain | +46% | **+107%** |

At one episode `tg15` led on both measures, and the earlier version of this
document argued from that that `tg30` was nearer its ceiling. That was wrong.
Comparing episode 2 alone, *within* the two-episode runs so the schedules match:
it added **+1.89 points to `tg30` and +0.76 to `tg15`**. `tg30` was still gaining
2.5x faster at the point where `tg15` had nearly stopped.

Episode 3 then showed `tg30` reaching its own ceiling (+0.53, inside noise), so
the ordering is not that `tg30` improves without limit — it is that `tg30`'s
limit is further out. Both models saturate; the one with more TinyGSM in
pretraining saturates later and higher.

The lesson is about stopping rules, not about the models: **a single episode is
not long enough to rank conditions by RL gain**, and the ranking it produces is
inverted rather than merely noisy.

### ③ Format collapse and accuracy do decouple — but not where we first said.

The one-episode reading located the decoupling in `tg30`, where `code_count`
kept rising through the window pass@1 supposedly fell. With ① withdrawn, that
instance is gone.

The decoupling is real and shows up cleanly in `tg15` instead. Split its
two-episode run at step 58:

| | first 58 steps | remaining 174 steps |
|---|---|---|
| pass@1 | 3.87% → 7.13% (**+3.26 pts, 84% of the total gain**) | 7.13% → 7.73% (+0.60) |
| `text_count` | 10.69% → 9.55% (−1.14 pts, 20% of the total fall) | 9.55% → 4.93% (**−4.62, 80%**) |

**Accuracy is bought early; format is collapsed late.** The sharpest case is
steps 164 → 232, where pass@1 does not move at all (7.73% → 7.73%) and
`text_count` still falls 5.08% → 4.93%. Behaviour keeps moving after capability
has stopped.

---

## Per-model detail

### `tg00` — RL cannot bootstrap from nothing

One full episode (116 rollout steps, 2h54m) changed nothing measurable.

| RL step | pass@1 | `text_count` | | RL step | pass@1 | `text_count` |
|---|---|---|---|---|---|---|
| before | 0.68% | 100.00% | | 14 | 0.61% | 100.00% |
| 1 | 0.61% | 100.00% | | 20 | 0.45% | 100.00% |
| 2 | 0.68% | 100.00% | | 29 | 0.68% | 100.00% |
| 3 | 0.61% | 100.00% | | 41 | 0.76% | 100.00% |
| 5 | 0.53% | 100.00% | | 58 | 0.76% | 100.00% |
| 7 | 0.45% | 100.00% | | 82 | 0.61% | 100.00% |
| 10 | 0.53% | 100.00% | | **116** | **0.68%** | **100.00%** |

pass@1 oscillates in 0.45–0.76% with no trend — a range of **4 problems out of
1,319** — and `text_count` is exactly 100.00% on **all thirteen** checkpoints.
Mean reward over the first 20 rollout steps is 0.0133 ± 0.005 and over the last
20 is 0.0120 ± 0.005: the run is flat, not slow.

`tinygsm-code_count` is reported as 0.00% here and in exp1, but note that the
evaluator does not emit the key at all for this model — the JSON contains only
`final_accuracy`, `text_acc` and `text_count`. The zero is the complement of a
directly measured `text_count = 1.0`, not a separate measurement.

Two reasons, and only the first is about sample efficiency:

1. **Too few informative groups.** GRPO normalises advantage within a group of
   8, so a group whose generations all score alike contributes no gradient. At
   0.68% pass@1, `1 − 0.9932⁸ = 5.3%` of groups carry signal, against 41% for
   `tg30`. Raising `n_samples_per_prompt` to 32 would lift that to 19.6%.
2. **The reward is not measuring what its name says.** `tg00`'s "correct"
   answers are the answer parser recovering a number from unbounded repetition
   (exp1 § tg00). Optimising it would reinforce repetition that happens to
   contain the right number — reward hacking, not learning.

A second episode was not run. More episodes fix neither reason, and the flat
line is already the result: **RL amplifies what pretraining
installed, and when pretraining installed nothing usable, there is nothing to
amplify.** The paper has no 0% condition, so this direction of its claim is
untested there.

### `tg15` — format keeps collapsing after accuracy stops

| RL step | pass@1 | `code_count` | `text_count` | |
|---|---|---|---|---|
| **before** | **3.87%** | **89.31%** | **10.69%** | |
| 29 | 4.40% | 89.16% | 10.84% | |
| 58 | 7.13% | 90.45% | 9.55% | |
| 82 | 6.67% | 91.28% | 8.72% | |
| 116 | 6.97% | 93.18% | 6.82% | end of episode 1 |
| 164 | 7.73% | 94.92% | 5.08% | |
| **232** | **7.73%** | **95.07%** | **4.93%** | |

84% of the accuracy gain lands in the first 58 steps; 80% of the format collapse
lands after them. `text_count` more than halves overall. This is correction ③
above, where the split is tabulated.

The 1-episode run (a separate run to step 116, with the shorter schedule) ended
at pass@1 7.13% / `code_count` 91.51% / `text_count` 8.49%.

### `tg30` — accuracy keeps paying

| RL step | pass@1 | `code_count` | `text_count` | |
|---|---|---|---|---|
| **before** | **6.44%** | **97.12%** | **2.88%** | |
| 20 | 6.82% | 97.42% | 2.58% | |
| 41 | 9.55% | 97.88% | 2.12% | |
| 58 | 10.69% | 98.18% | 1.82% | |
| 82 | 11.37% | 98.41% | 1.59% | |
| 116 | 11.45% | 98.41% | 1.59% | end of episode 1 |
| 164 | 12.51% | 98.56% | 1.44% | |
| 232 | 13.34% | 98.64% | 1.36% | end of episode 2 |
| 240 | 13.65% | 98.64% | 1.36% | |
| 246 | 12.89% | 98.64% | 1.36% | |
| 280 | 13.65% | 98.64% | 1.36% | |
| 320 | 14.03% | 98.64% | 1.36% | |
| 340 | **14.33%** | 98.71% | 1.29% | highest of the 23 |
| **348** | **13.87%** | **98.71%** | **1.29%** | end of episode 3 |

Monotone through episode 2, then flat. **Episode 3 bought nothing measurable.**
Per-episode pass@1 gain:

```
episode 1   6.44% -> 11.45%   +5.01 pts
episode 2  11.45% -> 13.34%   +1.89 pts
episode 3  13.34% -> 13.87%   +0.53 pts      <- inside the noise band
```

Roughly a third of the previous episode's gain each time. Episode 3's +0.53 is
not distinguishable from noise: within that episode alone the nine checkpoints
span 12.89%–14.33%, a range of **±0.7 points (±9 problems)** with no trend, and
the endpoint is below the maximum. `code_count` moves 0.07 points across the
whole episode.

Episode 3 was run because pass@1 was still rising at step 232 and the learning
rate had decayed to `min_lr`, leaving the two explanations — model saturated vs.
schedule exhausted — confounded. The warm restart (§ Extending a finished run)
raised the LR 3.4x and the model did not respond, so **the ceiling is the
model's, not the schedule's.** That is worth the 1h50m it cost: without it the
2-episode endpoint would have looked like an arbitrary stopping point.

`code_count` moves only 1.5 points because it starts saturated. Read from the
other side it is not small: **`text_count` falls 2.88% → 1.36%, so RL eliminates
53% of the model's remaining off-format output.** For scale, exp1 measured that
a **75-fold** increase in pretraining tokens moves this metric by 0.15 points
(`tg30` 97.12% vs `as_fm3_8xtg` 97.27%). Three hours of GRPO moves it ten times
further than 75x the pretraining compute does. The behaviour axis is cheap to
push and expensive to reach by scale — which is exactly the asymmetry that makes
the paper's concern a practical one.

---

## Training-side signals

Reward is a single rollout batch (64 prompts x 8 samples) and is far too noisy
to read endpoint-to-endpoint — `tg30`'s 2-episode series visits 0.058, 0.160,
0.104 and 0.148 in its last four sampled steps. All figures below are **means
over the first and last 20 rollout steps**, with the standard deviation across
that window, which is the smallest honest summary.

| run | reward, first 20 → last 20 | `kl`, first 20 → last 20 | `response_length` |
|---|---|---|---|
| `tg00` 1 ep | 0.0133 ± 0.005 → **0.0120 ± 0.005** | 0.00010 → **0.00033** | 899 → 896 |
| `tg15` 1 ep | 0.0334 ± 0.013 → **0.0633 ± 0.021** | 0.00013 → **0.01736** | 263 → 231 |
| `tg15` 2 ep | 0.0340 ± 0.015 → **0.0919 ± 0.024** | 0.00012 → **0.03395** | 258 → 204 |
| `tg30` 1 ep | 0.0541 ± 0.013 → **0.1077 ± 0.030** | 0.00013 → **0.01475** | 227 → 213 |
| `tg30` 2 ep | 0.0555 ± 0.018 → **0.1415 ± 0.021** | 0.00011 → **0.02678** | 229 → 187 |

**KL rises then plateaus; it does not run away.** Sampled across `tg30`'s
2-episode run: 0.0045 (29), 0.0212 (58), 0.0285 (87), 0.0233 (116), 0.0223
(145), 0.0291 (174), 0.0292 (203), 0.0267 (232) — flat from step 58 onward.
`tg15` plateaus the same way, around 0.038 from step 174. This is the
measurement the plan said would decide between *amplification* and *retraining*,
and it reads as amplification: the policy settles a small fixed distance from
its pretrained reference and stays there while reward goes on rising 2.5x.

**`tg15` ends further from its reference than `tg30`** (0.0340 vs 0.0268)
despite lower reward and lower accuracy. The model with less TinyGSM in
pretraining has to move further to produce the same behaviour. At one episode
the two were indistinguishable (0.0174 vs 0.0148) and this document previously
described that as a threshold effect; with the longer run they separate, in the
direction the amplification story predicts.

**Generations get shorter as format tightens** — `tg30` 229 → 187 tokens
(−18%), `tg15` 258 → 204 (−21%) — while `tg00`, which never adopts the format,
stays pinned near 900 and generates until it hits the cap.

---

## Sampled decoding, n = 64

Temperature 0.7, 64 samples per problem. Steps 58 and 116 come from the
1-episode run; 232 and 348 from the 2/3-episode run. Both start from the same
pretrained checkpoint, but they are **separate runs with different LR
schedules** — read each series against the shared row 0, not across them.

| `tg30` | run | pass@1 | pass@64 | majority@64 | pass@64 − pass@1 | `code_count`@64 |
|---|---|---|---|---|---|---|
| pretrained | — | 6.44% | 50.57% | 6.75% | +44.1 | 94.63% |
| step 58 | 1 ep | 9.78% | 59.44% | 13.87% | +49.7 | 95.55% |
| step 116 | 1 ep | 9.40% | 58.15% | 13.95% | +48.8 | 95.98% |
| step 232 | 2 ep | 13.34% | 59.82% | **17.13%** | +46.5 | 97.46% |
| step 348 | 3 ep | 13.87% | **60.58%** | 16.91% | +46.7 | 97.61% |

| `tg15` | run | pass@1 | pass@64 | majority@64 | pass@64 − pass@1 | `code_count`@64 |
|---|---|---|---|---|---|---|
| pretrained | — | 3.87% | 45.49% | 5.76% | +41.6 | 83.77% |
| step 232 | 2 ep | 7.73% | 52.01% | 12.66% | +44.3 | 91.19% |

### pass@64 moves once, early, and then stops

**Every post-RL measurement lands in 58.15–60.58%, whether RL ran 58 steps or
348.** The jump off the 50.57% baseline is +8.9 points and it is already complete
at the earliest checkpoint measured; six times more RL adds at most another 1.1
points, which is inside the spread of the measurements themselves.

Meanwhile pass@1 goes on rising to 13.87% and majority@64 to 17.13%. So RL has
two phases, and only the first one adds anything to the model:

| | pass@1 | pass@64 | majority@64 |
|---|---|---|---|
| **early** (to step 58) | +3.3 pts | **+8.9 pts** | +7.1 pts |
| **late** (58 → 348) | **+4.1 pts** | +1.1 pts | +3.0 pts |

Early RL **extends** what the model can produce at all; late RL only
**redistributes** probability onto answers already inside the distribution. The
`pass@64 − pass@1` gap tells the same story from the other side: it widens
44.1 → 49.7 during the extension phase, then closes back to 46.7 as
sharpening takes over.

### majority@64 is where the gain concentrates

6.75% → 17.13%, **2.5x** — the largest relative move of any metric here, and
larger than pass@1's 2.2x. Before RL the correct answer usually existed among 64
samples (50.57%) but was almost never the mode (6.75%). Turning
occasionally-sampled correct answers into the mode is the bulk of what this RL
does, which is also why greedy pass@1 rises faster than the sampled training
reward.

### RL closes the greedy–sampled format gap

exp1 found that sampling drives these models off-format — `tg30` writes code
97.12% of the time greedily but only 94.63% at temperature 0.7. RL removes most
of that gap:

```
                 greedy   sampled   gap
pretrained       97.12%    94.63%   2.49
1 ep, step 116   98.18%    95.98%   2.20
2 ep, step 232   98.64%    97.46%   1.18
3 ep, step 348   98.71%    97.61%   1.10
```

The format preference stops being a property of the mode and becomes a property
of the whole output distribution. `text_count` under sampling falls 3.88% →
1.36%, a **65% reduction** — larger than the 55% seen greedily. Sampled
decoding, which is where these models used to escape the template, is where RL
tightens it most.

`tg15` shows the same pattern from further out: `code_count`@64 83.77% → 91.19%,
`text_count`@64 12.67% → 6.82% (−46%).

---

## Predictions vs. outcomes

Written in [PLAN.md](PLAN.md) before running, and left unedited there.

**① "`tg30`'s `tinygsm-code_count` rises from 97.12% toward ~100%; the effect may
be visible mainly as the sampled count converging on the greedy one."**
→ **Right, including the mechanism** — but only visible past one episode. Greedy
goes 97.12% → 98.71%. The greedy–sampled gap barely moved over episode 1
(2.49 → 2.20), which an earlier version of this document scored as the mechanism
being wrong; by episode 3 it has more than halved, to 1.10. The convergence is
real and slower than the headline metric.

**② "`tg00`'s behaviour is the interesting case. If RL still drives it toward
code format, the source is Algebraic-Stack and the paper's story needs
qualifying."**
→ **Interesting, but not in the predicted way.** RL drove it nowhere:
`code_count` stayed at exactly 0.00% on all 13 checkpoints. The paper's story
does not need qualifying; it gains a negative control it did not have.

**③ "pass@1 rises substantially for `tg30`."**
→ **Right, and understated.** +115% by episode 3 (6.44% → 13.87%), though the
gain is exhausted by the end of episode 2.

**④ "pass@64 stays flat or falls. RL sharpens rather than extends. If pass@64
climbs meaningfully, RL is doing something more than redistribution."**
→ **Wrong early, right late.** The prediction assumed one regime; there are two.
pass@64 climbs +8.9 points in the first 58 steps — RL genuinely extending the
distribution, roughly 117 problems that 64 samples could not solve before —
and then does not move for the next 290 steps while pass@1 gains another 4.1.

So both halves of the prediction are correct about *some* phase of the run and
neither describes the whole of it. Had we stopped at step 58 (where the
1-episode reading initially put the peak) the conclusion would have been
"extension"; had we only compared the two endpoints, "redistribution". This is
the finding that most depended on running the study long enough, and it is the
one a follow-up should replicate first: one seed, one KL setting.

---

## Method notes

### Extending a finished run

Episode 3 resumes from the 2-episode DeepSpeed state (`LOAD_CKPT=1
EPISODES=3`), which costs ~1h40m instead of ~5h for a fresh 3-episode run. The
schedule is not seamless, and the seam is documented because it is visible in
the curve:

`max_steps` is `ceil(num_episodes x 934)`, so the scheduler is rebuilt for the
new horizon (2,802 steps), while `load_ckpt(..., load_lr_scheduler_states=True)`
restores the step counter at 1,868. Evaluated with the repo's own scheduler
construction:

```
2ep run at its endpoint  (last_epoch=1868, total=1868)   1.000e-07   = min_lr
resumed as 3ep           (last_epoch=1868, total=2802)   3.374e-07   <- warm restart
a fresh 3ep run at step 1868                             3.374e-07   <- identical
```

The learning rate jumps 3.4x at the seam and then follows the 3-episode cosine
tail — **identical to a fresh 3-episode run from step 1,868 onward.** Only the
history differs.

In the event the seam is invisible in the curve: 13.34% (232) → 13.65% (240) →
12.89% (246) → 13.34% (260), a wobble no larger than the ±0.7-point noise band
elsewhere in the episode. The restart is worth documenting anyway, because it is
what makes episode 3 informative: the model was handed 3.4x the learning rate it
had been running at and still did not improve.

The resumed run also writes to the directory named `...-grpo-2ep`, because
`--ckpt_path` has to point at the existing `_actor` state. **That directory
contains 3-episode results.** Renaming it would break resumption.

### Test-set selection

The paper reports the maximum of pass@1, pass@64 and majority@64 **across
epochs** (Figure 5 caption). Each maximum is taken independently, so the three
numbers may come from different checkpoints, and all are selected on the test
set.

Correction ① removed the evidence that accuracy peaks and declines, so the
argument that max-across-epochs captures a real peak is gone. The selection
concern is narrower but still stands: with a monotone curve, taking the maximum
is nearly the same as taking the endpoint, and the inflation is small — but it
is not zero, and it is not stated. Every headline number in this document is the
**pre-committed endpoint** (step 232, or 116 for `tg00`), never a selected
maximum.

### Reward is not the metric

Training reward is sampled at temperature 0.7 on GSM8K *train*; pass@1 is greedy
on GSM8K *test*. They are not interchangeable, and the gap is informative: over
episode 2 `tg30`'s windowed reward rose 2.5x while greedy pass@1 rose 2.1x, with
majority@64 (measured at 1 episode) rising fastest of all. RL is concentrating
probability mass on answers the model could already sometimes produce.

---

## Run metadata

| | |
|---|---|
| Algorithm | GRPO — `group_norm`, `use_kl_loss`, `kl_estimator k3`, `init_kl_coef 1e-3` |
| Prompts | GSM8K train, 7,473 |
| Reward | rule-based, `openrlhf/utils/math_verifier.py` — no reward model |
| Rollout | 64 prompts x 8 samples, temperature 0.7 |
| Steps | 116 rollout / 934 optimizer per episode |
| LR | 1e-6 peak, `cosine_with_min_lr`, `min_lr` 1e-7, 3% warmup |
| Checkpoints | 13 (1 ep) / 15 (2 ep) / 23 (3 ep), log-spaced via `--save_log_scale_count 15`, plus `--save_steps 20` in episode 3 |
| Hardware | one rented RTX 4090; actor + ref + vLLM colocated, no critic |

| run | wall clock |
|---|---|
| `tg00` 1 ep | 2h54m (90.0 s/step) — generations run to the 1024 cap |
| `tg30` 1 ep | 1h20m (41.7 s/step) |
| `tg15` 2 ep | 2h49m (episode 2: 1h22m, 42.7 s/step) |
| `tg30` 2 ep | 3h18m (episode 2: 1h37m, 50.1 s/step) |
| `tg30` 3rd ep | 1h50m, resumed — a fresh 3-episode run would have been ~5h |

Evaluation: pass@1 ~2.5 min per checkpoint; pass@64 22 min (`tg30`) to 36 min
(`tg15`, longer generations) with `--max_num_seqs 1024`, against ~32 min at
vLLM's default cap of 256. That flag was added to
`inference/run_inference_all.py` for this study and defaults to unset, so every
earlier measurement still reproduces.

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
  It did not fire during training in any run.

A third is worth recording as a planning note rather than a bug: the log-scale
checkpoint grid is computed over the *whole* run, so extending to 3 episodes
placed only two new checkpoints (246, 348) in the added episode. `--save_steps
20` was passed alongside it to densify the tail to 8 points.
