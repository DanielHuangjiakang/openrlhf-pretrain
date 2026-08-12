# exp1 — Pretraining mixture sweep

## Question

Does the share of TinyGSM in pretraining determine what *format* a model answers
in — specifically, whether it writes a Python function and lets the interpreter
do the arithmetic, versus answering in prose?

This is the first half of the Echo Chamber thesis. The second half (does RL
amplify it) is [exp2](../exp2-rl-amplification/).

## Design

Three models, **identical compute**, one variable: TinyGSM's share.

| group | TinyGSM | FineMath-3+ | Algebraic-Stack | total |
|---|---|---|---|---|
| `tg00` | 0 | 244,096 | 244,096 | 488,192 blocks |
| `tg15` | 73,229 (15.00%) | 207,482 | 207,481 | 488,192 |
| `tg30` | 146,458 (30.00%) | 170,867 | 170,867 | 488,192 |

1 block = 2,048 tokens, so every group trains on **999,817,216 tokens** in
**1,907 steps** (488,192 / batch 256). The two background corpora are held at
1:1 with each other in every group, so the only thing that changes is how much
background gets displaced by TinyGSM.

### Nested subsets

Each group takes a *prefix* of every source file, so the smaller take is a
byte-exact subset of the larger:

```
tinygsm           tg00 ⊆ tg15 ⊆ tg30       (higher share -> take more)
finemath3         tg30 ⊆ tg15 ⊆ tg00       (higher share -> take less)
algebraic-stack   tg30 ⊆ tg15 ⊆ tg00
```

Going 0% → 15% → 30% therefore does exactly one thing: swap a tail of background
text for an equal number of TinyGSM blocks. Everything the groups share is
identical token for token. `build_mixture.py --verify` re-reads the written
bytes and confirms this with sha256 rather than trusting the arithmetic.

### Held-out evaluation

2,000 blocks per source are reserved from the *tail* of every file and excluded
from all three groups, so leakage is structurally impossible rather than merely
unlikely.

This matters more than usual here: **train loss is not comparable across
groups.** TinyGSM is templated synthetic code and scores far lower than web text
regardless of model quality, so a group with more TinyGSM has lower train loss
for reasons that say nothing about the model. The three held-out sets are the
same files for every group and are the only cross-group signal.

## Metrics

| metric | what it measures |
|---|---|
| **`tinygsm-code_count`** | **the actual research question** — fraction of generations that answer with Python |
| `tinygsm-code_acc` / `text_acc` | accuracy *within* each format |
| pass@1 | greedy accuracy; also the gate for whether GRPO can run at all |
| pass@64 | whether the ability exists anywhere in the distribution |
| majority@64 | whether the correct answer is the *mode* |
| `pass@64 − pass@1` | headroom RL has to work with |
| held-out CE loss × 3 | language-modelling cost/benefit of the mixture change |

Scoring is the paper's own `openrlhf/utils/math_verifier.py`: a generation
containing `def` is classified `tinygsm-code`, its function is `exec()`d, and the
return value is compared against the gold answer. Everything else is parsed as
text by `math-verify`.

Evaluation is GSM8K **test** (1,319 problems). RL in exp2 uses GSM8K **train**
(7,473). No overlap.

## Reference points

The authors released 150M checkpoints over the same three corpora with TinyGSM
repeated 1/2/4/8 times. Using our measured corpus sizes (FineMath-3+ 41.4B,
Algebraic-Stack 12.2B, TinyGSM 2.66B tokens), their shares work out to:

| their model | TinyGSM share | tokens | closest group of ours |
|---|---|---|---|
| `as_fm3_tg` | 4.7% | 56.3B | — |
| `as_fm3_2xtg` | 9.0% | 58.9B | — |
| `as_fm3_4xtg` | 16.6% | 64.3B | `tg15` (15%) |
| `as_fm3_8xtg` | 28.4% | 74.9B | **`tg30` (30%)** |

`as_fm3_8xtg` is a near-perfect control: essentially the same mixture ratio,
**75x the tokens**. Comparing against it separates "what the mixture does" from
"what the token budget does".

## Scope

Planned as `tg00` and `tg30` only — the extremes; if 0% vs 30% showed no
difference in `tinygsm-code_count`, the midpoint would not either, and the
remaining GPU time was better spent on exp2.

**`tg15` was run after all**, and the deferral turned out to be the wrong call.
The extremes are 0.00% and 97.12%, which says nothing about the *shape* between
them; `tg15`'s 89.31% is what shows the transition is steep-but-graded rather
than a step function, and it is the only point either this study or the paper
has below saturation. All three groups are reported in
[RESULTS.md](RESULTS.md).

## Known limitation

1B tokens is 6.2 tokens/parameter, well under Chinchilla's ~20, and the held-out
loss was still falling when training stopped. Absolute accuracy is therefore far
below the paper's. This is a deliberate trade: compute-matched comparability
across groups, at the cost of absolute capability. The `tinygsm-code_count`
comparison against `as_fm3_8xtg` is what tests whether that trade was sound.
