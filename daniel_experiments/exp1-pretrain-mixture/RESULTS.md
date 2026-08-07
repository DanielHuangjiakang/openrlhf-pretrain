# exp1 — Results

All three groups complete. Five of the authors' released 150M checkpoints
evaluated with the same script for reference.

All numbers are GSM8K **test** (1,319 problems), scored by the paper's own
`openrlhf/utils/math_verifier.py`.

---

## The headline: format saturates early, capability does not

Greedy decoding, pass@1. Ours at 1B tokens; the authors' at 56–75B.

| TinyGSM share | model | tokens | **`tinygsm-code_count`** | **pass@1** |
|---|---|---|---|---|
| **0%** | **ours `tg00`** | **1B** | **0.00%** | 0.68% * |
| 4.7% | `as_fm3_tg` | 56.3B | 96.74% | 24.03% |
| 9.0% | `as_fm3_2xtg` | 58.9B | 99.01% | 26.23% |
| **15%** | **ours `tg15`** | **1B** | **89.31%** | **3.87%** |
| 16.6% | `as_fm3_4xtg` | 64.3B | 99.17% | 30.78% |
| 28.4% | `as_fm3_8xtg` | 74.9B | 97.27% | 33.06% |
| **30%** | **ours `tg30`** | **1B** | **97.12%** | **6.44%** |
| 100% | `4xtg` | ~10.6B | 99.32% | 44.58% |

Bold rows are ours, at 1B tokens; the rest are the authors' released
checkpoints at 56–75B. **The two groups are not directly comparable** — mixture
and budget both differ — which is exactly the confound the compute-matched rows
below separate out.

\* artifact, not capability — see § tg00 below.

**`tinygsm-code_count` saturates early, but it is not a step function.** Read
only the authors' row it looks like one — 0% to 4.7% appears to jump from 0 to
96.74% — but those two points differ by 56x in tokens as well as in mixture. The
three compute-matched groups separate the variables:

```
TinyGSM     0%        15%       30%
code_count  0.00%  →  89.31%  →  97.12%
text_count  100%   →  10.69%  →   2.88%
```

Steep but graded: 89 points of the change happen by 15%, the remaining 8 over
the next 15. Roughly 15% of the mixture is enough to nearly fix how the model
answers; beyond that the axis is close to exhausted.

**pass@1 is monotone and unsaturated**: 24% → 26% → 31% → 33% → 45%. It keeps
paying all the way to 100% TinyGSM.

The authors' lowest setting is 4.7%, where the format metric is already at
ceiling — **the transition is not visible anywhere in their released models.**
The 0% point is ours.

### Token budget moves one and not the other

At essentially the same mixture ratio:

| | tokens | `tinygsm-code_count` | pass@1 |
|---|---|---|---|
| ours `tg30` (30%) | **1B** | **97.12%** | **6.44%** |
| `as_fm3_8xtg` (28.4%) | **74.9B** | 97.27% | 33.06% |
| difference | **75x** | **0.15 points** | **5.1x** |

A 75-fold difference in compute moves the behaviour metric by 0.15 points and
the accuracy metric by a factor of five. **Format preference and capability are
separable, and the phenomenon this paper studies is the cheap one** — cheap
enough that a 5M-parameter model trained for 50 steps during pipeline testing
already emitted 34.5% code-format answers.

---

## Sampled decoding, n=64, temperature 0.7

| | pass@64 | majority@64 | `tinygsm-code_count` | `text_count` |
|---|---|---|---|---|
| ours `tg15` | **45.49%** | 5.76% | 83.77% | 12.67% |
| ours `tg30` | **50.57%** | 6.75% | 94.63% | 3.88% |
| `as_fm3_8xtg` | **83.55%** | 35.56% | 89.13% | 10.29% |

**Correct answers are not the mode.** For half the problems tg30 gets at least
one of 64 samples right, but the most common answer is right only 6.75% of the
time. Each of the 64 attempts fails somewhere different. The gap —
**pass@64 − pass@1 = +44.1 points** — is the headroom RL has to work with, and
[exp2](../exp2-rl-amplification/RESULTS.md) measures what RL does with it.

**Sampling drives the better-trained model off-format, not ours.** Going greedy
→ temperature 0.7, `tinygsm-code_count` falls 97.12% → 94.63% for tg30 but
97.27% → **89.13%** for the authors' model. The model trained on 75x more data
is *more* likely to leave the code format when sampled: its output distribution
is broader, with real natural-language ability to fall back on. Ours is locked
more rigidly onto the template. Format collapse is not purely a function of the
mixture — how thoroughly the model was pretrained matters too.

**Neither model can answer in prose.** `text_acc = 0.00%` for every model in
this table under greedy decoding. They score entirely through "write Python, let
the interpreter compute".

---

## tg00: why its accuracy numbers are not measurements

`tg00` produces **no code at all** (`tinygsm-code_count = 0.00%`,
`text_count = 100.00%`). Its outputs are unbounded repetition:

```
Q: Every day, Wendi feeds each of her chickens three cups of ...   (gold: 20)
A: 1. 20 cups of feed. 2. 20 cups of feed. 3. 20 cups of feed. ...  (x156)
```

`math_verify` recovers "20" from that and scores it correct. **All 9 of its
"correct" answers have the number buried past token ~1,000 inside the loop.**
The 0.68% is the answer parser latching onto a number in a wall of repetition,
not the model solving anything.

**pass@64 was not measured for tg00**, deliberately. Nothing emits `</s>`, so
51% of generations run to the 2,048-token cap and the median generation is 1,897
tokens against tg30's 200. Evaluation runs at 5.7 sequences/s instead of 29 —
**3.8 hours** to estimate a quantity that does not mean what the metric name
implies. Truncating to speed it up would destroy the measurement rather than
approximate it: capping at 512 tokens loses 9 of the 9 "correct" answers, and
that zero would be an artifact of the cap, not a property of the model.

`tinygsm-code_count = 0.00%` and `text_count = 100.00%` already characterise
this model completely.

### It can model TinyGSM without ever choosing to write it

| held-out set | **tg00** (0%) | **tg30** (30%) | |
|---|---|---|---|
| **tinygsm** | 1.449 (ppl 4.3) | **0.460 (ppl 1.58)** | tg30 better by 0.99 nats |
| algebraic-stack | **1.366** (ppl 3.9) | 1.411 (ppl 4.10) | tg00 better by 0.045 |
| finemath3 | **2.351** (ppl 10.5) | 2.408 (ppl 11.11) | tg00 better by 0.057 |

tg00 reaches perplexity 4.3 on held-out TinyGSM having never seen a token of it,
by transfer from Algebraic-Stack's mathematical code. **It can predict TinyGSM
perfectly well. It just never chooses to produce it.**

Being able to model a distribution and being disposed to generate it are
different things. The disposition comes from the *pairing* in the training data
— TinyGSM pairs a word problem with a Python solution — not from exposure to
code in general. This strengthens the paper's claim rather than weakening it.

### The trade is 10x asymmetric

Adding 30% TinyGSM buys **0.99 nats** on TinyGSM and costs **0.10 nats** across
the two background corpora. TinyGSM is templated and low-entropy, so a little
goes very far; the displaced FineMath and Algebraic-Stack are high-entropy real
text where losing 30% barely registers.

This also validates the data pipeline: tg00's TinyGSM held-out loss confirms it
genuinely never saw TinyGSM. A slicing bug that leaked any into tg00 would show
up here immediately.

---

## Held-out trajectory (tg30)

| step | tokens | tinygsm | algebraic | finemath | Δ finemath |
|---|---|---|---|---|---|
| 200 | 105M | 1.753 | 3.511 | 4.437 | |
| 400 | 210M | 0.747 | 2.022 | 3.171 | −1.266 |
| 600 | 315M | 0.628 | 1.771 | 2.885 | −0.286 |
| 800 | 419M | 0.567 | 1.640 | 2.717 | −0.168 |
| 1000 | 524M | 0.531 | 1.558 | 2.616 | −0.101 |
| 1200 | 629M | 0.503 | 1.502 | 2.536 | −0.080 |
| 1907 | 1000M | 0.460 | 1.411 | 2.408 | |

Still descending at the end — **1B tokens is not saturated**. Extrapolating the
local power-law slope, each doubling of tokens buys roughly another 11% off the
FineMath held-out loss. Absolute accuracy would improve substantially at 10B;
`tinygsm-code_count` would not, since it is already at ceiling.

---

## Run metadata

| | |
|---|---|
| Hardware | 1x RTX 4090 24GB, rented, $0.651/hr |
| Model | 162,398,208 params (137,822,208 non-embedding) |
| Throughput | 77,363 tok/s mean; min 77,171, max 77,757 over 1,906 samples (±0.4%) |
| MFU | 49% (81.4 of 165 TFLOPS bf16) |
| GPU | 67°C, 405/450 W — power-limited, not thermal — 100% util |
| tg30 pretraining | 11:52:59 → 15:29:58 UTC = **3h37m** |
| tg00 pretraining | 17:21 → 20:58 UTC = **3h37m** |
| HF conversion | 17 s |
| pass@1 | 2.5 min · pass@64 32 min (tg30-like models) |

Corpus sizes, measured with the Llama-2 tokenizer rather than taken from dataset
cards:

| corpus | shards used | tokens | full corpus |
|---|---|---|---|
| FineMath-3+ | 4 / 128 | 1.295B | ~41.4B |
| Algebraic-Stack | 6 / 79 | 0.923B | ~12.2B |
| TinyGSM | 17 / 17 | 2.665B | 2.665B |

One epoch of the full three-corpus mixture is ~56.3B tokens — 8.4 days per group
on this GPU. That figure also cross-checks the paper's own setup: on 4xH200 it
works out to roughly 16 hours per group.
