# exp1 — Results

Status as of 2026-08-06 17:00 UTC. `tg30` complete, `tg00` training, `tg15`
deferred (see PLAN.md § Scope).

| group | pretrain | pass@1 | pass@64 |
|---|---|---|---|
| `tg30` (30% TinyGSM) | done | done | done |
| `tg00` (0% TinyGSM) | running | queued | queued |
| `tg15` (15% TinyGSM) | deferred | — | — |

---

## GSM8K test (1,319 problems)

### greedy — pass@1

| | **tg30**<br>30% TinyGSM, 1B tok | **tg00**<br>0% TinyGSM, 1B tok | **`as_fm3_8xtg`** (authors')<br>28.4% TinyGSM, 74.9B tok |
|---|---|---|---|
| **pass@1** | **6.44%** | _pending_ | **33.06%** |
| `tinygsm-code_count` | **97.12%** | _pending_ | **97.27%** |
| `tinygsm-code_acc` | 6.64% | _pending_ | 33.98% |
| `text_count` | 2.88% | _pending_ | 2.73% |
| `text_acc` | 0.00% | _pending_ | 0.00% |

### sampled, n=64, temperature 0.7

| | **tg30** | **tg00** | **`as_fm3_8xtg`** |
|---|---|---|---|
| **pass@64** | **50.57%** | _pending_ | **83.55%** |
| **majority@64** | **6.75%** | _pending_ | **35.56%** |
| `tinygsm-code_count` | 94.63% | _pending_ | 89.13% |
| `tinygsm-code_acc` | 4.21% | _pending_ | 25.51% |
| `text_count` | 3.88% | _pending_ | 10.29% |
| `text_acc` | 1.49% | _pending_ | 1.49% |

---

## Findings so far

### 1. The format behaviour is fully formed at 1B tokens

`tinygsm-code_count` is **97.12%** for our 1B-token model against **97.27%** for
the authors' model trained on **74.9B** tokens at essentially the same mixture
ratio. A 75x difference in compute moves this metric by 0.15 points.

Accuracy, by contrast, differs 5x (6.44% vs 33.06%).

**Format preference and capability decouple.** The phenomenon the paper studies
is the cheap one — cheap enough that a 5M-parameter model trained for 50 steps
during pipeline testing already emitted 34.5% code-format answers.

### 2. Correct answers are not the mode

| | pass@1 | majority@64 | pass@64 |
|---|---|---|---|
| tg30 | 6.44% | 6.75% | **50.57%** |
| authors' 8xtg | 33.06% | 35.56% | **83.55%** |

For half the problems, at least one of 64 samples is right — but the most common
answer is right only 6.75% of the time. Correct answers are rare individual
draws, not a peak in the distribution. Each of the 64 attempts fails somewhere
different.

This is precisely the shape RL is meant to fix, and it quantifies the headroom:
**+44.1 points** between pass@1 and pass@64 for tg30.

### 3. Sampling drives the better-trained model off-format, not ours

Going from greedy to temperature 0.7, `tinygsm-code_count` falls
97.12% → 94.63% for tg30 but 97.27% → **89.13%** for the authors' model.

The model trained on 75x more data is *more* likely to wander out of the code
format when sampled. Its output distribution is broader — it has real natural
language ability to fall back on. Our undertrained model is locked more rigidly
onto the template.

Worth noting for the paper's framing: the degree of format collapse depends on
how thoroughly the model was pretrained.

### 4. Neither model can answer in prose at all

`text_acc = 0.00%` for both under greedy decoding. These models score entirely
through "write Python, let the interpreter compute". They do not do arithmetic.

---

## Held-out language modelling

Same files for every group; the only cross-group comparable signal.

### Final (tg30, step 1907)

| held-out set | CE loss | perplexity |
|---|---|---|
| tinygsm | 0.4602 | **1.58** |
| algebraic-stack | 1.4106 | 4.10 |
| finemath3 | 2.4079 | 11.11 |

### Cross-group at matched compute (step 200)

| held-out set | **tg00** (0%) | **tg30** (30%) | |
|---|---|---|---|
| **tinygsm** | 4.243 (ppl 69.6) | **1.753 (ppl 5.8)** | tg30 **12x** better |
| algebraic-stack | **3.394** (ppl 29.8) | 3.511 (ppl 33.5) | tg00 11% better |
| finemath3 | **4.387** (ppl 80.4) | 4.437 (ppl 84.5) | tg00 5% better |

**The trade is extremely asymmetric.** Adding 30% TinyGSM buys 2.5 nats on
TinyGSM and costs 0.05–0.12 nats on the background corpora — a 20–50x better
return than cost. TinyGSM is templated and low-entropy, so a little goes a very
long way; the displaced FineMath and Algebraic-Stack are high-entropy real text
where losing 30% barely registers.

This also validates the data pipeline: tg00's TinyGSM held-out loss of 4.243
confirms it has genuinely never seen TinyGSM. A slicing bug that leaked TinyGSM
into tg00 would show up here immediately.

### tg30 held-out trajectory

| step | tokens | tinygsm | algebraic | finemath | Δ finemath |
|---|---|---|---|---|---|
| 200 | 105M | 1.753 | 3.511 | 4.437 | |
| 400 | 210M | 0.747 | 2.022 | 3.171 | −1.266 |
| 600 | 315M | 0.628 | 1.771 | 2.885 | −0.286 |
| 800 | 419M | 0.567 | 1.640 | 2.717 | −0.168 |
| 1000 | 524M | 0.531 | 1.558 | 2.616 | −0.101 |
| 1200 | 629M | 0.503 | 1.502 | 2.536 | −0.080 |
| 1907 | 1000M | 0.460 | 1.411 | 2.408 | |

Still descending at the end — 1B tokens is **not** saturated. Extrapolating the
local power-law slope, each doubling of tokens buys roughly another 11% off the
FineMath held-out loss. The absolute numbers would improve substantially at 10B;
`tinygsm-code_count` would not, since it is already at the ceiling.

---

## Run metadata

| | |
|---|---|
| Hardware | 1x RTX 4090 24GB, rented, $0.651/hr |
| Model | 162,398,208 params (137,822,208 non-embedding) |
| Throughput | 77,363 tok/s mean, min 77,171, max 77,757 over 1,906 samples |
| MFU | 49% (81.4 of 165 TFLOPS bf16) |
| GPU state | 67°C, 405/450 W (power-limited, not thermal), 100% util |
| **tg30 pretraining** | 11:52:59 → 15:29:58 UTC = **3h37m** |
| tg30 → HF conversion | 17 s |
| tg30 pass@1 | 2.5 min |
| tg30 pass@64 | 32 min |
| **tg30 end to end** | **~4h12m** |

Corpus sizes, measured with the Llama-2 tokenizer rather than taken from dataset
cards:

| corpus | shards used | tokens | full corpus |
|---|---|---|---|
| FineMath-3+ | 4 / 128 | 1.295B | ~41.4B |
| Algebraic-Stack | 6 / 79 | 0.923B | ~12.2B |
| TinyGSM | 17 / 17 | 2.665B | 2.665B |

One epoch of the full three-corpus mixture is ~56.3B tokens, which would be
8.4 days per group on this GPU. That figure is also a consistency check on the
paper's own setup: at 4xH200 it works out to roughly 16 hours per group.
