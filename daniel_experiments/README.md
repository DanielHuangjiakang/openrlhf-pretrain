# Experiments

Reproduction of *Echo Chamber: RL Post-training Amplifies Behaviors Learned in
Pretraining* ([arXiv 2504.07912](https://arxiv.org/abs/2504.07912)) at 150M
scale, with two methodological corrections to the original codebase.

| # | Experiment | Question | Status |
|---|---|---|---|
| [exp1](exp1-pretrain-mixture/) | Pretraining mixture sweep | Does the TinyGSM share in pretraining determine the model's output *format*? | **complete** — 3 groups at 0/15/30%, plus 5 released checkpoints as reference |
| [exp2](exp2-rl-amplification/) | RL amplification | Does GRPO amplify that format preference? | **complete** — all 3 groups, up to 3 GRPO episodes, with pass@64 at the endpoints |

Headline: both halves of the paper's claim reproduce at 1B tokens — pretraining
mixture sets the answer format, and GRPO pushes it further while roughly doubling
accuracy — and the 0% group, which the paper does not have, shows RL amplifying
**nothing** when pretraining installed nothing.

| | TinyGSM | pass@1, pretrained → best GRPO | `text_count` (off-format output) |
|---|---|---|---|
| `tg00` | 0% | 0.68% → 0.68% | 100% → 100% |
| `tg15` | 15% | 3.87% → 7.73% | 10.69% → 4.93% |
| `tg30` | 30% | 6.44% → 13.87% | 2.88% → 1.29% |

Two findings that go beyond the paper, both of which needed the study run longer
than planned: RL has an **extension phase and a redistribution phase** — pass@64
gains its entire +8.9 points in the first 58 of 348 steps and then never moves,
while pass@1 keeps climbing — and **RL saturates**, with per-episode pass@1 gains
of +5.0, +1.9, +0.5 points.

Format collapse outlasts the accuracy gain where there is any format left to
collapse: `tg15` does 80% of its collapsing after 84% of its accuracy gain is
already banked. `tg30` starts at 97% code output and runs out of both at once.

Each directory holds `PLAN.md` (design and rationale, written before running),
`RESULTS.md` (measured numbers) and `logs/`.

## What differs from the original codebase

Both changes exist because the original setup makes the mixture and the compute
budget the same knob, which means no two mixtures in the paper were ever trained
on the same number of tokens.

**1. Fixed token budget instead of one epoch.** Upstream sets
`max_duration: 1ep`, so training stops when the data runs out and total compute
is a function of how much data the mixture happens to contain. Adding 8x TinyGSM
also makes the run 30% longer. We set `max_duration: 1e9T`, so every group sees
999,817,216 tokens in 1,907 optimizer steps.

**2. Exact proportional slicing instead of file duplication.** Upstream raises a
dataset's share by symlinking its files N times (`mix_datasets.py`) — the "4x"
in the paper's model names. Shares are then only adjustable in whole-dataset
multiples, and the resulting percentage is never stated anywhere.
`pretraining/data/build_mixture.py` instead slices each `.ds` at an exact block
count. Groups take prefixes, so shared data is byte-identical across groups
(verified by sha256 on 37 file pairs).

## Setup

Single rented RTX 4090 (24 GB), Ubuntu 24.04, CUDA driver 565.77.
Measured throughput 77,363 tokens/s, 49% MFU, stable to within 0.5% over 3.6 h.

Model: OLMo 150M — 162,398,208 parameters (137,822,208 non-embedding),
d_model 768, 12 layers, 12 heads, SwiGLU, RoPE, seq len 2048,
Llama-2 tokenizer (vocab 32,000).

wandb: https://wandb.ai/danielhuang-research/echo-chamber
