# exp2 — Does RL amplify the pretrained format preference?

**Status: planned, not started.** Waiting on exp1's `tg00` to finish.

## Question

exp1 establishes what format each pretrained model prefers. This asks the
paper's actual claim: **does RL push that preference further, or does it just
overwrite the model?**

The distinction matters for how the result reads. "RL amplifies what pretraining
taught" and "RL retrains the model" predict the same accuracy curve but very
different `tinygsm-code_count` and KL curves.

## Design

GRPO on the two extremes from exp1 — `tg00` (0% TinyGSM) and `tg30` (30%).
`tg15` is skipped for the same reason it was skipped in exp1: if the extremes do
not separate, the midpoint will not.

| | |
|---|---|
| Algorithm | GRPO — `--advantage_estimator group_norm --use_kl_loss --kl_estimator k3` |
| Prompts | GSM8K **train**, 7,473 problems (eval stays on the 1,319-problem test split) |
| Reward | rule-based: `openrlhf/utils/math_verifier.py` parses the answer, `exec()`s TinyGSM-style code, scores 0/1. No reward model, no preference data. |
| KL coefficient | 1e-3 (middle of the paper's {0, 1e-3, 1e-2} sweep) |
| Samples per prompt | 8 |
| Rollout batch | 64 prompts → 512 generations per rollout |
| Episodes | **1** (see below) |
| Checkpoints | 15, on a log-scale step grid |

Launch: `bash scripts/run_grpo_single.sh <hf-checkpoint>`

### Why one episode and not the paper's ten

openrlhf derives optimizer steps as
`len(prompts) * n_samples / train_batch_size` (`ppo_actor.py:838`):

```
7,473 x 8 / 64 = 934 optimizer steps per episode
```

against **1,907** pretraining steps for a 1B-token run.

| | RL steps | vs. pretraining steps |
|---|---|---|
| 1 episode | 934 | 49% |
| 3 episodes | 2,802 | **147%** |
| 10 episodes (paper) | 9,340 | **490%** |
| paper's own ratio | 9,340 | ~17% (their pretraining is ~53,600 steps at 56B tokens) |

At ten episodes RL would do five times as many updates as pretraining did.
"RL amplifies what pretraining taught" is not a defensible reading of that.

Two things cut the other way: the learning rate is 1e-6 against pretraining's
1e-3, so one episode carries ~0.05% of pretraining's lr-weighted parameter
movement, and the KL loss anchors the policy to the pretrained reference. So the
step ratio overstates the risk — but the way to settle it is the KL curve, not
arithmetic.

The paper also places the effect inside the first epoch: *"The model quickly
shifts toward generating answers in the format of one distribution — TinyGSM in
this case — within the first epoch."* One episode should be enough to see it.

### Checkpoints on a log grid, not an even one

`--save_log_scale_count 15` places checkpoints at
`np.logspace(-2.1, 0, 15) * total_steps` (`ppo_actor.py:184`). The format shift
happens early and fast; evenly spaced checkpoints would miss it entirely. This
is also why the paper's figures use a log x-axis.

One episode therefore yields a 15-point curve, not a single endpoint. If
`tinygsm-code_count` is still moving at the last point, extend; if it plateaus
early, one episode was more than enough.

## What to measure

Evaluate all 15 checkpoints on GSM8K test:

| metric | reading |
|---|---|
| **`tinygsm-code_count` vs step** | **the core result.** Rising = RL amplifies the pretrained preference |
| pass@1 vs step | RL's headline gain |
| pass@64 vs step | flat or **falling** would mean the distribution is narrowing, not improving |
| `pass@64 − pass@1` | the gap RL is closing |
| `kl` vs step | small and stable = amplification; growing = the pretrained behaviour is being overwritten |
| `reward` vs step | flat at 0 means the run is broken, not that the model cannot learn |

## Predictions

Written before running, so they can be wrong.

1. **`tg30`'s `tinygsm-code_count` rises from 97.12% toward ~100%.** Little room
   left; the effect may be visible mainly as the *sampled* count (94.63%)
   converging on the greedy one.
2. **`tg00`'s behaviour is the interesting case.** With no TinyGSM in
   pretraining, if RL still drives it toward code format, the source is
   Algebraic-Stack rather than TinyGSM and the paper's story needs qualifying.
   If it converges on some other single format, that supports the more general
   claim — RL collapses onto *one* distribution, whichever the pretraining
   favoured.
3. **pass@1 rises substantially for `tg30`** — there are 44.1 points between
   pass@1 and pass@64 to convert.
4. **pass@64 stays flat or falls.** RL sharpens rather than extends. If pass@64
   climbs meaningfully, RL is doing something more than redistribution.

## Risks

| risk | signal | response |
|---|---|---|
| Reward stays 0 | `reward` flat at 0 from step 1 | `--remote_rm_url` must be absolute; a Ray worker imports it by path |
| Too few informative groups | reward moves but pass@1 does not | raise `--n_samples_per_prompt` 8 → 32; at 6.44% pass@1, 41% of groups carry signal, which should be adequate |
| Policy drifts off | `kl` growing without bound | raise `--init_kl_coef` to 1e-2, the top of the paper's sweep |
| OOM | crash at startup | actor + ref + vLLM share one 24GB card; `--vllm_gpu_memory_utilization 0.4` leaves ~9.9GB of KV cache. GRPO needs no critic (`train_ppo_ray.py:461` sets `critic_pretrain = None` for non-`gae` estimators) |

The first minutes are worth watching by hand rather than leaving unattended:
confirm rollouts are non-empty, reward is not identically zero, and KL is finite.

## Estimated cost

~1h20m per model, two models, on the same rented 4090. About $2.
