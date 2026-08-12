# exp2 logs

Everything the numbers in [../RESULTS.md](../RESULTS.md) were computed from.

## Data

**`grpo-curves.csv`** — one row per evaluated checkpoint, 80 rows. This is the
file to plot from.

| column | |
|---|---|
| `run`, `group`, `episodes`, `step` | `step` is the rollout step; `episodes` is which episode it falls in, so the 3-episode `tg30` rows (step > 232) are separable |
| `final_accuracy`, `tinygsm-code_count`, `text_count` | greedy, n=1 |
| `pass64`, `maj64`, `code_count64`, `text_count64` | temperature 0.7, n=64 |

The n=64 columns are populated for 7 rows only — the pretrained baselines and
the run endpoints. They cost 22–36 min per checkpoint, so they were not measured
across the curve. `tg00` has none at all, deliberately: exp1 § tg00 explains why
the metric would not mean what its name says for that model.

Step 0 rows (`*-pretrained`) are exp1's checkpoints, included so every curve has
its pre-RL origin in the same file.

**`eval-metrics.json`** — the same 80 rows as JSON.

## Training logs

`grpo-<group>.log.gz` (1 episode), `grpo-<group>-2ep.log.gz` (2 episodes),
`grpo-tg30-ep3.log.gz` (episode 3, resumed from the 2-episode optimizer state).
These carry the per-rollout-step `reward`, `kl` and `response_length` that the
windowed means in RESULTS.md § Training-side signals are computed from — the
tqdm postfix, which needs `tr '\r' '\n'` before grepping.

Note `grpo-tg30-2ep.log.gz` and `grpo-tg30-ep3.log.gz` are two runs writing to
one checkpoint directory (`...-grpo-2ep`, which despite the name holds
3-episode results — renaming it would have broken resumption).

## Job logs

`*-status.txt` — timestamped stage markers for the unattended chains:
`round2` (2-episode runs), `fill`/`fill-tg00` (backfilled checkpoint
evaluations), `ep3` (the resumed third episode), `pass64` (endpoint n=64
evaluations, with per-checkpoint wall clock).

wandb: https://wandb.ai/danielhuang-research/echo-chamber-rl
(project `echo-chamber-rl`, group `grpo-150m`). Note each resumed run starts a
new wandb run whose x-axis restarts, so the CSV is the reliable source for
step-aligned curves.
