# exp2 logs

Empty until the GRPO runs start. Will hold:

- `grpo-tg30.log.gz` / `grpo-tg00.log.gz` — openrlhf training logs (reward, KL, response length per rollout step)
- `eval-metrics.json` — GSM8K test metrics for all 15 log-scale checkpoints per model
- `checkpoint-curve.md` — the metric-vs-step tables the plots are built from

Live curves will be at https://wandb.ai/danielhuang-research/echo-chamber-rl
(project `echo-chamber-rl`, group `grpo-150m`).
