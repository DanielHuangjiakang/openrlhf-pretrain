# exp1 logs

| file | what |
|---|---|
| `train-tg30.log.gz` | full OLMo training log for tg30 (config dump stripped): per-step loss, throughput, held-out evals, checkpointing |
| `train-tg00.log.gz` | same for tg00 |
| `eval-metrics.json` | the metrics line from every `eval_gsm8k_*.json`, keyed by model/eval |
| `mixture-manifest.json` | `build_mixture.py` output: exact per-file block allocation for all three groups |
| `mixture-selfcheck.txt` | the alignment / share / sha256-nesting / holdout checks printed at mixture build time |
| `sample-generations.txt` | one generation per response type per model, for reading what `tinygsm-code` vs `text` actually look like |

Not included: the raw `eval_gsm8k_64.json` files, 75-82 MB each, which hold all
84,416 generations per model. They stay on the training box at
`workspace/checkpoints/<model>/latest-unsharded-hf/` and
`workspace/reference/<model>/`. Everything summarised in RESULTS.md is derived
from the metrics line captured in `eval-metrics.json`.

Live curves: https://wandb.ai/danielhuang-research/echo-chamber
