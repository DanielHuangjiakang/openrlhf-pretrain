#!/usr/bin/env bash
#
# GRPO fine-tuning of one pretrained checkpoint on a single GPU.
#
#   bash scripts/run_grpo_single.sh <hf-checkpoint-dir> [output-dir]
#
# The repo ships scripts/ppo_sweep.py + run_ppo_sweep_1_gpu.sh for this, but
# they are built around SLURM array jobs: they read $SLURM_ARRAY_TASK_ID to pick
# one point from a hyperparameter grid, hand-allocate a dozen Ray ports to keep
# concurrent array tasks from colliding, and submit through `ray job submit`
# with a working_dir of ./openrlhf_work_dir -- a directory that does not exist
# in the repository. None of that applies to one run on one box.
#
# Ray is left to auto-initialise. openrlhf never calls ray.init(), so the first
# Ray API call starts a local cluster on its own, which is all a single GPU
# needs and avoids the job-submission path entirely.
set -euo pipefail

PRETRAIN="${1:?usage: run_grpo_single.sh <hf-checkpoint-dir> [output-dir]}"
OUT="${2:-${PRETRAIN%/}-grpo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"

# --- knobs worth revisiting -------------------------------------------------
EPISODES="${EPISODES:-3}"          # passes over GSM8K train (7473 prompts)
KL_COEF="${KL_COEF:-1e-3}"         # middle of the paper's {0, 1e-3, 1e-2} sweep
N_SAMPLES="${N_SAMPLES:-8}"        # generations per prompt; see the note below
ROLLOUT_BS="${ROLLOUT_BS:-64}"     # prompts per rollout -> 116 rollout steps/episode
VLLM_MEM="${VLLM_MEM:-0.4}"        # ~9.9GB of KV cache, ~131 sequences at 2048 tokens

# n_samples_per_prompt is the single most important knob if learning stalls.
# GRPO normalises advantage within each group, so a group whose generations all
# score the same contributes no gradient at all. At this model's measured 6.4%
# pass@1, 1 - 0.936^8 = 41% of groups carry signal, which is healthy. If pass@1
# were nearer 1%, raising N_SAMPLES to 32 would roughly triple the useful
# fraction -- far cheaper than pretraining for longer.

mkdir -p "$OUT/ckpt"
echo "pretrain : $PRETRAIN"
echo "output   : $OUT"
echo "episodes : $EPISODES   kl: $KL_COEF   n_samples: $N_SAMPLES"

# Ray keeps state in /tmp across runs and a stale cluster silently reuses the
# previous run's GPU reservations.
ray stop --force >/dev/null 2>&1 || true

exec "$PY" -m openrlhf.cli.train_ppo_ray \
    --pretrain "$PRETRAIN" \
    --save_path "$OUT" \
    --ckpt_path "$OUT/ckpt" \
    \
    `# Reward is a rule, not a model: math_verify parses the answer and, for` \
    `# TinyGSM-style output, exec()s the generated function. Must be absolute --` \
    `# openrlhf imports it by path from inside a Ray worker.` \
    --remote_rm_url "$HERE/openrlhf/utils/math_verifier.py" \
    \
    --prompt_data openai/gsm8k --input_key question --label_key answer \
    `# No --apply_chat_template: these are base models with no chat template,` \
    `# and the eval script prompts them with the bare question too.` \
    \
    --advantage_estimator group_norm --use_kl_loss --kl_estimator k3 \
    --init_kl_coef "$KL_COEF" \
    `# group_norm makes train_ppo_ray drop the critic entirely (see its` \
    `# args.critic_pretrain = None), so only actor + ref + vLLM share the GPU.` \
    \
    --colocate_all_models --enable_prefix_caching --vllm_enable_sleep \
    --vllm_num_engines 1 --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --actor_num_nodes 1 --actor_num_gpus_per_node 1 \
    --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
    \
    --num_episodes "$EPISODES" --max_epochs 1 --max_samples 5000000 \
    --rollout_batch_size "$ROLLOUT_BS" --n_samples_per_prompt "$N_SAMPLES" \
    --train_batch_size 64 --micro_train_batch_size 8 --micro_rollout_batch_size 16 \
    --prompt_max_len 1024 --generate_max_len 1024 --temperature 0.7 \
    --actor_learning_rate 1e-6 \
    \
    --zero_stage 3 --bf16 --flash_attn --gradient_checkpointing --adam_offload \
    --normalize_reward --packing_samples \
    \
    `# Checkpoints on a log-scale step grid, which is what the paper's figures` \
    `# plot against: the format shift happens inside the first epoch, so evenly` \
    `# spaced checkpoints would miss it entirely.` \
    --save_log_scale_count 15 --save_hf_ckpt --disable_ds_ckpt \
    \
    --use_wandb true --wandb_project echo-chamber-rl \
    --wandb_group "grpo-150m" --wandb_prefix "$(basename "$PRETRAIN")" \
    "${@:3}"
