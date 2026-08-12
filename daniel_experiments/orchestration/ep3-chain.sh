#!/usr/bin/env bash
# tg30 第 3 轮 GRPO —— 从 2 轮的 DeepSpeed 存档续跑，不是从头。
#
#   consumed_samples = 232*64 = 14848  ->  steps 233, start_episode 2
#   range(2, 3) => 只跑 1 轮 116 个 rollout step，约 1h40m
#
# 输出目录必须沿用 -grpo-2ep，否则 --ckpt_path 找不到 _actor（目录名会与内容
# 不符，这是续跑的代价）。--save_steps 20 让第 3 轮多存几个点：log-scale 网格
# 在 348 步的尾部只有 246 和 348 两个落点，太稀。
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export HF_TOKEN=$(sed -n 's/^export HF_TOKEN=//p' /root/round2.sh | head -1)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PY=/root/openrlhf-pretrain/workspace/venv/bin/python

S=/root/ep3-status.txt; : > $S
note(){ echo "[$(date -u +%H:%M)] $*" | tee -a $S; }
gpuclean(){
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
  ray stop --force >/dev/null 2>&1
  sleep 8
}

note "等 tg00 补齐结束..."
while pgrep -f fill-tg00.sh >/dev/null; do sleep 60; done
note "tg00 已结束，开始第 3 轮"

gpuclean
note "=== tg30 第 3 轮 GRPO 开始（续跑 233 -> 348）==="
EPISODES=3 LOAD_CKPT=1 bash scripts/run_grpo_single.sh \
    workspace/checkpoints/OLMo-150M-tg30/latest-unsharded-hf \
    workspace/checkpoints/OLMo-150M-tg30-grpo-2ep \
    --save_steps 20 > /root/grpo-tg30-ep3.log 2>&1
note "=== tg30 第 3 轮训练完成 ==="

# 幂等：已有 final_accuracy 的跳过，所以只会测第 3 轮新产生的点。
gpuclean
B=workspace/checkpoints/OLMo-150M-tg30-grpo-2ep/ckpt
base=workspace/checkpoints/OLMo-150M-tg30/latest-unsharded-hf
todo=""
for d in $(ls -d $B/global_step*_hf 2>/dev/null | sed -E "s|.*global_step([0-9]+)_hf|\1|" | sort -n); do
  tail -1 "$B/global_step${d}_hf/eval_gsm8k_1.json" 2>/dev/null | grep -q final_accuracy || todo="$todo $d"
done
note "需测评:${todo:-（无）}"
for d in $todo; do
  p=$B/global_step${d}_hf
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    [ -f "$p/$f" ] || cp $base/$f "$p/" 2>/dev/null
  done
  gpuclean
  note "  tg30-ep3 step $d"
  timeout 2400 $PY inference/run_inference_all.py -c "$p" -t gsm8k --no_multiple >>/root/eval-tg30-ep3.log 2>&1
done
gpuclean
note "=== 全部完成 ==="
