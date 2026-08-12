#!/usr/bin/env bash
# 整夜接力。刻意不用 set -e：某一阶段失败也要继续跑后面的，
# 无人值守时"跑完能跑的"比"第一个错就全停"有价值得多。
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PY=/root/openrlhf-pretrain/workspace/venv/bin/python
S=/root/overnight-status.txt; : > $S
note(){ echo "[$(date -u +%H:%M)] $*" | tee -a $S; }

# vLLM 用 spawn 起的 worker 在父进程被杀后会变孤儿，ps 查不到但仍占 22GB 显存，
# 下一阶段直接 OOM。必须按 GPU 上的 pid 清。
gpuclean(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; ray stop --force >/dev/null 2>&1; sleep 8; }

note "=== 1/5 等 tg00 GRPO 结束 ==="
while pgrep -f "train_ppo_ray" >/dev/null; do sleep 60; done
note "tg00 GRPO 完成"
gpuclean

note "=== 2/5 tg00 RL checkpoint 评测（4 个点）==="
B=workspace/checkpoints/OLMo-150M-tg00-grpo/ckpt
for s in 1 20 58 116; do
  d=$B/global_step${s}_hf; [ -d "$d" ] || continue
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    [ -f "$d/$f" ] || cp workspace/checkpoints/OLMo-150M-tg00/latest-unsharded-hf/$f "$d/" 2>/dev/null
  done
  gpuclean
  note "  tg00-rl step $s"
  timeout 2400 $PY inference/run_inference_all.py -c "$d" -t gsm8k --no_multiple >>/root/eval-tg00-rl.log 2>&1
done
gpuclean

note "=== 3/5 tg15 预训练（1B tokens, ~3h37m）==="
rm -rf workspace/checkpoints/OLMo-150M-tg15
workspace/venv/bin/torchrun --nproc_per_node=1 OLMo/scripts/train.py \
    pretraining/configs/RL-150M-tg15.yaml > /root/train-tg15.log 2>&1
note "tg15 预训练完成"
gpuclean

note "=== 4/5 tg15 转换 + 评测 ==="
bash pretraining/scripts/convert_to_hf.sh workspace/checkpoints/OLMo-150M-tg15/latest-unsharded >>/root/train-tg15.log 2>&1
C=workspace/checkpoints/OLMo-150M-tg15/latest-unsharded-hf
gpuclean; note "  tg15 pass@1"
timeout 2400 $PY inference/run_inference_all.py -c $C -t gsm8k --no_multiple >>/root/eval-tg15.log 2>&1
gpuclean; note "  tg15 pass@64"
timeout 5400 $PY inference/run_inference_all.py -c $C -t gsm8k -n 64 --no_greedy >>/root/eval-tg15.log 2>&1
gpuclean

note "=== 5/5 tg15 GRPO（~1h20m）==="
bash scripts/run_grpo_single.sh $C workspace/checkpoints/OLMo-150M-tg15-grpo > /root/grpo-tg15.log 2>&1
note "tg15 GRPO 完成"
gpuclean

note "=== 6/5 tg15 RL checkpoint 评测 ==="
B=workspace/checkpoints/OLMo-150M-tg15-grpo/ckpt
for s in 1 5 14 29 58 116; do
  d=$B/global_step${s}_hf; [ -d "$d" ] || continue
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    [ -f "$d/$f" ] || cp $C/$f "$d/" 2>/dev/null
  done
  gpuclean; note "  tg15-rl step $s"
  timeout 2400 $PY inference/run_inference_all.py -c "$d" -t gsm8k --no_multiple >>/root/eval-tg15-rl.log 2>&1
done
gpuclean
note "=== 全部完成 ==="
