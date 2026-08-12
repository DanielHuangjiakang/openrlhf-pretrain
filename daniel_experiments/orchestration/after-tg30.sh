#!/usr/bin/env bash
set -x
cd /root/openrlhf-pretrain
export PY=/root/openrlhf-pretrain/workspace/venv/bin/python
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 1) 等训练进程退出
while pgrep -f "train.py pretraining/configs/RL-150M-tg30" >/dev/null; do sleep 30; done
echo "=== TRAINING FINISHED $(date -u) ==="

# 2) 转 HF
bash pretraining/scripts/convert_to_hf.sh workspace/checkpoints/OLMo-150M-tg30/latest-unsharded
echo "=== CONVERTED $(date -u) ==="

# 3) 我们的 tg30，只跑 pass@1（1319 条，约 1 分钟）
$PY inference/run_inference_all.py \
    -c workspace/checkpoints/OLMo-150M-tg30/latest-unsharded-hf -t gsm8k --no_multiple
echo "=== OURS tg30 DONE $(date -u) ==="

# 4) 作者的 8xtg（28.4% TinyGSM, 74.9B tokens）作为参照
$PY inference/run_inference_all.py \
    -c workspace/reference/OLMo-150M-as_fm3_8xtg -t gsm8k --no_multiple
echo "=== REFERENCE 8xtg DONE $(date -u) ==="
