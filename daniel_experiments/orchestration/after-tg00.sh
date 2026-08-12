#!/usr/bin/env bash
set -x
cd /root/openrlhf-pretrain
export PY=/root/openrlhf-pretrain/workspace/venv/bin/python
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

while pgrep -f "train.py pretraining/configs/RL-150M-tg00" >/dev/null; do sleep 30; done
echo "=== TG00 TRAINING FINISHED $(date -u) ==="

bash pretraining/scripts/convert_to_hf.sh workspace/checkpoints/OLMo-150M-tg00/latest-unsharded
echo "=== CONVERTED $(date -u) ==="

CKPT=workspace/checkpoints/OLMo-150M-tg00/latest-unsharded-hf
$PY inference/run_inference_all.py -c $CKPT -t gsm8k --no_multiple
echo "=== TG00 pass@1 DONE $(date -u) ==="

$PY inference/run_inference_all.py -c $CKPT -t gsm8k -n 64 --no_greedy
echo "=== TG00 pass@64 DONE $(date -u) ==="
