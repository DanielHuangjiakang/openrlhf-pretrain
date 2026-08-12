#!/usr/bin/env bash
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
gpuclean(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; }
for s in 116 58; do
  gpuclean
  d=workspace/checkpoints/OLMo-150M-tg30-grpo/ckpt/global_step${s}_hf
  echo "=== STEP $s  $(date -u +%H:%M) ==="
  timeout 3600 python inference/run_inference_all.py -c "$d" -t gsm8k -n 64 --no_greedy 2>&1 \
    | grep -aE "Pass Accuracy|Majority Accuracy|OutOfMemory|Traceback" | head -3
done
gpuclean
echo "=== ALL DONE $(date -u) ==="
