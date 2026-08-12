#!/usr/bin/env bash
cd /root/openrlhf-pretrain
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=workspace/venv/bin/python
for d in workspace/checkpoints/OLMo-150M-tg30/latest-unsharded-hf \
         workspace/reference/OLMo-150M-as_fm3_8xtg; do
  echo "=== START $d $(date -u +%H:%M:%S) ==="
  $PY inference/run_inference_all.py -c "$d" -t gsm8k -n 64 --no_greedy
  echo "=== END $d $(date -u +%H:%M:%S) ==="
done
