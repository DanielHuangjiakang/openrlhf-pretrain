#!/usr/bin/env bash
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
gpuclean(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; }
B=workspace/checkpoints/OLMo-150M-tg30-grpo/ckpt
for s in 1 2 3 5 7 10 14 20 29 41 58 82 116; do
  d=$B/global_step${s}_hf
  [ -d "$d" ] || continue
  # RL 的 checkpoint 只存了权重，tokenizer 要从基座补过来
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    [ -f "$d/$f" ] || cp workspace/checkpoints/OLMo-150M-tg30/latest-unsharded-hf/$f "$d/" 2>/dev/null
  done
  gpuclean
  echo "=== STEP $s ==="
  timeout 600 python inference/run_inference_all.py -c "$d" -t gsm8k --no_multiple 2>&1 \
    | grep -aE "Greedy Accuracy|OutOfMemory|Traceback" | head -2
done
gpuclean
echo "=== ALL DONE $(date -u) ==="
