#!/usr/bin/env bash
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PY=/root/openrlhf-pretrain/workspace/venv/bin/python
S=/root/round2-status.txt; : > $S
note(){ echo "[$(date -u +%H:%M)] $*" | tee -a $S; }
gpuclean(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; ray stop --force >/dev/null 2>&1; sleep 8; }

for g in tg15 tg30; do
  gpuclean
  note "=== $g GRPO 2 episodes 开始 ==="
  EPISODES=2 bash scripts/run_grpo_single.sh \
      workspace/checkpoints/OLMo-150M-$g/latest-unsharded-hf \
      workspace/checkpoints/OLMo-150M-$g-grpo-2ep > /root/grpo-$g-2ep.log 2>&1
  note "=== $g GRPO 完成 ==="
  gpuclean
  # log-scale checkpoint 现在跨 232 步，取其中 7 个评测
  B=workspace/checkpoints/OLMo-150M-$g-grpo-2ep/ckpt
  for d in $(ls -d $B/global_step*_hf 2>/dev/null | sed -E 's/.*global_step([0-9]+)_hf/\1/' | sort -n | awk 'NR%2==1' | tail -7); do
    p=$B/global_step${d}_hf; [ -d "$p" ] || continue
    for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
      [ -f "$p/$f" ] || cp workspace/checkpoints/OLMo-150M-$g/latest-unsharded-hf/$f "$p/" 2>/dev/null
    done
    gpuclean; note "  $g-2ep step $d"
    timeout 2400 $PY inference/run_inference_all.py -c "$p" -t gsm8k --no_multiple >>/root/eval-$g-2ep.log 2>&1
  done
done
gpuclean
note "=== 全部完成 ==="
