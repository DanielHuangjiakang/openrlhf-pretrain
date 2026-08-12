#!/usr/bin/env bash
# 补齐 tg00-grpo 曲线剩下的 9 个 checkpoint。fill-curves.sh 的循环里漏了 tg00。
# 幂等：已有 final_accuracy 的直接跳过。
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/root/openrlhf-pretrain/workspace/venv/bin/python
S=/root/fill-tg00-status.txt; : > $S
note(){ echo "[$(date -u +%H:%M)] $*" | tee -a $S; }
gpuclean(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 6; }

B=workspace/checkpoints/OLMo-150M-tg00-grpo/ckpt
base=workspace/checkpoints/OLMo-150M-tg00/latest-unsharded-hf
todo=""
for d in $(ls -d $B/global_step*_hf 2>/dev/null | sed -E "s|.*global_step([0-9]+)_hf|\1|" | sort -n); do
  tail -1 "$B/global_step${d}_hf/eval_gsm8k_1.json" 2>/dev/null | grep -q final_accuracy || todo="$todo $d"
done
note "tg00-grpo 需补:${todo:-（无）}"
for d in $todo; do
  p=$B/global_step${d}_hf
  for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
    [ -f "$p/$f" ] || cp $base/$f "$p/" 2>/dev/null
  done
  gpuclean
  note "  tg00-grpo step $d"
  # tg00 生成跑满 1024 token，单点约 4 分钟，给 40 分钟上限
  timeout 2400 $PY inference/run_inference_all.py -c "$p" -t gsm8k --no_multiple >>/root/eval-fill-tg00.log 2>&1
done
gpuclean
note "=== tg00 补齐完成 ==="
