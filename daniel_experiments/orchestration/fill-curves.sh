#!/usr/bin/env bash
# 把四条 RL 曲线的缺失 checkpoint 补齐。幂等：已有 metrics 的直接跳过，
# 所以中断后重跑不会浪费时间。
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export HF_TOKEN=${HF_TOKEN:?export HF_TOKEN=... before running}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/root/openrlhf-pretrain/workspace/venv/bin/python
S=/root/fill-status.txt; : > $S
note(){ echo "[$(date -u +%H:%M)] $*" | tee -a $S; }
gpuclean(){ for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 6; }

note "等 round2 结束..."
while pgrep -f round2.sh >/dev/null; do sleep 60; done
note "round2 已结束，开始补齐"

for run in tg15-grpo tg15-grpo-2ep tg30-grpo tg30-grpo-2ep; do
  g=${run%%-*}
  B=workspace/checkpoints/OLMo-150M-$run/ckpt
  [ -d "$B" ] || { note "跳过 $run（不存在）"; continue; }
  base=workspace/checkpoints/OLMo-150M-$g/latest-unsharded-hf
  todo=""
  for d in $(ls -d $B/global_step*_hf 2>/dev/null | sed -E "s|.*global_step([0-9]+)_hf|\1|" | sort -n); do
    p=$B/global_step${d}_hf
    tail -1 "$p/eval_gsm8k_1.json" 2>/dev/null | grep -q final_accuracy || todo="$todo $d"
  done
  note "$run 需补: ${todo:-（无）}"
  for d in $todo; do
    p=$B/global_step${d}_hf
    for f in tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer.model; do
      [ -f "$p/$f" ] || cp $base/$f "$p/" 2>/dev/null
    done
    gpuclean
    note "  $run step $d"
    timeout 2400 $PY inference/run_inference_all.py -c "$p" -t gsm8k --no_multiple >>/root/eval-fill.log 2>&1
  done
done
gpuclean
note "=== 补齐完成 ==="
