#!/usr/bin/env bash
# 补 pass@64 / majority@64 —— 只补缺的那几个点，不重跑任何已有测评。
#
#   --no_greedy    跳过整个 greedy 分支，包括它的 open(..., "w")，
#                  所以已有的 eval_gsm8k_1.json 不会被碰。
#   --max_num_seqs vLLM 默认只跑 256 路并发；1319x64 = 84,416 条生成时
#                  这是瓶颈。150M 模型每 token 36KB KV，21GB 够 ~1900 条。
#
# 幂等：eval_gsm8k_64.json 里已有 final_maj_accuracy 的直接跳过。
# 顺序按信息量排：tg30 的剂量曲线（2ep -> 3ep）优先，tg15 垫后，
# 这样中途叫停也拿得到最关键的两点。
cd /root/openrlhf-pretrain
source workspace/venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn
PY=/root/openrlhf-pretrain/workspace/venv/bin/python

S=/root/pass64-status.txt; : > $S
note(){ echo "[$(date -u +%H:%M)] $*" | tee -a $S; }
gpuclean(){
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
  sleep 6
}

note "等第 3 轮那条链结束..."
while pgrep -f ep3-chain.sh >/dev/null; do sleep 60; done
note "开始补 pass@64"

TARGETS="
OLMo-150M-tg30-grpo-2ep/ckpt/global_step232_hf|tg30@232(2ep 终点)
OLMo-150M-tg30-grpo-2ep/ckpt/global_step348_hf|tg30@348(3ep 终点)
OLMo-150M-tg15-grpo-2ep/ckpt/global_step232_hf|tg15@232(2ep 终点)
"

for entry in $TARGETS; do
  p="workspace/checkpoints/${entry%%|*}"
  label="${entry##*|}"
  [ -d "$p" ] || { note "跳过 $label —— 目录不存在"; continue; }
  if tail -1 "$p/eval_gsm8k_64.json" 2>/dev/null | grep -q final_maj_accuracy; then
    note "跳过 $label —— 已有 n=64 结果"; continue
  fi
  gpuclean
  note "  $label 开始"
  t0=$(date +%s)
  timeout 5400 $PY inference/run_inference_all.py \
      -c "$p" -t gsm8k --no_greedy --max_num_seqs 1024 >>/root/eval-pass64.log 2>&1
  rc=$?
  note "  $label 结束 rc=$rc 用时 $(( ($(date +%s) - t0) / 60 )) 分钟"
done

gpuclean
note "=== pass@64 补齐完成 ==="
