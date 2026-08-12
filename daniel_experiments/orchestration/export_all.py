import json, glob, os, re, csv, sys
OUT = "/root/export"
os.makedirs(OUT, exist_ok=True)
RUNS = ["tg00-grpo", "tg15-grpo", "tg15-grpo-2ep", "tg30-grpo", "tg30-grpo-2ep"]
BASES = {"tg00": "tg00", "tg15": "tg15", "tg30": "tg30"}

def load(p):
    try: return json.loads(open(p).read().strip().split("\n")[-1])
    except Exception: return None

rows = []
for g, d in BASES.items():                       # pretrained baselines, step 0
    b = "workspace/checkpoints/OLMo-150M-%s/latest-unsharded-hf" % d
    m1, m64 = load(b + "/eval_gsm8k_1.json"), load(b + "/eval_gsm8k_64.json")
    rows.append(dict(run=g + "-pretrained", group=g, episodes=0, step=0,
                     **{k: (m1 or {}).get(k) for k in ("final_accuracy", "tinygsm-code_count", "text_count")},
                     pass64=(m64 or {}).get("final_accuracy"), maj64=(m64 or {}).get("final_maj_accuracy"),
                     code_count64=(m64 or {}).get("tinygsm-code_count"), text_count64=(m64 or {}).get("text_count")))

for run in RUNS:
    g = run.split("-")[0]
    ep = 2 if run.endswith("-2ep") else 1
    b = "workspace/checkpoints/OLMo-150M-%s/ckpt" % run
    for p in sorted(glob.glob(b + "/global_step*_hf"), key=lambda x: int(re.search(r"step(\d+)_", x).group(1))):
        s = int(re.search(r"step(\d+)_", p).group(1))
        m1, m64 = load(p + "/eval_gsm8k_1.json"), load(p + "/eval_gsm8k_64.json")
        if not m1 and not m64: continue
        rows.append(dict(run=run, group=g, episodes=(3 if s > 232 else ep), step=s,
                         **{k: (m1 or {}).get(k) for k in ("final_accuracy", "tinygsm-code_count", "text_count")},
                         pass64=(m64 or {}).get("final_accuracy"), maj64=(m64 or {}).get("final_maj_accuracy"),
                         code_count64=(m64 or {}).get("tinygsm-code_count"), text_count64=(m64 or {}).get("text_count")))

cols = ["run","group","episodes","step","final_accuracy","tinygsm-code_count","text_count",
        "pass64","maj64","code_count64","text_count64"]
with open(OUT + "/grpo-curves.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c: r.get(c) for c in cols})
json.dump(rows, open(OUT + "/eval-metrics.json", "w"), indent=1)
print("rows:", len(rows), "| with pass@64:", sum(1 for r in rows if r.get("pass64") is not None))
