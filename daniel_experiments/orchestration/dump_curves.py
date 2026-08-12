import glob, json, os, re
runs = ["tg00-grpo", "tg15-grpo", "tg15-grpo-2ep", "tg30-grpo", "tg30-grpo-2ep"]
def pct(v):
    if v is None: return "      --"
    if isinstance(v, float) and v <= 1.0: return "%7.2f%%" % (v * 100)
    return "%8s" % v
for run in runs:
    b = "workspace/checkpoints/OLMo-150M-%s/ckpt" % run
    ds = sorted(glob.glob(b + "/global_step*_hf"),
                key=lambda p: int(re.search(r"step(\d+)_", p).group(1)))
    if not ds: continue
    print("\n### %s  (%d ckpt)" % (run, len(ds)))
    print("%6s %8s %8s %8s" % ("step", "pass@1", "code%", "text%"))
    for d in ds:
        s = int(re.search(r"step(\d+)_", d).group(1))
        f = os.path.join(d, "eval_gsm8k_1.json")
        try:
            m = json.loads(open(f).read().strip().split("\n")[-1])
        except Exception:
            print("%6d   (未测)" % s); continue
        print("%6d %s %s %s" % (s, pct(m.get("final_accuracy")),
                                pct(m.get("tinygsm-code_count")), pct(m.get("text_count"))))
