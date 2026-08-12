import re
pat = re.compile(r"Episode \[(\d+)/(\d+)\][^|]*\|[^|]*\|\s*(\d+)/(\d+).*?kl=([\d.e+-]+), reward=([\d.e+-]+), response_length=([\d.]+)")
def stats(v): 
    m = sum(v)/len(v); sd = (sum((x-m)**2 for x in v)/len(v))**.5
    return m, sd
print("%-10s %-22s %-22s %-16s" % ("run", "reward 前20 / 后20", "kl 前20 / 后20", "resp_len 前/后"))
for tag, path in [("tg00-1ep","/root/grpo-tg00.log"), ("tg15-1ep","/root/grpo-tg15.log"),
                  ("tg15-2ep","/root/grpo-tg15-2ep.log"), ("tg30-1ep","/root/grpo-tg30.log"),
                  ("tg30-2ep","/root/grpo-tg30-2ep.log")]:
    try: txt = open(path, errors="replace").read().replace("\r","\n")
    except Exception: print("%-10s (无日志)" % tag); continue
    rows = [(float(m[5]), float(m[6]), float(m[7])) for m in pat.finditer(txt)]
    if len(rows) < 40: print("%-10s (样本不足 %d)" % (tag, len(rows))); continue
    a, b = rows[:20], rows[-20:]
    rm0,rs0 = stats([r[1] for r in a]); rm1,rs1 = stats([r[1] for r in b])
    km0,_   = stats([r[0] for r in a]); km1,_   = stats([r[0] for r in b])
    lm0,_   = stats([r[2] for r in a]); lm1,_   = stats([r[2] for r in b])
    print("%-10s %.4f±%.3f -> %.4f±%.3f   %.5f -> %.5f   %.0f -> %.0f" %
          (tag, rm0, rs0, rm1, rs1, km0, km1, lm0, lm1))
