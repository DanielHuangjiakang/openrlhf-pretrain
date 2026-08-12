import re, sys
pat = re.compile(r"Episode \[(\d+)/(\d+)\][^|]*\|[^|]*\|\s*(\d+)/(\d+).*?kl=([\d.e+-]+), reward=([\d.e+-]+), response_length=([\d.]+)")
for tag, path in [("tg15-1ep","/root/grpo-tg15.log"), ("tg15-2ep","/root/grpo-tg15-2ep.log"),
                  ("tg30-1ep","/root/grpo-tg30.log"), ("tg30-2ep","/root/grpo-tg30-2ep.log")]:
    try: txt = open(path, errors="replace").read().replace("\r", "\n")
    except Exception: print("%s  (无日志)" % tag); continue
    rows = []
    for m in pat.finditer(txt):
        ep, eptot, d, t = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        rows.append(((ep-1)*t + d, float(m[5]), float(m[6]), float(m[7])))
    if not rows: print("%s  (无匹配)" % tag); continue
    print("\n### %s   (%d 步)" % (tag, rows[-1][0]))
    print("%6s %10s %9s %8s" % ("step", "kl", "reward", "resp_len"))
    n = len(rows)
    idx = sorted(set([0] + [int(n*f) for f in (.125,.25,.375,.5,.625,.75,.875)] + [n-1]))
    for i in idx:
        s, kl, r, rl = rows[i]
        print("%6d %10.5f %9.4f %8.0f" % (s, kl, r, rl))
