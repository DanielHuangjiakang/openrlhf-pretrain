# 用仓库里真实的调度器构造，算出续跑接缝处的学习率跳变。
import torch
from transformers.trainer import get_scheduler

PEAK, MINR, WARM = 1e-6, 0.03, 0.1     # actor_lr, lr_warmup_ratio, min_lr = 0.1*peak
PER_EP = 7473 * 8 // 64                # num_update_steps_per_episodes

def lr_at(num_episodes, last_epoch):
    import math
    max_steps = math.ceil(num_episodes * PER_EP)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=PEAK)
    s = get_scheduler("cosine_with_min_lr", opt,
                      num_warmup_steps=math.ceil(max_steps * MINR),
                      num_training_steps=max_steps,
                      scheduler_specific_kwargs={"min_lr": PEAK * WARM})
    st = s.state_dict(); st["last_epoch"] = last_epoch; s.load_state_dict(st)
    opt.step(); s.step()                # advance once so get_last_lr reflects last_epoch
    return max_steps, s.get_last_lr()[0]

print("per-episode optimizer steps :", PER_EP)
m2, _ = lr_at(2, 0)
m3, _ = lr_at(3, 0)
print("max_steps  2ep / 3ep        :", m2, "/", m3)
print()
print("%-46s %s" % ("2ep 跑到终点 (last_epoch=1868, total=1868)", "%.3e" % lr_at(2, m2)[1]))
print("%-46s %s" % ("以 3ep 续跑, 恢复 last_epoch=1868",        "%.3e" % lr_at(3, m2)[1]))
print("%-46s %s" % ("从头跑 3ep, 走到 step 1868",               "%.3e" % lr_at(3, m2)[1]))
print("%-46s %s" % ("3ep 终点 (step 2802)",                     "%.3e" % lr_at(3, m3)[1]))
