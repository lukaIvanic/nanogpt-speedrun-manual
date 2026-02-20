"""Plot the LR schedule over steps without needing CUDA/distributed."""

import os
import matplotlib.pyplot as plt
from itertools import accumulate
from dataclasses import dataclass


@dataclass
class Stage:
    duration: float
    lr_mul: float
    lr_floor: float = 0.15

# Mirror the stages from s07_schedule.py
STAGES = [
    Stage(duration=1/3, lr_mul=1.0, lr_floor=0.15),
    Stage(duration=1/3, lr_mul=1.52, lr_floor=0.15),
    Stage(duration=1/3, lr_mul=1.73, lr_floor=0.30),
    Stage(duration=None, lr_mul=1.0, lr_floor=0.15),  # extension
]

SCHEDULED_ITERATIONS = 1020
EXTENSION_ITERATIONS = 510
COOLDOWN_FRAC = 0.55

TOTAL_STEPS = SCHEDULED_ITERATIONS + EXTENSION_ITERATIONS
ends = [0] + [round(c * SCHEDULED_ITERATIONS) for c in accumulate(s.duration for s in STAGES[:-1])] + [TOTAL_STEPS]
boundaries = list(zip(ends, ends[1:]))


def lookup(step):
    for i, (start, end) in enumerate(boundaries):
        if step < end:
            return STAGES[i], i
    return STAGES[-1], len(STAGES) - 1


def get_lr_original(step):
    """Original schedule (used for stages 1 & 2, and to find stage 3 start LR)."""
    stage, _ = lookup(step)
    lr = stage.lr_mul
    cd_start = int(SCHEDULED_ITERATIONS * (1 - COOLDOWN_FRAC))
    if step >= cd_start:
        t = min(1.0, (step - cd_start) / (SCHEDULED_ITERATIONS - cd_start))
        lr = lr * (1 - t) + 0.15 * t
    return lr


# Precompute the LR at the start of stage 3 (using original schedule)
STAGE3_START_LR = get_lr_original(boundaries[2][0])
# And the LR at the start of extension (end of stage 3 with new floor)
STAGE3_END_LR = STAGES[2].lr_floor  # 0.30


def get_lr(step):
    stage, idx = lookup(step)
    # Stages 0 and 1: original behavior
    if idx <= 1:
        return get_lr_original(step)
    # Stage 2 (third stage): linear from original start LR to lr_floor
    if idx == 2:
        start, end = boundaries[2]
        t = (step - start) / (end - start)
        return STAGE3_START_LR * (1 - t) + STAGE3_END_LR * t
    # Extension: flat at floor
    return STAGES[3].lr_floor


steps = list(range(TOTAL_STEPS + 1))
lrs = [get_lr(s) for s in steps]

plt.figure(figsize=(10, 5))
plt.plot(steps, lrs, linewidth=1.2)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.title(f"LR Schedule (scheduled={SCHEDULED_ITERATIONS}, ext={EXTENSION_ITERATIONS}, cooldown={COOLDOWN_FRAC})", fontsize=13)
plt.grid(True, alpha=0.3)

# Mark stage boundaries
for start, end in boundaries[:-1]:
    plt.axvline(x=end, linestyle="--", linewidth=0.8, alpha=0.4, color="red")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "lr_schedule.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
