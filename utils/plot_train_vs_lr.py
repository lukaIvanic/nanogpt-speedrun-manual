"""Plot train loss vs learning rate for comparing runs, split by stage."""

import re
import os
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
UTILS_DIR = os.path.join(os.path.dirname(__file__), "..", "utils")

# (filename, label, stage_boundaries)
# boundaries: [(start, end), ...] for stages w0, w1, w2, ext
RUNS = [
    ("regular_ce_1530_run.txt", "1530 ce", [(0, 497), (497, 993), (993, 1490), (1490, 1530)]),
    ("regular_ce_1060_run.txt", "1060 ce", [(0, 340), (340, 680), (680, 1020), (1020, 1060)]),
]

STAGE_NAMES = ["w0", "w1", "w2", "ext"]


def parse_log(filepath):
    """Extract (step, train_loss, lr) tuples from log file."""
    with open(filepath) as f:
        lines = f.readlines()

    data = []
    for line in lines:
        # Match train loss line: step:1/1530, ... train loss:151.5210,  lr:1.0
        match = re.search(r"step:(\d+)/\d+,.*train loss:([\d.]+),\s+lr:([\d.]+)", line)
        if match:
            step = int(match.group(1))
            train_loss = float(match.group(2))
            lr = float(match.group(3))
            data.append((step, train_loss, lr))

    return data


def filter_by_stage(data, start, end):
    """Filter data points where start <= step < end."""
    return [(step, train_loss, lr) for step, train_loss, lr in data if start < step < end]


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for filename, label, boundaries in RUNS:
    filepath = os.path.join(LOG_DIR, filename)
    data = parse_log(filepath)

    for idx, (start, end) in enumerate(boundaries):
        stage_data = filter_by_stage(data, start, end)
        if stage_data:
            lrs = [d[2] for d in stage_data]
            train_losses = [d[1] for d in stage_data]
            axes[idx].plot(lrs, train_losses, marker=".", markersize=2, linewidth=0.5, label=label, alpha=0.5)

for idx, name in enumerate(STAGE_NAMES):
    axes[idx].set_xlabel("Learning Rate", fontsize=10)
    axes[idx].set_ylabel("Train Loss", fontsize=10)
    axes[idx].set_title(f"Stage: {name}", fontsize=12)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Train Loss vs Learning Rate by Stage", fontsize=14)
plt.tight_layout()

out = os.path.join(UTILS_DIR, "train_vs_lr.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
