"""Plot validation loss vs learning rate for comparing runs, split by stage."""

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
    """Extract (step, val_loss, lr) tuples from log file."""
    with open(filepath) as f:
        lines = f.readlines()

    data = []
    for i, line in enumerate(lines):
        # Match val_loss line: step:50/1530 val_loss:5.7883 ...
        match = re.search(r"step:(\d+)/\d+\s+val_loss:([\d.]+)", line)
        if match:
            step = int(match.group(1))
            val_loss = float(match.group(2))

            # Get lr from next line (step+1)
            if i + 1 < len(lines):
                lr_match = re.search(r"lr:([\d.]+)", lines[i + 1])
                if lr_match:
                    lr = float(lr_match.group(1))
                    data.append((step, val_loss, lr))

    return data


def filter_by_stage(data, start, end):
    """Filter data points where start <= step < end."""
    return [(step, val_loss, lr) for step, val_loss, lr in data if start <= step < end]


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for filename, label, boundaries in RUNS:
    filepath = os.path.join(LOG_DIR, filename)
    data = parse_log(filepath)

    for idx, (start, end) in enumerate(boundaries):
        stage_data = filter_by_stage(data, start, end)
        if stage_data:
            lrs = [d[2] for d in stage_data]
            val_losses = [d[1] for d in stage_data]
            axes[idx].plot(lrs, val_losses, marker="o", markersize=4, linewidth=1, label=label, alpha=0.7)

for idx, name in enumerate(STAGE_NAMES):
    axes[idx].set_xlabel("Learning Rate", fontsize=10)
    axes[idx].set_ylabel("Validation Loss", fontsize=10)
    axes[idx].set_title(f"Stage: {name}", fontsize=12)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Validation Loss vs Learning Rate by Stage", fontsize=14)
plt.tight_layout()

out = os.path.join(UTILS_DIR, "val_vs_lr.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
