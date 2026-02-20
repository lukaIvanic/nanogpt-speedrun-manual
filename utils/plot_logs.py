"""Parse training logs from ./logs/*.txt and generate charts."""

import re
import glob
import os
import matplotlib.pyplot as plt


def parse_log(path):
    train_steps, train_losses = [], []
    val_steps, val_losses, val_times_s = [], [], []

    started = False
    with open(path) as f:
        for line in f:
            if "Resetting Model" in line:
                started = True
                continue
            if not started:
                continue

            # Val line: step:N/M val_loss:V train_time:Xms step_avg:Yms
            m = re.match(r"step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms", line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                val_times_s.append(int(m.group(3)) / 1000)
                continue

            # Train line: step:N/M, train_time:Xms, step_avg:Yms, train loss:Z,  lr:W
            m = re.match(r"step:(\d+)/\d+,.* train loss:([\d.]+)", line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))

    return train_steps, train_losses, val_steps, val_losses, val_times_s


def plot_run(run_id, train_steps, train_losses, val_steps, val_losses, val_times_s, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(run_id[:30], fontsize=14)

    axes[0].plot(train_steps, train_losses, linewidth=0.8, label="train loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Train Loss vs Step")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_steps, val_losses, marker="o", markersize=3, label="val loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Val Loss vs Step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(val_times_s, val_losses, marker="o", markersize=3, label="val loss")
    axes[2].set_xlabel("Train Time (s)")
    axes[2].set_ylabel("Val Loss")
    axes[2].set_title("Val Loss vs Time")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    txt_files = sorted(glob.glob(os.path.join(logs_dir, "*.txt")))

    all_runs = []
    for path in txt_files:
        run_id = os.path.splitext(os.path.basename(path))[0]
        train_steps, train_losses, val_steps, val_losses, val_times_s = parse_log(path)

        if not val_steps and not train_steps:
            continue

        all_runs.append((run_id, train_steps, train_losses, val_steps, val_losses, val_times_s))

        out_path = os.path.join(logs_dir, f"{run_id}.png")
        plot_run(run_id, train_steps, train_losses, val_steps, val_losses, val_times_s, out_path)

    # Combined plot
    if not all_runs:
        print("No runs with training data found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("All Runs", fontsize=14)

    for run_id, train_steps, train_losses, val_steps, val_losses, val_times_s in all_runs:
        label = run_id[:30]
        if train_steps:
            axes[0].plot(train_steps, train_losses, linewidth=0.8, label=label)
        if val_steps:
            axes[1].plot(val_steps, val_losses, marker="o", markersize=3, label=label)
            axes[2].plot(val_times_s, val_losses, marker="o", markersize=3, label=label)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Train Loss vs Step")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Val Loss vs Step")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Train Time (s)")
    axes[2].set_ylabel("Val Loss")
    axes[2].set_title("Val Loss vs Time")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(logs_dir, "all_runs.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
