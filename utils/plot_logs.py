"""Parse training logs from ./logs/*.txt and generate charts."""

import re
import glob
import os
import argparse
import matplotlib.pyplot as plt


def parse_log(path):
    train_steps, train_losses, train_lrs, train_times_s = [], [], [], []
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
            m = re.match(r"step:(\d+)/\d+, train_time:(\d+)ms,.* train loss:([\d.]+),\s+lr:([\d.]+)", line)
            if m:
                train_steps.append(int(m.group(1)))
                train_times_s.append(int(m.group(2)) / 1000)
                train_losses.append(float(m.group(3)))
                train_lrs.append(float(m.group(4)))
                continue

            # Train line without train loss: step:N/M, train_time:Xms, step_avg:Yms, lr:W
            m = re.match(r"step:(\d+)/\d+, train_time:(\d+)ms,.*lr:([\d.]+)", line)
            if m:
                train_steps.append(int(m.group(1)))
                train_times_s.append(int(m.group(2)) / 1000)
                train_losses.append(None)
                train_lrs.append(float(m.group(3)))

    # Filter out None losses
    filtered_steps = [s for s, l in zip(train_steps, train_losses) if l is not None]
    filtered_losses = [l for l in train_losses if l is not None]

    return filtered_steps, filtered_losses, train_lrs, list(train_steps), train_times_s, val_steps, val_losses, val_times_s


def filter_by(keys, *arrays, min_val=0):
    """Filter parallel arrays to only include entries where key >= min_val."""
    indices = [i for i, k in enumerate(keys) if k >= min_val]
    return tuple([[keys[i] for i in indices]] + [[a[i] for i in indices] for a in arrays])


def annotate_final(ax, xs, ys, color=None):
    """Add a dashed horizontal line and text at the final data point."""
    if not xs:
        return
    ax.axhline(y=ys[-1], linestyle="--", linewidth=0.8, alpha=0.5, color=color)
    ax.annotate(f"{ys[-1]:.4f}", xy=(xs[-1], ys[-1]), fontsize=9,
                xytext=(5, 5), textcoords="offset points", color=color)


def apply_log_x(axes, log_x):
    if log_x:
        for ax in axes:
            ax.set_xscale("log")


def plot_run(run_id, train_steps, train_losses, lr_steps, train_lrs, train_times_s, val_steps, val_losses, val_times_s, out_path, log_x, min_step, min_time):
    # Filter data
    f_train_steps, f_train_losses = filter_by(train_steps, train_losses, min_val=min_step)
    f_lr_steps, f_train_lrs = filter_by(lr_steps, train_lrs, min_val=min_step)
    f_val_steps, f_val_losses = filter_by(val_steps, val_losses, min_val=min_step)
    f_val_times, f_val_losses_t = filter_by(val_times_s, val_losses, min_val=min_time)
    f_lr_times, f_lr_times_lrs = filter_by(train_times_s, train_lrs, min_val=min_time)

    fig, axes = plt.subplots(5, 1, figsize=(10, 20))
    fig.suptitle(run_id[:30], fontsize=17)

    axes[0].plot(f_train_steps, f_train_losses, linewidth=1.2, label="train loss")
    annotate_final(axes[0], f_train_steps, f_train_losses)
    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Train Loss", fontsize=12)
    axes[0].set_title("Train Loss vs Step", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f_val_steps, f_val_losses, marker="o", markersize=4, linewidth=1.2, label="val loss")
    annotate_final(axes[1], f_val_steps, f_val_losses)
    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("Val Loss", fontsize=12)
    axes[1].set_title("Val Loss vs Step", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(f_lr_steps, f_train_lrs, linewidth=1.2, label="lr")
    axes[2].set_xlabel("Step", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].set_title("Learning Rate vs Step", fontsize=13)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(f_val_times, f_val_losses_t, marker="o", markersize=4, linewidth=1.2, label="val loss")
    annotate_final(axes[3], f_val_times, f_val_losses_t)
    axes[3].set_xlabel("Train Time (s)", fontsize=12)
    axes[3].set_ylabel("Val Loss", fontsize=12)
    axes[3].set_title("Val Loss vs Time", fontsize=13)
    axes[3].legend(fontsize=11)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(f_lr_times, f_lr_times_lrs, linewidth=1.2, label="lr")
    axes[4].set_xlabel("Train Time (s)", fontsize=12)
    axes[4].set_ylabel("Learning Rate", fontsize=12)
    axes[4].set_title("Learning Rate vs Time", fontsize=13)
    axes[4].legend(fontsize=11)
    axes[4].grid(True, alpha=0.3)

    apply_log_x(axes, log_x)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training logs")
    parser.add_argument("--log-x", action="store_true", help="Use log scale for x axes")
    parser.add_argument("--min-step", type=int, default=0, help="Start plotting from this step (for train/val vs step charts)")
    parser.add_argument("--min-time", type=float, default=0, help="Start plotting from this time in seconds (for val vs time chart)")
    opts = parser.parse_args()

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    txt_files = sorted(glob.glob(os.path.join(logs_dir, "*.txt")))

    all_runs = []
    for path in txt_files:
        run_id = os.path.splitext(os.path.basename(path))[0]
        train_steps, train_losses, train_lrs, lr_steps, train_times_s, val_steps, val_losses, val_times_s = parse_log(path)

        if not val_steps and not train_steps:
            continue

        all_runs.append((run_id, train_steps, train_losses, lr_steps, train_lrs, train_times_s, val_steps, val_losses, val_times_s))

        out_path = os.path.join(logs_dir, f"{run_id}.png")
        plot_run(run_id, train_steps, train_losses, lr_steps, train_lrs, train_times_s, val_steps, val_losses, val_times_s, out_path, opts.log_x, opts.min_step, opts.min_time)

    # Combined plot
    if not all_runs:
        print("No runs with training data found.")
        return

    fig, axes = plt.subplots(5, 1, figsize=(12, 22))
    fig.suptitle("All Runs", fontsize=17)

    for run_id, train_steps, train_losses, lr_steps, train_lrs, train_times_s, val_steps, val_losses, val_times_s in all_runs:
        label = run_id[:30]
        f_train_steps, f_train_losses = filter_by(train_steps, train_losses, min_val=opts.min_step)
        f_lr_steps, f_train_lrs = filter_by(lr_steps, train_lrs, min_val=opts.min_step)
        f_val_steps, f_val_losses = filter_by(val_steps, val_losses, min_val=opts.min_step)
        f_val_times, f_val_losses_t = filter_by(val_times_s, val_losses, min_val=opts.min_time)
        f_lr_times, f_lr_times_lrs = filter_by(train_times_s, train_lrs, min_val=opts.min_time)
        if f_train_steps:
            line, = axes[0].plot(f_train_steps, f_train_losses, linewidth=1.2, label=label)
            annotate_final(axes[0], f_train_steps, f_train_losses, color=line.get_color())
        if f_val_steps:
            line, = axes[1].plot(f_val_steps, f_val_losses, marker="o", markersize=4, linewidth=1.2, label=label)
            annotate_final(axes[1], f_val_steps, f_val_losses, color=line.get_color())
        if f_train_lrs:
            axes[2].plot(f_lr_steps, f_train_lrs, linewidth=1.2, label=label)
        if f_val_times:
            line, = axes[3].plot(f_val_times, f_val_losses_t, marker="o", markersize=4, linewidth=1.2, label=label)
            annotate_final(axes[3], f_val_times, f_val_losses_t, color=line.get_color())
        if f_lr_times:
            axes[4].plot(f_lr_times, f_lr_times_lrs, linewidth=1.2, label=label)

    axes[0].set_xlabel("Step", fontsize=12)
    axes[0].set_ylabel("Train Loss", fontsize=12)
    axes[0].set_title("Train Loss vs Step", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step", fontsize=12)
    axes[1].set_ylabel("Val Loss", fontsize=12)
    axes[1].set_title("Val Loss vs Step", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Step", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].set_title("Learning Rate vs Step", fontsize=13)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel("Train Time (s)", fontsize=12)
    axes[3].set_ylabel("Val Loss", fontsize=12)
    axes[3].set_title("Val Loss vs Time", fontsize=13)
    axes[3].legend(fontsize=11)
    axes[3].grid(True, alpha=0.3)

    axes[4].set_xlabel("Train Time (s)", fontsize=12)
    axes[4].set_ylabel("Learning Rate", fontsize=12)
    axes[4].set_title("Learning Rate vs Time", fontsize=13)
    axes[4].legend(fontsize=11)
    axes[4].grid(True, alpha=0.3)

    apply_log_x(axes, opts.log_x)
    plt.tight_layout()
    out_path = os.path.join(logs_dir, "all_runs.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
