import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(run_dir):
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    data = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        data[tag] = ([e.step for e in events], [e.value for e in events])
    return data


def smooth(values, weight):
    out = []
    last = None
    for v in values:
        if v != v or v in (float("inf"), float("-inf")):
            out.append(last if last is not None else v)
            continue
        last = v if last is None else last * weight + v * (1 - weight)
        out.append(last)
    return out


def find_runs(paths):
    runs = []
    for p in paths:
        if any(f.startswith("events.out") for f in os.listdir(p)):
            runs.append(p)
            continue
        for name in sorted(os.listdir(p)):
            sub = os.path.join(p, name)
            if os.path.isdir(sub) and any(f.startswith("events.out") for f in os.listdir(sub)):
                runs.append(sub)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="curves.png")
    parser.add_argument("--smooth", type=float, default=0.9)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--logy", nargs="*", default=[])
    args = parser.parse_args()

    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit("no event files found")

    all_data = {os.path.basename(os.path.normpath(r)): load_scalars(r) for r in runs}
    tags = args.tags
    if not tags:
        tags = sorted({t for d in all_data.values() for t in d})

    rows = math.ceil(len(tags) / args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(5.5 * args.cols, 3.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, tag in zip(axes.flat, tags):
        ax.axis("on")
        for name, data in all_data.items():
            if tag not in data:
                continue
            steps, values = data[tag]
            marker = "o" if len(values) < 15 else None
            line, = ax.plot(steps, smooth(values, args.smooth) if len(values) >= 15 else values, label=name, linewidth=1.6, marker=marker)
            if len(values) == 1:
                ax.annotate(f"{values[0]:.4g}", (steps[0], values[0]), textcoords="offset points", xytext=(6, 6), fontsize=8)
            if args.smooth > 0 and len(values) > 20:
                ax.plot(steps, values, color=line.get_color(), alpha=0.15, linewidth=0.8)
        ax.set_title(tag, fontsize=10)
        ax.grid(alpha=0.3)
        if tag in args.logy:
            ax.set_yscale("log")
        if len(all_data) > 1:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out} with {len(tags)} plots from {len(runs)} run(s)")
    for t in tags:
        print(" ", t)


if __name__ == "__main__":
    main()