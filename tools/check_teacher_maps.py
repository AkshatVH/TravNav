import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EXTS = [".jpg", ".jpeg", ".png"]


def find_frame(data_dir, traj, stem):
    for e in EXTS:
        p = data_dir / traj / f"{stem}{e}"
        if p.exists():
            return p
    return None


def composite(frame_path, map_path):
    img = Image.open(frame_path).convert("RGB")
    m = np.load(map_path).astype(np.float32)
    m = np.asarray(Image.fromarray(m).resize(img.size, Image.BILINEAR))
    m = np.clip(m, 0, 1)
    rgb = np.asarray(img).astype(np.float32) / 255
    heat = plt.cm.jet(m)[..., :3]
    overlay = 0.5 * rgb + 0.5 * heat
    gray = np.stack([m] * 3, axis=-1)
    return np.concatenate([rgb, gray, overlay], axis=1)


def save_grid(items, data_dir, out_dir, path, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (traj, stem, note) in zip(axes, items):
        frame_path = find_frame(data_dir, traj, stem)
        map_path = out_dir / traj / f"{stem}.npy"
        if frame_path is None or not map_path.exists():
            continue
        ax.imshow(composite(frame_path, map_path))
        ax.set_title(f"{traj}/{stem}{note}", fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    save_dir = out_dir.parent
    random.seed(0)

    trajs = [d for d in sorted(out_dir.iterdir()) if d.is_dir()]
    picked = random.sample(trajs, min(20, len(trajs)))
    items = []
    for t in picked:
        stems = [p.stem for p in t.glob("*.npy")]
        if stems:
            items.append((t.name, random.choice(stems), ""))
    save_grid(items, data_dir, out_dir, save_dir / f"check_grid_{out_dir.name}.png", 4, 5)

    csv_path = save_dir / f"seed_stats_{out_dir.name}.csv"
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["seed_std"]), reverse=True)
    worst = [(r["traj"], r["frame"], f"  std={float(r['seed_std']):.3f}") for r in rows[:10]]
    save_grid(worst, data_dir, out_dir, save_dir / f"check_worst_{out_dir.name}.png", 5, 2)

    print("Saved grids to", save_dir)


if __name__ == "__main__":
    main()