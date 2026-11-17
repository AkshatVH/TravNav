import os
import random
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def to_image(t):
    a = t.detach().cpu().float().numpy().transpose(1, 2, 0)
    if a.min() < 0:
        a = a * STD + MEAN
    return np.clip(a, 0, 1)


def raw_frame(dataset, i):
    f_curr, curr_time = dataset.index_to_data[i][0], dataset.index_to_data[i][1]
    path = os.path.join(dataset.data_folder, f_curr, f"{curr_time}.jpg")
    if not os.path.exists(path):
        print("missing raw frame", path)
        return None, f_curr, curr_time
    return np.asarray(Image.open(path).convert("RGB")), f_curr, curr_time


def to_numpy_map(tmap):
    if isinstance(tmap, torch.Tensor):
        return tmap.detach().cpu().float().numpy()[0]
    return np.asarray(tmap, dtype=np.float32)[0]


def available_indices(dataset, teacher_dir, dataset_name):
    root = os.path.join(teacher_dir, dataset_name)
    have = set(os.listdir(root))
    out = []
    for i, entry in enumerate(dataset.index_to_data):
        f_curr, curr_time = entry[0], entry[1]
        if f_curr in have and os.path.exists(os.path.join(root, f_curr, f"{curr_time}.npy")):
            out.append(i)
    return out


def run(dataset, n=20, per_page=5, out_prefix="dataset_check", obs_index=0, teacher_index=-1, seed=0, indices=None):
    random.seed(seed)
    pool = list(range(len(dataset))) if indices is None else list(indices)
    print("items with maps:", len(pool))
    idx = random.sample(pool, min(n, len(pool)))
    pages = [idx[k:k + per_page] for k in range(0, len(idx), per_page)]
    for p, page in enumerate(pages):
        fig, axes = plt.subplots(len(page), 4, figsize=(14, 3.2 * len(page)))
        axes = np.atleast_2d(axes)
        for r, i in enumerate(page):
            item = dataset[i]
            obs = item[obs_index]
            tmap = item[teacher_index]
            m = to_numpy_map(tmap)
            cur = to_image(obs[-3:])
            raw, f_curr, t = raw_frame(dataset, i)
            h, w = cur.shape[:2]
            print(i, f_curr, t, tuple(tmap.shape), tmap.dtype, float(m.min()), float(m.max()), float(m.mean()))
            if raw is not None:
                axes[r, 0].imshow(raw)
            axes[r, 0].set_title(f"raw {f_curr} {t}", fontsize=8)
            axes[r, 1].imshow(cur)
            axes[r, 1].set_title("dataset frame", fontsize=8)
            axes[r, 2].imshow(m, vmin=0, vmax=1, cmap="viridis")
            axes[r, 2].set_title(f"map mean {m.mean():.2f}", fontsize=8)
            axes[r, 3].imshow(cur)
            axes[r, 3].imshow(m, alpha=0.5, extent=(0, w, h, 0), cmap="jet", vmin=0, vmax=1)
            axes[r, 3].set_title("overlay", fontsize=8)
            for a in axes[r]:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{p}.png", dpi=110)
        plt.close(fig)