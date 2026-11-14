import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage

from dinov2.models.vision_transformer import vit_small


class PrototypeHead:
    def __init__(self, seed_rows=3, sigma=2.0):
        self.seed_rows = seed_rows
        self.sigma = sigma
        self.stats = []

    def __call__(self, grid):
        B, C, H, W = grid.shape
        seed = torch.zeros(H, W, dtype=torch.bool, device=grid.device)
        seed[H - self.seed_rows:H, W // 3:2 * W // 3] = True
        seed_flat = seed.flatten()
        seed_np = seed.cpu().numpy()

        maps = []
        self.stats = []
        for i in range(B):
            feats = F.normalize(grid[i].flatten(1).T, dim=-1)
            proto = F.normalize(feats[seed_flat].mean(dim=0), dim=0)
            sim = feats @ proto

            seed_sim = sim[seed_flat]
            seed_mean = seed_sim.mean()
            seed_std = seed_sim.std()
            thr = seed_mean - self.sigma * seed_std

            binary = (sim > thr).view(H, W).cpu().numpy()
            labeled, _ = ndimage.label(binary)
            labels = np.unique(labeled[seed_np])
            labels = labels[labels > 0]
            connected = torch.from_numpy(
                np.isin(labeled, labels).astype(np.float32)
            ).to(grid.device)

            soft = ((sim.view(H, W) - thr) / (seed_mean - thr).clamp(min=1e-6)).clamp(0, 1)
            maps.append(soft * connected)
            self.stats.append((seed_mean.item(), seed_std.item()))

        return torch.stack(maps).unsqueeze(1)


def load_dino(device, weights_path):
    model = vit_small(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0,
    )
    print("Loading weights from:", weights_path)
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(missing), "| Unexpected keys:", len(unexpected))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_head(seed_rows=3, sigma=2.0):
    return PrototypeHead(seed_rows, sigma)


@torch.no_grad()
def teacher_map(backbone, head, frames):
    feats = backbone.forward_features(frames)["x_norm_patchtokens"]
    side = frames.shape[-1] // 14
    grid = feats.transpose(1, 2).reshape(feats.shape[0], -1, side, side)
    t_map = head(grid)
    t_map = F.interpolate(t_map, size=(64, 64), mode="bilinear", align_corners=False)
    return t_map.clamp(0, 1)


def list_frames(traj):
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in traj.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: int(p.stem))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="deployment/model_weights/dinov2_vits14_pretrain.pth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed_rows", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print("Device:", device)
    backbone = load_dino(device, args.weights)
    head = load_head(args.seed_rows, args.sigma)
    flag = True

    tf = T.Compose([
        T.Resize((448, 448)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_dir = Path(args.data)
    if any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in data_dir.iterdir()):
        trajs = [data_dir]
    else:
        trajs = sorted(d for d in data_dir.iterdir() if d.is_dir())

    rows = []
    done = 0

    for traj in trajs:
        files = list_frames(traj)
        if not files:
            print("No frames found in", traj)
            continue

        out_dir = Path(args.out) / traj.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            if args.limit:
                chunk = chunk[:args.limit - done]
            if not chunk:
                break
            if flag: 
                # single = Image.open(chunk[0])
                print(chunk)
                flag = False
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in chunk]).to(device)
            
            maps = teacher_map(backbone, head, x).cpu().numpy().astype(np.float16)

            for f, m, (s_mean, s_std) in zip(chunk, maps, head.stats):
                # if flag: 
                #     print(np.shape(m[0]))
                #     flag = False
                np.save(out_dir / f"{f.stem}.npy", m[0])
                rows.append((traj.name, f.stem, s_mean, s_std))

            done += len(chunk)
            if args.limit and done >= args.limit:
                break

        print(f"{traj.name}: {done} frames done in total")
        if args.limit and done >= args.limit:
            break

    if not rows:
        raise RuntimeError(f"No frames processed from {args.data}")

    csv_path = Path(args.out).parent / f"seed_stats_{Path(args.out).name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["traj", "frame", "seed_mean", "seed_std"])
        writer.writerows(rows)
    print("Saved stats to", csv_path)


if __name__ == "__main__":
    main()