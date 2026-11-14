import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT
from vint_train.models.travnav.trav_modules import TraversabilityDecoder


def load_encoder(weights, device):
    model = NoMaD_ViNT(
        obs_encoding_size=256,
        context_size=3,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    ckpt = torch.load(weights, map_location="cpu")
    if hasattr(ckpt, "state_dict"):
        ckpt = ckpt.state_dict()
    prefix = "vision_encoder."
    state = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded", len(state), "tensors | missing:", len(missing), "| unexpected:", len(unexpected))
    encoder = model.obs_encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", default="overfit_decoder.png")
    ap.add_argument("--loss_save", default="overfit_loss.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    tf = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    stems = list(range(args.n))
    frames, imgs, targets = [], [], []
    for s in stems:
        path = None
        for e in (".jpg", ".jpeg", ".png"):
            p = Path(args.data) / args.traj / f"{s}{e}"
            if p.exists():
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"No frame {s} in {Path(args.data) / args.traj}")
        img = Image.open(path).convert("RGB")
        imgs.append(np.asarray(img.resize((args.size, args.size))))
        frames.append(tf(img))
        m = np.load(Path(args.maps) / args.traj / f"{s}.npy").astype(np.float32)
        targets.append(torch.from_numpy(m).unsqueeze(0))

    x = torch.stack(frames).to(device)
    target = torch.stack(targets).to(device)
    print("Frames:", tuple(x.shape), "| Targets:", tuple(target.shape))

    encoder = load_encoder(args.weights, device)

    with torch.no_grad():
        ends = encoder.extract_endpoints(x)
    s8, s16, s32 = ends["reduction_3"], ends["reduction_4"], ends["reduction_5"]
    for name, t in zip(["s8", "s16", "s32"], [s8, s16, s32]):
        print(name, tuple(t.shape))

    decoder = TraversabilityDecoder([s8.shape[1], s16.shape[1], s32.shape[1]]).to(device)
    print("Decoder parameters:", sum(p.numel() for p in decoder.parameters()))
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    losses = []
    for step in range(args.steps + 1):
        opt.zero_grad()
        pred = decoder(s8, s16, s32)
        loss = F.l1_loss(pred, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:4d} | L1 {loss.item():.4f}")


    fig_l, ax_l = plt.subplots(figsize=(6, 4))
    ax_l.plot(losses)
    ax_l.set_xlabel("step")
    ax_l.set_ylabel("L1 loss")
    ax_l.set_title("Decoder overfit loss")
    ax_l.grid(alpha=0.3)
    fig_l.tight_layout()
    fig_l.savefig(args.loss_save, dpi=120)
    plt.close(fig_l)
    print("Saved", args.loss_save)

    
    decoder.eval()
    with torch.no_grad():
        pred = decoder(s8, s16, s32).cpu().numpy()
    target_np = target.cpu().numpy()

    fig, axes = plt.subplots(args.n, 3, figsize=(9, args.n * 3))
    axes = np.atleast_2d(axes)
    for i in range(args.n):
        axes[i, 0].imshow(imgs[i])
        axes[i, 1].imshow(pred[i, 0], vmin=0, vmax=1, cmap="gray")
        axes[i, 2].imshow(target_np[i, 0], vmin=0, vmax=1, cmap="gray")
        for j in range(3):
            axes[i, j].axis("off")
    axes[0, 0].set_title("frame")
    axes[0, 1].set_title("predicted")
    axes[0, 2].set_title("teacher")
    plt.tight_layout()
    fig.savefig(args.save, dpi=120)
    print("Saved", args.save)


if __name__ == "__main__":
    main()