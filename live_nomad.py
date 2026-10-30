import argparse
import sys
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--goal_img", default=None)
    ap.add_argument("--goal_offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--out", default="results/nomad_frames")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def frame_stream(args):
    files = sorted(Path(args.traj).glob("*.jpg"), key=lambda p: int(p.stem))
    for i in range(0, len(files), args.stride):
        goal = None
        if args.goal_offset > 0:
            j = min(i + args.goal_offset, len(files) - 1)
            goal = Image.open(files[j]).convert("RGB")

        yield i, Image.open(files[i]).convert("RGB"), goal


@torch.no_grad()
def predict(model, sched, params, queue, goal, device, n, transform_images, get_action, to_numpy):
    size = params["image_size"]
    obs = transform_images(list(queue), size, center_crop=False).to(device)
    if goal is None:
        g = torch.randn(1, 3, size[1], size[0], device=device)
        mask = torch.ones(1, device=device).long()
    else:
        g = transform_images(goal, size, center_crop=False).to(device)
        mask = torch.zeros(1, device=device).long()
    cond = model("vision_encoder", obs_img=obs, goal_img=g, input_goal_mask=mask)
    dist = None
    if goal is not None:
        dist = model("dist_pred_net", obsgoal_cond=cond).item()
    cond = cond.repeat(n, 1)
    a = torch.randn((n, params["len_traj_pred"], 2), device=device)
    sched.set_timesteps(params["num_diffusion_iters"])
    for k in sched.timesteps:
        noise = model("noise_pred_net", sample=a, timestep=k, global_cond=cond)
        a = sched.step(model_output=noise, timestep=k, sample=a).prev_sample
    return to_numpy(get_action(a)), dist


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "deployment" / "src"))
    sys.modules["sensor_msgs"] = MagicMock()
    sys.modules["sensor_msgs.msg"] = MagicMock()
    from utils import load_model, to_numpy, transform_images
    from vint_train.training.train_utils import get_action

    config_path = args.config or str(repo / "train" / "config" / "nomad.yaml")
    ckpt_path = args.ckpt or str(repo / "deployment" / "model_weights" / "nomad.pth")
    with open(config_path) as f:
        params = yaml.safe_load(f)

    device = torch.device(args.device)
    model = load_model(ckpt_path, params, device).to(device).eval()
    sched = DDPMScheduler(
        num_train_timesteps=params["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    fixed_goal = Image.open(args.goal_img).convert("RGB") if args.goal_img else None
    queue = deque(maxlen=params["context_size"] + 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_img, ax_path) = plt.subplots(1, 2, figsize=(11, 5))
    saved = 0

    for idx, frame, goal in frame_stream(args):
        print(frame.size())
        queue.append(frame)
        if len(queue) < queue.maxlen:
            continue
        goal = fixed_goal or goal
        actions, dist = predict(
            model, sched, params, queue, goal, device, args.samples,
            transform_images, get_action, to_numpy,
        )

        ax_img.clear()
        ax_img.imshow(frame)
        ax_img.axis("off")
        title = f"frame {idx}  " + ("goal-conditioned" if goal is not None else "exploration")
        if dist is not None:
            title += f"  dist={dist:.1f}"
        ax_img.set_title(title)

        ax_path.clear()
        for traj in actions:
            xy = np.vstack([[0, 0], traj[:, :2]])
            ax_path.plot(-xy[:, 1], xy[:, 0], marker="o", markersize=3, alpha=0.6)
        ax_path.scatter([0], [0], c="k", marker="^", s=80)
        ax_path.set_xlim(-6, 6)
        ax_path.set_ylim(-0.5, 12)
        ax_path.set_aspect("equal")
        ax_path.grid(True)
        ax_path.set_xlabel("left (waypoint units)")
        ax_path.set_ylabel("forward (waypoint units)")

        fig.savefig(out_dir / f"{saved:06d}.png", dpi=80)
        saved += 1
        print(f"saved {saved} (frame {idx})", flush=True)
        if args.max_steps and saved >= args.max_steps:
            break

    print(f"done, {saved} images in {out_dir}")


if __name__ == "__main__":
    main()