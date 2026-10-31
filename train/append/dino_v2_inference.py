import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import ndimage
from torchvision.transforms import v2

from dinov2.models.vision_transformer import vit_small

model = vit_small(
    patch_size=14,
    img_size=518,
    init_values=1.0,
    block_chunks=0,
)

weights_path = Path("deployment/model_weights/dinov2_vits14_pretrain.pth")
print("Loading weights from: ", weights_path)
state_dict = torch.load(weights_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", len(missing), "| Unexpected keys:", len(unexpected))

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"
model.to(device)
model.eval()
print("Model Loaded")

H, W = 560, 560

transform = v2.Compose([
    v2.Resize((H, W), interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(H),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

frames_dir = Path("train/vint_train/data/go_stanford/no1vc_9_1")
file_list = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
batch_size = len(file_list)
print("Found", batch_size, "frames")
if batch_size == 0:
    raise FileNotFoundError(f"No .jpg files found in {frames_dir}")

imgs_tensor = torch.zeros(batch_size, 3, H, W)
for i, path in enumerate(file_list):
    img = Image.open(path).convert("RGB")
    imgs_tensor[i] = transform(img)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

features = []
with torch.no_grad():
    for start in range(0, batch_size, 8):
        chunk = imgs_tensor[start:start + 8].to(device)
        out = model.forward_features(chunk)["x_norm_patchtokens"]
        features.append(F.normalize(out, dim=-1).cpu())
features = torch.cat(features)

B, N, C = features.shape
grid_h, grid_w = H// 14, W//14
assert N == grid_h * grid_w
grid = int(N ** 0.5)
print("Patch features:", features.shape)

seed_mask = torch.zeros(grid_h, grid_w,  dtype=torch.bool)
seed_mask[grid_h - 4:grid_h, grid_w // 3:2 * grid_w // 3] = True
seed_flat = seed_mask.flatten()
seed_np = seed_mask.numpy()

sim_maps = []
free_maps = []
for i in range(B):
    feats = features[i]
    prototype = F.normalize(feats[seed_flat].mean(dim=0), dim=0)
    sim = feats @ prototype
    seed_sim = sim[seed_flat]
    threshold = seed_sim.mean() - 2.0 * seed_sim.std()
    binary = (sim > threshold).view(grid_h, grid_w).numpy()
    labeled, _ = ndimage.label(binary)
    seed_labels = np.unique(labeled[seed_np])
    seed_labels = seed_labels[seed_labels > 0]
    connected = np.isin(labeled, seed_labels)
    sim_maps.append(sim.view(grid_h, grid_w).numpy())
    free_maps.append(connected.astype(np.float32))

sim_maps = np.stack(sim_maps)
free_maps = np.stack(free_maps)


def upsample(arr, size=560):
    t = torch.from_numpy(arr).unsqueeze(1).float()
    return F.interpolate(t, size=size, mode="bilinear", align_corners=False)[:, 0].numpy()


sim_up = upsample(sim_maps)
free_up = (upsample(free_maps) > 0.5).astype(np.float32)

originals = (imgs_tensor * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()

out_dir = Path("train/append/dinov2_results")
out_dir.mkdir(parents=True, exist_ok=True)

# np.save(out_dir / "traversability_similarity.npy", sim_maps)
# np.save(out_dir / "traversability_free_mask.npy", free_maps)

num_show = min(4, B)
fig, axes = plt.subplots(3, num_show, figsize=(4 * num_show, 12))
axes = np.atleast_2d(axes).reshape(3, num_show)
for i in range(num_show):
    axes[0, i].imshow(originals[i])
    axes[1, i].imshow(sim_up[i], cmap="viridis")
    axes[2, i].imshow(originals[i])
    axes[2, i].imshow(free_up[i], alpha=0.45, cmap="RdYlGn", vmin=0, vmax=1)
    for row in range(3):
        axes[row, i].axis("off")
plt.tight_layout()
plt.savefig(out_dir / "dino_traversability.jpg", dpi=150)
print("Saved to", out_dir)