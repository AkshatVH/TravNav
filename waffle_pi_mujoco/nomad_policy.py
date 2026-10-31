from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

ACTION_MIN = np.array([-2.5, -4.0], dtype=np.float32)
ACTION_MAX = np.array([5.0, 4.0], dtype=np.float32)

PREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class NoMaDPolicy:
    def __init__(
        self,
        ckpt_path,
        device=None,
        context_size=3,
        image_size=(96, 96),
        len_traj_pred=8,
        encoding_size=256,
        num_diffusion_iters=10,
        num_samples=8,
        down_dims=(64, 128, 256),
        mha_heads=4,
        mha_layers=4,
        mha_ff_factor=4,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
            # device = "cpu"
        )
        self.context_size = context_size
        self.image_size = tuple(image_size)
        self.len_traj_pred = len_traj_pred
        self.num_diffusion_iters = num_diffusion_iters
        self.num_samples = num_samples
        self.queue = deque(maxlen=context_size + 1)
        self.goal = None

        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_heads,
            mha_num_attention_layers=mha_layers,
            mha_ff_dim_factor=mha_ff_factor,
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=encoding_size,
            down_dims=list(down_dims),
            cond_predict_scale=False,
        )
        dist_pred_net = DenseNetwork(embedding_dim=encoding_size)
        self.model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_net,
        )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()

        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _to_tensor(self, frame):
        img = Image.fromarray(frame).convert("RGB").resize(self.image_size)
        return PREPROCESS(img)

    def reset(self):
        self.queue.clear()

    def push(self, frame):
        tensor = self._to_tensor(frame)
        if len(self.queue) == 0:
            for _ in range(self.queue.maxlen):
                self.queue.append(tensor)
        else:
            self.queue.append(tensor)

    def set_goal(self, frame):
        self.goal = self._to_tensor(frame).unsqueeze(0).to(self.device)

    def clear_goal(self):
        self.goal = None

    @torch.no_grad()
    def infer(self):
        obs = torch.cat(list(self.queue), dim=0).unsqueeze(0).to(self.device)

        if self.goal is None:
            goal = torch.randn((1, 3, *self.image_size), device=self.device)
            mask = torch.ones(1, dtype=torch.long, device=self.device)
        else:
            goal = self.goal
            mask = torch.zeros(1, dtype=torch.long, device=self.device)

        obs_cond = self.model(
            "vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=mask
        )
        obs_cond = obs_cond.repeat(self.num_samples, 1)

        naction = torch.randn(
            (self.num_samples, self.len_traj_pred, 2), device=self.device
        )
        self.scheduler.set_timesteps(self.num_diffusion_iters)
        for k in self.scheduler.timesteps:
            noise_pred = self.model(
                "noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond
            )
            naction = self.scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        deltas = naction.detach().cpu().numpy().reshape(self.num_samples, -1, 2)
        deltas = (deltas + 1.0) / 2.0
        deltas = deltas * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
        return np.cumsum(deltas, axis=1)

    @torch.no_grad()
    def goal_distance(self, obs_frames, goal_frame):
        goal = self._to_tensor(goal_frame).unsqueeze(0).to(self.device)
        obs = torch.cat(
            [self._to_tensor(f) for f in obs_frames], dim=0
        ).unsqueeze(0).to(self.device)
        mask = torch.zeros(1, dtype=torch.long, device=self.device)
        cond = self.model(
            "vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=mask
        )
        return self.model("dist_pred_net", obsgoal_cond=cond).item()