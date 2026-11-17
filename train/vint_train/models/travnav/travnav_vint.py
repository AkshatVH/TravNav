import torch
import torch.nn as nn
from typing import Optional, Sequence

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT
from vint_train.models.vint.self_attention import PositionalEncoding
from vint_train.models.travnav.trav_modules import (
    TraversabilityDecoder,
    TraversabilityTokenizer,
)


class TravNav_ViNT(NoMaD_ViNT):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        num_trav_tokens: int = 4,
        trav_dropout: float = 0.1,
        use_trav: bool = True,
        decoder_in_channels: Sequence[int] = (40, 112, 320),
    ) -> None:
        super().__init__(
            context_size=context_size,
            obs_encoder=obs_encoder,
            obs_encoding_size=obs_encoding_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        self.num_trav_tokens = num_trav_tokens
        self.trav_dropout = trav_dropout
        self.use_trav = use_trav
        self.decoder_in_channels = list(decoder_in_channels)
        self.last_trav_map = None

        self.grid = int(round(num_trav_tokens ** 0.5))
        if self.grid * self.grid != num_trav_tokens:
            raise ValueError(
                f"num_trav_tokens must be a perfect square, got {num_trav_tokens}"
            )

        if not use_trav:
            self.num_tokens = context_size + 2
            return

        self.num_tokens = (context_size + 1) + num_trav_tokens + 1

        self.decoder = TraversabilityDecoder(
            in_channels=self.decoder_in_channels, mid_channels=64
        )
        self.tokenizer = TraversabilityTokenizer(
            grid_size=self.grid, embed_dim=obs_encoding_size
        )
        self.type_embedding = nn.Embedding(3, obs_encoding_size)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.01)

        type_ids = torch.cat([
            torch.zeros(context_size + 1, dtype=torch.long),
            torch.ones(num_trav_tokens, dtype=torch.long),
            torch.tensor([2], dtype=torch.long),
        ]).unsqueeze(0)
        self.register_buffer("type_ids", type_ids, persistent=False)

        self._rebuild_for_tokens()

    def _rebuild_for_tokens(self):
        n = self.num_tokens
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=n)
        self.no_mask = torch.zeros((1, n), dtype=torch.bool)
        self.goal_mask = torch.zeros((1, n), dtype=torch.bool)
        self.goal_mask[:, -1] = True
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)
        self.avg_pool_mask = torch.cat(
            [
                1 - self.no_mask.float(),
                (1 - self.goal_mask.float()) * (n / (n - 1)),
            ],
            dim=0,
        )

    def _drop_trav_tokens(self, tokens):
        if not self.training or self.trav_dropout <= 0:
            return tokens
        keep = (torch.rand(tokens.size(0), 1, 1, device=tokens.device) >= self.trav_dropout)
        return tokens * keep.to(tokens.dtype)

    def forward(self, obs_img, goal_img, input_goal_mask=None):
        device = obs_img.device

        if not self.use_trav:
            self.last_trav_map = None
            if input_goal_mask is None:
                input_goal_mask = torch.zeros(obs_img.size(0), dtype=torch.long, device=device)
            return super().forward(obs_img, goal_img, input_goal_mask)

        B = obs_img.size(0)

        goal_mask = None
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        obsgoal_img = torch.cat([obs_img[:, 3 * self.context_size:, :, :], goal_img], dim=1)
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding)
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)
        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        goal_encoding = obsgoal_encoding

        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        endpoints = self.obs_encoder.extract_endpoints(obs_img)
        keys = sorted(endpoints.keys())
        s8, s16, s32 = [endpoints[k][-B:] for k in keys[-4:-1]]
        obs_encoding = endpoints[keys[-1]]

        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        trav_map = self.decoder(s8, s16, s32)
        self.last_trav_map = trav_map
        trav_tokens = self.tokenizer(trav_map)
        trav_tokens = self._drop_trav_tokens(trav_tokens)

        tokens = torch.cat((obs_encoding, trav_tokens, goal_encoding), dim=1)

        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None

        tokens = self.positional_encoding(tokens)
        tokens = tokens + self.type_embedding(self.type_ids)

        out = self.sa_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
            out = out * avg_mask
        out = torch.mean(out, dim=1)
        return out




def load_pretrained_travnav(model, ckpt_path, prefix=""):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
        ckpt = ckpt["model"]
    if hasattr(ckpt, "state_dict"):
        ckpt = ckpt.state_dict()

    cleaned = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if prefix:
            if not k.startswith(prefix):
                continue
            k = k[len(prefix):]
        cleaned[k] = v

    own = model.state_dict()
    keep, not_in_model, shape_mismatch = {}, [], []
    for k, v in cleaned.items():
        if k not in own:
            not_in_model.append(k)
        elif own[k].shape != v.shape:
            shape_mismatch.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            keep[k] = v

    result = model.load_state_dict(keep, strict=False)

    expected_new = ("decoder.", "tokenizer.", "type_embedding.", "positional_encoding.")
    suspicious = [k for k in result.missing_keys if not k.startswith(expected_new)]

    print("checkpoint keys:", len(ckpt), "| usable:", len(keep))
    print("missing:", len(result.missing_keys), "| unexpected:", len(result.unexpected_keys))
    print("not in model:", len(not_in_model), "| shape mismatch:", len(shape_mismatch))
    for item in shape_mismatch:
        print("  shape mismatch:", item)
    print("suspicious missing keys:", len(suspicious))
    for k in suspicious[:20]:
        print("  ", k)

    if keep:
        probe = next(iter(keep))
        same = torch.equal(model.state_dict()[probe].cpu(), keep[probe])
        print("spot check", probe, "->", "OK" if same else "MISMATCH")

    return {
        "missing": result.missing_keys,
        "unexpected": result.unexpected_keys,
        "not_in_model": not_in_model,
        "shape_mismatch": shape_mismatch,
        "suspicious": suspicious,
    }

