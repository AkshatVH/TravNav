import torch

from vint_train.models.travnav.travnav_vint import TravNav_ViNT, load_pretrained_travnav

CKPT = "deployment/model_weights/nomad.pth"

model = TravNav_ViNT(
    context_size=3,
    obs_encoder="efficientnet-b0",
    obs_encoding_size=256,
    mha_num_attention_heads=4,
    mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
    num_trav_tokens=4,
    trav_dropout=0.1,
    use_trav=True,
    decoder_in_channels=(40, 112, 320),
)

report = load_pretrained_travnav(model, CKPT, prefix="vision_encoder.")

print("missing sample:", report["missing"][:10])
print("shape mismatches:", report["shape_mismatch"])


print(model.positional_encoding.pos_enc.shape)