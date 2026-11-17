import pytest
import torch

from vint_train.models.travnav.travnav_vint import TravNav_ViNT

B = 2
CONTEXT = 5
SIZE = 64
IMG = 96
DECODER_CH = (40, 112, 320)


def make_model(use_trav=True, trav_dropout=0.1):
    torch.manual_seed(0)
    model = TravNav_ViNT(
        context_size=CONTEXT,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=SIZE,
        num_trav_tokens=4,
        trav_dropout=trav_dropout,
        use_trav=use_trav,
        decoder_in_channels=DECODER_CH,
    )
    model.eval()
    return model


def make_inputs(batch=B, seed=0):
    torch.manual_seed(seed)
    obs = torch.rand(batch, 3 * (CONTEXT + 1), IMG, IMG)
    goal = torch.rand(batch, 3, IMG, IMG)
    mask = torch.zeros(batch, dtype=torch.long)
    return obs, goal, mask


def record_shapes(model):
    shapes = []

    def hook(module, args):
        shapes.append(tuple(args[0].shape))

    handle = model.sa_encoder.register_forward_pre_hook(hook)
    return shapes, handle


@pytest.fixture(scope="module")
def trav_model():
    return make_model(use_trav=True)


@pytest.fixture(scope="module")
def plain_model():
    return make_model(use_trav=False)


def test_token_count_with_trav(trav_model):
    obs, goal, mask = make_inputs()
    shapes, handle = record_shapes(trav_model)
    with torch.no_grad():
        trav_model(obs, goal, mask)
    handle.remove()
    assert shapes[-1] == (B, 11, SIZE)


def test_output_shape(trav_model):
    obs, goal, mask = make_inputs()
    with torch.no_grad():
        out = trav_model(obs, goal, mask)
    assert out.shape == (B, SIZE)
    assert torch.isfinite(out).all()


def test_goal_mask_changes_output(trav_model):
    obs, goal, _ = make_inputs()
    with torch.no_grad():
        out0 = trav_model(obs, goal, torch.zeros(B, dtype=torch.long))
        out1 = trav_model(obs, goal, torch.ones(B, dtype=torch.long))
    assert not torch.allclose(out0, out1)


def test_trav_map(trav_model):
    obs, goal, mask = make_inputs()
    with torch.no_grad():
        trav_model(obs, goal, mask)
    m = trav_model.last_trav_map
    assert m is not None
    assert m.shape == (B, 1, 64, 64)
    assert m.min() >= 0
    assert m.max() <= 1


def test_no_trav_path(plain_model):
    obs, goal, mask = make_inputs()
    shapes, handle = record_shapes(plain_model)
    with torch.no_grad():
        out = plain_model(obs, goal, mask)
    handle.remove()
    assert shapes[-1] == (B, 7, SIZE)
    assert out.shape == (B, SIZE)
    assert plain_model.last_trav_map is None
    assert not hasattr(plain_model, "decoder")
    assert not hasattr(plain_model, "tokenizer")


def test_dropout_method():
    model = make_model(trav_dropout=1.0)
    tokens = torch.randn(B, 4, SIZE)

    model.train()
    assert torch.equal(model._drop_trav_tokens(tokens), torch.zeros_like(tokens))

    model.eval()
    assert torch.equal(model._drop_trav_tokens(tokens), tokens)

    model.train()
    model.trav_dropout = 0.0
    assert torch.equal(model._drop_trav_tokens(tokens), tokens)

    torch.manual_seed(0)
    model.trav_dropout = 0.5
    big = torch.ones(1000, 4, SIZE)
    out = model._drop_trav_tokens(big)
    zeroed = (out.sum(dim=(1, 2)) == 0).float().mean().item()
    assert 0.4 < zeroed < 0.6


def test_current_frame_drives_map(trav_model):
    obs, goal, mask = make_inputs()
    with torch.no_grad():
        trav_model(obs, goal, mask)
        base = trav_model.last_trav_map.clone()

        obs_last = obs.clone()
        obs_last[:, 15:18] = torch.rand_like(obs_last[:, 15:18])
        trav_model(obs_last, goal, mask)
        changed = trav_model.last_trav_map.clone()

        obs_first = obs.clone()
        obs_first[:, 0:3] = torch.rand_like(obs_first[:, 0:3])
        trav_model(obs_first, goal, mask)
        unchanged = trav_model.last_trav_map.clone()

    assert not torch.allclose(base, changed)
    assert torch.allclose(base, unchanged, atol=1e-6)


def test_mask_sizes(trav_model, plain_model):
    for name in ["no_mask", "goal_mask", "all_masks", "avg_pool_mask"]:
        assert getattr(trav_model, name).shape[-1] == 11
        assert getattr(plain_model, name).shape[-1] == 7

    assert trav_model.all_masks.shape == (2, 11)
    assert trav_model.goal_mask[0, -1].item() is True
    assert not trav_model.goal_mask[0, :-1].any()
    assert torch.allclose(trav_model.avg_pool_mask[0], torch.ones(11))
    assert torch.allclose(trav_model.avg_pool_mask[1, :-1], torch.full((10,), 11 / 10))
    assert trav_model.avg_pool_mask[1, -1] == 0

    assert torch.allclose(plain_model.avg_pool_mask[1, :-1], torch.full((6,), 7 / 6))
    assert plain_model.avg_pool_mask[1, -1] == 0


def test_type_ids(trav_model):
    expected = torch.tensor([[0] * 6 + [1] * 4 + [2]])
    assert trav_model.type_ids.shape == (1, 11)
    assert torch.equal(trav_model.type_ids, expected)
    assert "type_ids" not in trav_model.state_dict()


def test_diffusion_head_accepts_output(trav_model):
    from diffusion_policy.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    head = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=SIZE,
        down_dims=[64, 128, 256],
        cond_predict_scale=True,
    )
    head.eval()
    obs, goal, mask = make_inputs()
    with torch.no_grad():
        cond = trav_model(obs, goal, mask)
        noisy = torch.randn(B, 8, 2)
        t = torch.randint(0, 10, (B,))
        pred = head(sample=noisy, timestep=t, global_cond=cond)
    assert pred.shape == noisy.shape


def test_gradients_reach_new_modules():
    model = make_model(trav_dropout=0.0)
    model.train()
    obs, goal, mask = make_inputs()
    out = model(obs, goal, mask)
    (out.sum() + model.last_trav_map.sum()).backward()
    for module in (model.decoder, model.tokenizer, model.type_embedding):
        for name, p in module.named_parameters():
            assert p.grad is not None, name
    enc_grads = [p.grad for p in model.obs_encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0


def test_map_not_detached():
    model = make_model(trav_dropout=0.0)
    model.train()
    obs, goal, mask = make_inputs()
    model(obs, goal, mask)
    assert model.last_trav_map.requires_grad
    assert model.last_trav_map.grad_fn is not None


@pytest.mark.parametrize("batch", [1, 3])
def test_batch_size_independence(trav_model, batch):
    obs, goal, mask = make_inputs(batch=batch)
    with torch.no_grad():
        out = trav_model(obs, goal, mask)
    assert out.shape == (batch, SIZE)
    assert trav_model.last_trav_map.shape == (batch, 1, 64, 64)