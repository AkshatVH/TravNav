import torch
import pytest

# Import all the components and functions from your test file
from vint_train.models.travnav.trav_modules import (
    TraversabilityDecoder,
    TraversabilityTokenizer,
    distil_loss,
    get_teacher_map,
)

IN_CHANNELS = [40, 112, 320]

def make_feats(b):
    return (
        torch.rand(b, IN_CHANNELS[0], 12, 12),
        torch.rand(b, IN_CHANNELS[1], 6, 6),
        torch.rand(b, IN_CHANNELS[2], 3, 3),
    )

# --- Individual Test Definitions ---

def test_decoder_shape():
    dec = TraversabilityDecoder(in_channels=IN_CHANNELS)
    out = dec(*make_feats(2))
    assert out.shape == (2, 1, 64, 64)

def test_decoder_range():
    dec = TraversabilityDecoder(in_channels=IN_CHANNELS)
    out = dec(*make_feats(2))
    assert out.min() >= 0
    assert out.max() <= 1

def test_decoder_batch_sizes():
    for b in [1, 5]:
        dec = TraversabilityDecoder(in_channels=IN_CHANNELS)
        out = dec(*make_feats(b))
        assert out.shape == (b, 1, 64, 64)

def test_decoder_dtype():
    dec = TraversabilityDecoder(in_channels=IN_CHANNELS)
    out = dec(*make_feats(2))
    assert out.dtype == torch.float32

def test_decoder_gradients():
    dec = TraversabilityDecoder(in_channels=IN_CHANNELS)
    dec(*make_feats(2)).sum().backward()
    for p in dec.parameters():
        assert p.grad is not None

def test_tokenizer_shape():
    for b in [1, 2, 5]:
        tok = TraversabilityTokenizer(grid_size=2, embed_dim=256)
        out = tok(torch.rand(b, 1, 64, 64))
        assert out.shape == (b, 4, 256)

def test_tokenizer_dtype():
    tok = TraversabilityTokenizer(grid_size=2, embed_dim=256)
    out = tok(torch.rand(2, 1, 64, 64))
    assert out.dtype == torch.float32

def test_tokenizer_grid():
    tok = TraversabilityTokenizer(grid_size=3, embed_dim=256)
    out = tok(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 9, 256)

def test_tokenizer_dim():
    tok = TraversabilityTokenizer(grid_size=2, embed_dim=128)
    out = tok(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 4, 128)

def test_tokenizer_gradients():
    tok = TraversabilityTokenizer(grid_size=2, embed_dim=256)
    tok(torch.rand(2, 1, 64, 64)).sum().backward()
    for p in tok.parameters():
        assert p.grad is not None

def test_tokenizer_input_size_independence():
    tok = TraversabilityTokenizer(grid_size=2, embed_dim=256)
    out = tok(torch.rand(2, 1, 96, 96))
    assert out.shape == (2, 4, 256)

def test_distil_loss_scalar():
    loss = distil_loss(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
    assert loss.dim() == 0

def test_distil_loss_zero_when_equal():
    x = torch.rand(2, 1, 64, 64)
    assert distil_loss(x, x.clone()).item() == 0

def test_distil_loss_positive_when_different():
    a = torch.zeros(2, 1, 64, 64)
    b = torch.ones(2, 1, 64, 64)
    assert distil_loss(a, b).item() > 0

def test_distil_loss_gradient():
    pred = torch.rand(2, 1, 64, 64, requires_grad=True)
    distil_loss(pred, torch.rand(2, 1, 64, 64)).backward()
    assert pred.grad is not None

def test_distil_loss_float16_target():
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64).to(torch.float16)
    loss = distil_loss(pred, target)
    assert loss.dtype == torch.float32

def test_get_teacher_map_dino():
    batch = torch.rand(2, 1, 64, 64).to(torch.float16)
    out = get_teacher_map("dino", batch, None)
    assert out.shape == (2, 1, 64, 64)
    assert out.dtype == torch.float32

def test_get_teacher_map_unknown_mode():
    with pytest.raises(ValueError):
        get_teacher_map("foo", torch.rand(2, 1, 64, 64), None)

def test_get_teacher_map_path_not_implemented():
    with pytest.raises(NotImplementedError):
        get_teacher_map("path", torch.rand(2, 1, 64, 64), torch.rand(2, 8, 2))


# --- Main Runner ---

def main():
    # Gather all functions starting with 'test_' defined in this scope
    test_functions = [
        func for name, func in globals().items() 
        if name.startswith("test_") and callable(func)
    ]

    print(f"Running {len(test_functions)} tests sequentially...\n")
    
    passed_count = 0
    failed_count = 0

    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed_count += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed_count += 1

    print("\n" + "="*30)
    print(f"Results: {passed_count} passed, {failed_count} failed.")
    print("="*30)

if __name__ == "__main__":
    main()