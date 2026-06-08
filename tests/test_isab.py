import torch
import pytest
from isab_pytorch import ISAB

@pytest.mark.parametrize('inverted_attention', (False, True))
def test_isab_with_learned_latents(inverted_attention):
    attn = ISAB(
        dim = 512,
        heads = 8,
        num_latents = 128,
        latent_self_attend = True,
        inverted_attention = inverted_attention
    )

    seq = torch.randn(2, 1024, 512)
    mask = torch.ones((2, 1024)).bool()

    out, latents = attn(seq, mask = mask)

    assert out.shape == seq.shape
    assert latents.shape == (2, 128, 512)

@pytest.mark.parametrize('inverted_attention', (False, True))
def test_isab_with_external_latents(inverted_attention):
    attn = ISAB(
        dim = 512,
        heads = 8,
        inverted_attention = inverted_attention
    )

    seq = torch.randn(2, 1024, 512)
    latents = torch.randn(128, 512)

    out, new_latents = attn(seq, latents)

    assert out.shape == seq.shape
    assert new_latents.shape == (2, 128, 512)
