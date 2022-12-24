import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context,
        mask = None
    ):
        h, scale = self.heads, self.scale

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b 1 1 n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class ISAB(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        num_latents = None
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim)) if exists(num_latents) else None
        self.attn1 = Attention(dim, heads)
        self.attn2 = Attention(dim, heads)

    def forward(self, x, latents = None, mask = None):
        b, *_ = x.shape
        assert exists(latents) ^ exists(self.latents), 'you can only either learn the induced points within the module, or pass it in externally'
        latents = latents if exists(latents) else self.latents

        if latents.ndim == 2:
            latents = repeat(latents, 'n d -> b n d', b = b)

        latents = self.attn1(latents, x, mask = mask)
        out     = self.attn2(x, latents)

        return out, latents
