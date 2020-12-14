import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# classes

class Attention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context, mask = None):
        h, scale = self.heads, self.scale

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
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
        num_induced_points = None):
        super().__init__()
        self.induced_points = nn.Parameter(torch.randn(num_induced_points, dim)) if exists(num_induced_points) else None
        self.attn1 = Attention(dim, heads)
        self.attn2 = Attention(dim, heads)

    def forward(self, x, queries = None, mask = None):
        b, *_ = x.shape
        assert exists(queries) ^ exists(self.induced_points), 'you can only either learn the induced points within the module, or pass it in externally'
        queries = queries if exists(queries) else self.induced_points

        if queries.ndim == 2:
            queries = repeat(queries, 'n d -> b n d', b = b)

        induced = self.attn1(queries, x, mask = mask)
        out     = self.attn2(x, induced)
        return out, induced
