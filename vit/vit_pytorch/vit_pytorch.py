import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
import numpy as np

MIN_NUM_PATCHES = 8

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, num_patches=8, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        assert num_patches >= MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean', 'all'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool
        if self.pool != 'mean':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)) #torch.Size([1, 9, 1024])

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        
        self.to_latent = nn.Identity()
        


    def forward(self, x, mask=None, return_seq = False):
       
        
#         x = self.patch_to_embedding(X)
        b, n, _ = x.shape
        # print('b, n', b, n)

        
        if self.pool != 'mean':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            # print('cls_tokens', cls_tokens.shape)
            # print('x', x.shape)
            x = torch.cat((cls_tokens, x), dim=1)
            # print('after cls', x.shape)
            x += self.pos_embedding[:, :(n + 1)]
            # print('x, pos_embedding', x.shape, self.pos_embedding[:, :(n + 1)].shape)
        else:
            x += self.pos_embedding[:, :(n )]
            
        x = self.dropout(x)

        x_seq = self.transformer(x, mask)

        x = x_seq.mean(dim = 1) if self.pool == 'mean' else x_seq[:, 0]
        

        x = self.to_latent(x)
        if return_seq:
            return x, x_seq
        return x
