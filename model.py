from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np
import math

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class Similarity_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z_list, z_avg):
        z_sim = 0
        num_masked = len(z_list)
        
        for z in z_list:
            z_sim -= F.cosine_similarity(z, z_avg, dim=1).mean()
            
        z_sim = z_sim/num_masked
        
        return z_sim     

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self, z_list):
        loss = 0
        for X in z_list:
            loss -= self.compute_discrimn_loss(X.T)
        loss = loss/len(z_list)
        return loss

def posemb_sincos_1d(patches, temperature = 10000, dtype = torch.float32):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device = device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim = 1)
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class MTS_LOF(nn.Module):
    def __init__(self, configs):
        super(MTS_LOF, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(configs.input_channels, 48, kernel_size=configs.kernel_size,
                      stride=configs.stride, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(48),
            nn.GELU(),
            
            nn.Conv1d(48, 96, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(96),
            nn.GELU(),
            
            nn.Conv1d(96, 192, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(192),
            nn.GELU(),
            
            nn.Conv1d(192, 384, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(384),
            nn.GELU(),
            
            nn.Conv1d(384, configs.embed_dim, 1, 1, 0),
            
            Rearrange('b c p -> b p c'),
        )
        
        self.transformer_encoder = Transformer(configs.embed_dim, depth = 6, heads = 8, dim_head=configs.embed_dim//8, mlp_dim=configs.embed_dim*4)
        
        self.linear = nn.Linear(configs.embed_dim, configs.num_classes)

        self.inv_loss = Similarity_Loss()
        self.tcr_loss = TotalCodingRate(eps=0.2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, configs.embed_dim))
        self.decoder = Transformer(configs.embed_dim, depth = 4, heads = 8, dim_head=configs.embed_dim//8, mlp_dim=configs.embed_dim*4)

    def forward(self, x_in):
        x = self.conv_block(x_in)
        b, n, _ = x.shape

        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer_encoder(x).mean(dim=1)

        rep = x.detach()

        return self.linear(x), rep

    def supervised_train_forward(self, x, y, criterion=nn.CrossEntropyLoss()):
        pred, _ = self.forward(x)
        loss = criterion(pred, y)
        return loss, pred.detach()

    def ssl_train_forward(self, x, mask_ratio=0.8, num_masked=20):
        x = self.conv_block(x)
        b, n, _ = x.shape

        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        z_avg = self.transformer_encoder(x).mean(dim=1)
        z_avg = F.normalize(z_avg, p=2)

        z_list = []

        for _ in range(num_masked):
            z, mask, ids_restore = self.random_masking(x, mask_ratio)
            z = self.transformer_encoder(z)

            mask_tokens = self.mask_token.repeat(z.shape[0], ids_restore.shape[1] - z.shape[1], 1)
            z = torch.cat([z, mask_tokens], dim=1)
            z = torch.gather(z, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, z.shape[2]))  # unshuffle

            pe = posemb_sincos_1d(z)
            z = rearrange(z, 'b ... d -> b (...) d') + pe

            z = self.decoder(z).mean(dim=1)
            z = F.normalize(z, p=2)
            z_list.append(z)

        contrastive_loss = 100 * self.inv_loss(z_list, z_avg)
        diversity_loss = self.tcr_loss(z_list)
        loss = contrastive_loss + diversity_loss

        return loss, [contrastive_loss.item(), diversity_loss.item()]

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
