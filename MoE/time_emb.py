
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_emb_dim):
        super().__init__()
        self.dim = time_emb_dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim -1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
