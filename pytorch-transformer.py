from collections import OrderedDict

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

class MSA(torch.nn.Module):
    def __init__(self, embd, dim, heads, dropout=0):
        super(MSA, self).__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(embd, 3*dim*heads, bias=False)
        self.attn_out = nn.Linear(dim*heads, embd, bias=False)
        self.scale_factor = dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.to_qkv.weight, std=0.02)
        nn.init.normal_(self.attn_out.weight, std=0.02)

    def forward(self, x, mask=None):
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L186
        qkv = self.to_qkv(x)  # B T 3*D*H
        qkv = rearrange(qkv, 'b t (k d h) -> k b h t d', k=3, h=self.heads)  # 3 B H T D
        q, k, v = qkv.unbind(0)  # B H T D
        dots = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor  # B H T T
        attn = torch.softmax(dots, dim=-1)  # B H T T
        print(f"{attn=}")
        # if mask is not None:
            # if mask.dim()==2:
            #     mask = mask.unsqueeze(0)  # Expand head dimension
            # scores = scores.masked_fill(mask == 0, -1e9)
            # if mask.dtype == torch.bool:
            #     attn_output_weights.masked_fill_(mask, float('-inf'))
            # else:
            #     attn_output_weights += mask
        out = torch.einsum('... i j , ... j d -> ... i d', attn, v)  # B H T D
        out = rearrange(out, 'b h t d -> b t (h d)')  # B T D*H
        out = self.dropout(out)
        out =  self.attn_out(out)  # B T E
        return out

class MLP(torch.nn.Module):
    def __init__(self, embd, ff_multi, dropout=0):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(embd, embd*ff_multi)
        self.gelu = nn.GELU()
        self.out_proj = nn.Linear(embd*ff_multi, embd)
        nn.init.normal_(self.in_proj.weight, std=0.01)
        nn.init.normal_(self.out_proj.weight, std=0.01)

    def forward(self, x):
        x = self.dropout(x)
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, embd, dim, heads=0, ff_multi=0, dropout=0):
        super(ResidualAttentionBlock, self).__init__()
        self.ln = nn.LayerNorm(embd)
        self.dropout = nn.Dropout(dropout)
        self.msa = MSA(embd, dim, heads, dropout) if heads>0 else None
        self.mlp = MLP(embd, ff_multi, dropout) if ff_multi>0 else None

    def forward(self, x, mask=None):
        norm = self.ln(x)
        if self.mlp is not None:
            x = x + self.msa(norm, mask)
        if self.mlp is not None:
            x = x + self.mlp(norm)
        x = self.dropout(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, layers, embd, dim, heads=0, ff_multi=0, dropout=0):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([ResidualAttentionBlock(embd, dim, heads, ff_multi, dropout) for _ in range(layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

if __name__ == "__main__":

    batch = 1
    tokens = 4

    layers = 16
    embd = 1024
    dim = 64
    heads = 1
    ff_multi = 4
    dropout = 0.1
    
    x = torch.rand(batch, tokens, embd)
    print(f"{x.shape=}")

    msa = MSA(embd, dim, heads, dropout)
    mask = None
    # b, l, e = x.size()
    # mask = torch.tril(torch.ones(l, l))
    # print(f"{mask=}")
    # .view(1, l, l).repeat(b, 1, 1)
    y_msa = msa(x, mask)
    print(f"{y_msa.shape=}")

    # block = ResidualAttentionBlock(embd, dim, heads, ff_multi, dropout)
    # y_block = block(x)
    # print(f"{y_block.shape=}")

    # transformer = Transformer(layers, embd, dim, heads, ff_multi, dropout)
    # y_transformer = transformer(x)
    # print(f"{y_transformer.shape=}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # from torchsummary import summary
    # summary(transformer.to(device), input_size=(1, embd))
