from collections import OrderedDict
import time

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from torchsummary import summary

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
        dots = torch.einsum('... i d , ... j d -> ... i j', q, k).mul_(self.scale_factor)  # B H T T
        if mask is not None:
            dots = dots.masked_fill(mask==0, -1e11)
        attn = torch.softmax(dots, dim=-1)  # B H T T
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
        self.embd = embd
        self.blocks = nn.ModuleList([ResidualAttentionBlock(embd, dim, heads, ff_multi, dropout) for _ in range(layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x

class SimpleLM(nn.Module):
    def __init__(self, transformer, vocab_size, vocab_dim, tie_weights=False):
        super().__init__()
        self.transformer = transformer
        self.embedding = nn.Sequential()
        self.embedding.add_module("embedding", nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_dim))
        self.head = nn.Sequential()
        if transformer.embd != vocab_dim:
            self.embedding.add_module("enc_proj", nn.Linear(vocab_dim, transformer.embd, bias=False))
            self.head.add_module("dec_proj", nn.Linear(transformer.embd, vocab_dim, bias=False))
        self.head.add_module("out", nn.Linear(vocab_dim, vocab_size, bias=False))

        if tie_weights:
            for idx, (enc, dec)  in enumerate(zip(self.embedding, reversed(self.head))):
                if idx==0:
                    dec.weight = enc.weight
                else:
                    dec.weight.data = enc.weight.data.T

    def forward(self, token_ids, mask=None):
        x = self.embedding(token_ids)
        x = transformer(x, mask)
        logits = self.head(x)
        return logits

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_casual_mask = lambda x: torch.tril(torch.ones(x.shape[1], x.shape[1]))

    batch = 1
    tokens = 4

    layers = 6
    embd = 384
    dim = 32
    heads = 12
    ff_multi = 4
    dropout = 0.1
    
    # x = torch.rand(batch, tokens, embd)
    # print(f"{x.shape=}")
    # mask = gen_casual_mask(x)
    # mask = torch.rand(tokens, tokens) < 0.75
    # print(f"{mask=}")

    # msa = MSA(embd, dim, heads, dropout)
    # y_msa = msa(x, mask)
    # print(f"{y_msa.shape=}")

    # block = ResidualAttentionBlock(embd, dim, heads, ff_multi, dropout)
    # y_block = block(x, mask)
    # print(f"{y_block.shape=}")

    transformer = Transformer(layers, embd, dim, heads, ff_multi, dropout)
    # y_transformer = transformer(x, mask)
    # print(f"{y_transformer.shape=}")
    # summary(transformer, input_size=(1, embd), device="cpu")

    vocab_size = 50257
    vocab_dim = 64
    tie_weights = True

    token_ids = torch.randint(low=0, high=vocab_size, size=(1, 512)).to(device)
    lm = SimpleLM(transformer, vocab_size, vocab_dim, tie_weights)
    # logits = lm(token_ids)
    # print(f"{logits.shape=}")

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vocab_dims = [32, 64, 128, 256, 384, 512, 768, 1024]
    with torch.no_grad():
        for vocab_dim in vocab_dims:
            params = vocab_dim*transformer.embd if transformer.embd!=vocab_dim else 0
            params += vocab_dim*vocab_size
            lm = SimpleLM(transformer, vocab_size, vocab_dim, tie_weights).to(device)
            for i in range(10):
                out = lm(token_ids)
            n_iter = 35
            start = time.time()
            for i in range(n_iter):
                out = lm(token_ids)
            dt = time.time() - start
            print(f"{vocab_dim=}. embed params={params/1e6:.1f}M. {dt/n_iter * 1e3:.3f}ms.")

    # n_iter = 8
    # lm = lm.to(device)
    # for b in range(1, 1024, 1):
    #     x = torch.randint(low=0, high=vocab_size, size=(b, 512), device=device)
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     for _ in range(n_iter):
    #         lm(x)
    #     torch.cuda.synchronize()
    #     dt = time.time() - start
    #     throughput = b * n_iter / dt
    #     print(f"Batch: {b} \t {throughput:6.2f} samples/sec")
    #     del x
    #     torch.cuda.empty_cache()
