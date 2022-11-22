import torch

batch = 1
heads = 1
q_len = 512
kv_len = 256
qk_dim = 4
v_dim = 16

q = torch.rand(batch, heads, q_len, qk_dim)
k = torch.rand(batch, heads, kv_len, qk_dim)
v = torch.rand(batch, heads, kv_len, v_dim)
print(f"{q.shape=} {k.shape=} {v.shape=}")

# Regular attention
# (q * k.T) * v
# softmax( q * k.T / sqrt(qk_dim) ) * v
print("REGULAR ATTENTION")
dots = torch.einsum('... i d , ... j d -> ... i j', q, k).mul_(qk_dim ** -0.5)
# print(f"{dots.shape=}")  # batch heads q_len kv_len
# print(f"{dots=}")
attn = torch.softmax(dots, dim=-1)
print(f"{attn.shape=}")  # batch heads q_len kv_len
# print(f"{attn=}")
out = torch.einsum('... i j , ... j d -> ... i d', attn, v)
print(f"{out.shape=}")  # batch heads kv_len v_dim
# print(f"{out=}")

# New attention
# q * (k.T * v)
# row_softmax(q) * col_softmax(k.T * v)
print("NEW ATTENTION")
dots = torch.einsum('... i d , ... i p -> ... d p', k, v).mul_(v_dim ** -0.5)
# print(f"{dots.shape=}")  # batch heads qk_dim v_dim
# print(f"{dots=}")
attn = torch.softmax(dots, dim=-2)
print(f"{attn.shape=}")  # batch heads qk_dim v_dim
# print(f"{attn=}")
out = torch.einsum('... i d , ... d p -> ... i p', torch.softmax(q, dim=-1), attn)
print(f"{out.shape=}")  # batch heads kv_len v_dim
# print(f"{out=}")
