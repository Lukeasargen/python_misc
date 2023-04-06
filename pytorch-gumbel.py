import torch
from torch import nn
from torch.nn import functional as F


def gumbel_distribution(shape, eps=1e-11, device=None):
    u = torch.rand(*shape, device=device)
    return -torch.log(-torch.log(u + eps))


class GumbelSoftmax(nn.Module):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()

    def forward(self, logits, temperature=1.0, hard_sample=False):
        log_probs = F.log_softmax(logits, dim=-1)
        y = log_probs + gumbel_distribution(shape=logits.size(), device=logits.device)
        y_soft = F.softmax(y/temperature, dim=-1)

        if not hard_sample:
            return y_soft

        _, idx = y_soft.max(dim=-1)
        y_hard = torch.zeros_like(y_soft).view(-1, y_soft.size(-1))
        y_hard.scatter_(1, idx.view(-1, 1), 1)
        y_hard = y_hard.view(y_soft.size())
        # Straight-through https://pytorch.org/docs/1.12/generated/torch.nn.functional.gumbel_softmax.html
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard
