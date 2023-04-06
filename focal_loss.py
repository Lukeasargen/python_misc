import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.losses import FocalLoss  # pip install kornia

class MyFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inp, targ):
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

if __name__ == "__main__":

    batch = 3
    classes = 3

    logits = torch.randn(batch, classes)
    labels = torch.randint(low=0, high=classes, size=(batch,))
    print(f"{logits=}")
    print(f"{labels=}")

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_ce = ce_loss_fn(logits, labels)
    print(f"{loss_ce=}")

    focal_loss_fn = MyFocalLoss(reduction="none")
    loss_focal = focal_loss_fn(logits, labels)
    print(f"{loss_focal=}")

    focal_loss_kornia = FocalLoss(alpha=1, gamma=2, reduction="none")
    loss_focal_kornia = focal_loss_kornia(logits, labels)
    print(f"{loss_focal_kornia=}")
