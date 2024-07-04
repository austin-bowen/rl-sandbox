import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss


class NormalizedIfwBceWithLogitsLoss(BCEWithLogitsLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super().forward(input, target)

        total_count = target.size(0)
        pos_count = target.sum()
        neg_count = total_count - pos_count

        if pos_count == 0 or neg_count == 0:
            return loss

        pos_weight = total_count / pos_count
        neg_weight = total_count / neg_count

        weighted_loss = loss * (target * pos_weight + (1 - target) * neg_weight)

        total_orig_loss = loss.sum()
        total_weighted_loss = weighted_loss.sum()

        return weighted_loss * total_orig_loss / total_weighted_loss


class BCEWithLogitsFocalLoss(nn.Module):
    """
    Focal Loss.

    Implementation copied and modified from:

    https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py
    """

    def __init__(self, gamma: float = 0., alpha: float = None, reduction='mean'):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        losses = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        input_p = F.sigmoid(input)
        pt = input_p * target + (1 - input_p) * (1 - target)

        loss = (1 - pt) ** self.gamma * losses

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}')
