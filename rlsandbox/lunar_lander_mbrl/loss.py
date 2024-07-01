import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
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


class FocalLoss(nn.Module):
    """
    Focal Loss.

    Implementation copied from:

    https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
