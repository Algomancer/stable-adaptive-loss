import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(bias):
    out = bias.expm1().clamp_min(1e-6).log()
    return out

def affine_softplus(x, lo=0, ref=1):
  shift = inv_softplus(torch.ones_like(x))
  y = (ref - lo) * F.softplus(x + shift) + lo
  return y

def logit(y):
    return -torch.log(1 / y - 1)

def affine_sigmoid(logits, lo=0.0, hi=1.0):
    return torch.sigmoid(logits) * (hi - lo) + lo

def inv_affine_sigmoid(probs, lo=0.0, hi=1.0):
    return logit((probs - lo) / (hi - lo))

class AdaptiveLoss(nn.Module):
    def __init__(
        self,
        alpha_lo=0.001,
        alpha_hi=1.999,
        alpha_init=None,
        scale_lo=1e-5,
        scale_init=1.0,
        eps=1e-6,
        reduction='mean'
    ):
        super().__init__()
        
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.scale_lo = scale_lo
        self.eps = eps
        self.reduction = reduction
        
        # If alpha_init is not provided, default to (alpha_lo + alpha_hi) / 2.
        if alpha_init is None:
            alpha_init = (alpha_lo + alpha_hi) / 2.

        self.scale_init = scale_init
        self.alpha = nn.Parameter(
            inv_affine_sigmoid(
                torch.tensor([alpha_init], dtype=torch.bfloat16),
                lo=alpha_lo,
                hi=alpha_hi
            )
        )
        self.scale = nn.Parameter(torch.tensor([0.0], dtype=torch.bfloat16))

    @torch.compile()
    def forward(self, pred, target):
        x = pred - target
        eps = self.eps

        scale = affine_softplus(self.scale, lo=self.scale_lo, ref=self.scale_init)
        alpha = affine_sigmoid(self.alpha, lo=self.alpha_lo, hi=self.alpha_hi)
        b = torch.abs(alpha - 2.0) + eps
        d = torch.where(alpha >= 0, alpha + eps, alpha - eps)
        loss = (b / d) * (torch.pow((x / scale)**2 / b + 1.0, 0.5 * d) - 1.0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
