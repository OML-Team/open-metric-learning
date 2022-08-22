import numpy as np
import torch
from torch import nn


class GEM(nn.Module):
    def __init__(self, p: float, eps: float = 1e-6):
        """
        Generalised Mean Pooling (GEM)
        https://paperswithcode.com/method/generalized-mean-pooling

        Args:
            p: if p == 1 it's average pooling, if p == inf it's max-pooling
            eps: eps for numerical stability
        """
        super(GEM, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clip(x, min=self.eps, max=np.inf)
        x = torch.pow(x, self.p)

        bs, feat_dim, h, w = x.shape
        x = x.view(bs, feat_dim, h * w)

        x = x.mean(axis=-1)
        x = torch.pow(x, (1.0 / self.p))
        return x


__all__ = ["GEM"]
