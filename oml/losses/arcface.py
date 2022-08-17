from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        criterion: Optional[nn.Module] = None,
        label2category: Optional[Dict[str, Any]] = None,
        label_smoothing: Optional[float] = None,
        eps: float = 1e-8,
        m: float = 0.5,
        s: float = 64,
        use_weight_fix: bool = False,
    ):
        super(ArcFaceLoss, self).__init__()

        assert (label_smoothing is None) == (label2category is None)

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.label2category = torch.arange(num_classes).apply_(label2category.get)
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        self.rescale = nn.Parameter(torch.ones(1, dtype=torch.float) * s)
        self._eps = eps
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = -self.cos_m
        self.mm = self.sin_m * m
        self.use_weight_fix = use_weight_fix

    def renormalize(self) -> None:
        self.weight.data = F.normalize(self.weight.data, p=2)

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_weight_fix:
            qmax = torch.quantile(self.weight.data, 0.99)
            qmin = torch.quantile(self.weight.data, 0.01)
            ad = (qmax - qmin) / 98
            return F.linear(F.normalize(x, p=2), F.normalize(self.weight.clip(qmin - ad, qmax + ad), p=2)).clip(
                -1 + self._eps, 1 - self._eps
            )
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2)).clip(-1 + self._eps, 1 - self._eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = self.fc(x)
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))

        cos_w_margin = cos * self.cos_m - sin * self.sin_m
        cos_w_margin = torch.where(cos > self.th, cos_w_margin, cos - self.mm)

        ohe = torch.zeros_like(cos, device=x.device)
        ohe.scatter_(1, y.view(-1, 1), 1)

        logit = torch.where(ohe > 0, cos_w_margin, cos) * self.rescale

        if self.label_smoothing:
            if self.label2category.device.type != self.weight.device.type:
                self.label2category = self.label2category.to(self.weight.device)
            ohe *= 1 - self.label_smoothing
            mask_l2c = self.label2category[y].tile(self.label2category.shape[0], 1).t() == self.label2category
            y = torch.where(mask_l2c, self.label_smoothing / mask_l2c.sum(-1).view(-1, 1), 0) + ohe

        return self.criterion(logit, y)
