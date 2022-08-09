from typing import Any, Dict, Optional

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
        m: float = 0.2,
    ):
        super(ArcFaceLoss, self).__init__()

        assert (label_smoothing is None) == (label2category is None)

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.label2category = torch.arange(num_classes).apply_(label2category.get)
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        self.rescale = nn.Parameter(torch.ones(1, dtype=torch.float))
        self._eps = eps
        self.m = m

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2)).clip(-1 + self._eps, 1 - self._eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        fc = self.fc(x)
        # TODO: compute only arccos for target idxs
        # TODO: less compute with cos(fc) * cos(m) - sin(fc) * sin(m)
        fc_w_margin = torch.cos(torch.acos(fc) + self.m)

        ohe = torch.zeros_like(fc, device=self.weight.device)
        ohe.scatter_(1, y.view(-1, 1), 1)
        logit = torch.where(ohe > 0, fc_w_margin, fc)
        res = F.softmax(self.rescale * logit)

        if self.label_smoothing is not None:
            if self.label2category.device.type != self.weight.device.type:
                self.label2category = self.label2category.to(self.weight.device)
            ohe *= 1 - self.label_smoothing
            mask_l2c = self.label2category[y].tile(self.label2category.shape[0], 1).t() == self.label2category
            y = torch.where(mask_l2c, self.label_smoothing / mask_l2c.sum(-1).view(-1, 1), 0) + ohe

        return self.criterion(res, y)
