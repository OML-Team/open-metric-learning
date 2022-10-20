import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from oml.interfaces.criterions import ICriterion


class ArcFaceLoss(ICriterion):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        criterion: Optional[nn.Module] = None,
        label2category: Optional[Dict[Any, Any]] = None,
        label_smoothing: Optional[float] = None,
        m: float = 0.5,
        s: float = 64,
    ):
        super(ArcFaceLoss, self).__init__()

        if label2category is not None and label_smoothing is None:
            warnings.warn("Label smoothing in arcface head will not be used!")
        else:
            assert (
                label_smoothing is None or label2category is not None
            ), "You have to provide label2category to use label smoothing in arcface!"
        assert (
            label_smoothing is None or label_smoothing > 0 and label_smoothing < 1
        ), f"Choose another label_smoothing parametrization, got {label_smoothing}"

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.label2category = {} if label_smoothing is None else torch.arange(num_classes).apply_(label2category.get)
        self.label_smoothing = label_smoothing
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.rescale = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = -self.cos_m
        self.mm = self.sin_m * m
        self.last_logs: Dict[str, float] = {}

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = self.fc(x)

        self._log_accuracy_on_batch(cos, y)

        sin = torch.sqrt(1.0 - torch.pow(cos, 2))

        cos_w_margin = cos * self.cos_m - sin * self.sin_m
        cos_w_margin = torch.where(cos > self.th, cos_w_margin, cos - self.mm)

        ohe = F.one_hot(y, self.num_classes)
        logit = torch.where(ohe.bool(), cos_w_margin, cos) * self.rescale

        if self.label_smoothing:
            if self.label2category.device.type != self.weight.device.type:
                self.label2category = self.label2category.to(self.weight.device)
            ohe = ohe.float()
            ohe *= 1 - self.label_smoothing
            mask_l2c = self.label2category[y].tile(self.label2category.shape[0], 1).t() == self.label2category
            y = torch.where(mask_l2c, self.label_smoothing / mask_l2c.sum(-1).view(-1, 1), 0) + ohe

        return self.criterion(logit, y)

    def _log_accuracy_on_batch(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        accuracy = torch.mean((y == torch.argmax(logits, 1)).to(torch.float32))
        self.last_logs.update(
            {
                "accuracy": float(accuracy),
            }
        )


class ArcFaceLossWithMLP(ArcFaceLoss):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        mlp_features: List[int],
        criterion: Optional[nn.Module] = None,
        label2category: Optional[Dict[str, Any]] = None,
        label_smoothing: Optional[float] = None,
        m: float = 0.5,
        s: float = 64,
    ):
        super().__init__(mlp_features[-1], num_classes, criterion, label2category, label_smoothing, m, s)
        self.mlp = MLP(in_features, mlp_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return super().forward(self.mlp(x), y)
