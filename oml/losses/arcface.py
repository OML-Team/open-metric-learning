import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from oml.const import ACCURACY_KEY
from oml.utils.misc_torch import label_smoothing


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss from paper https://arxiv.org/abs/1801.07698 with possibility to use label smoothing.
    It contains projection (num_features x num_classes) inside itself so you don't have to produce output of
    ``num_classes`` yourself.
    """

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
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            criterion: Something like CrossEntropyLoss for margin-adjusted OHE
            label2category: Optional, for label smoothing. If you will not provide it label smoothing will be global and
                not category-wise
            label_smoothing: Label smoothing effect strength
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
        """
        super(ArcFaceLoss, self).__init__()

        assert label2category is None or label_smoothing is not None, "You should provide `label_smoothing`!"
        assert (
            label_smoothing is None or label_smoothing > 0 and label_smoothing < 1
        ), f"Choose another label_smoothing parametrization, got {label_smoothing}"

        self.criterion = criterion or nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.label2category = {} if label2category is None else torch.arange(num_classes).apply_(label2category.get)
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

    def smooth_labels(self, y: torch.Tensor) -> torch.Tensor:
        if isinstance(self.label2category, torch.Tensor):
            if self.label2category.device.type != self.weight.device.type:
                self.label2category = self.label2category.to(self.weight.device)
        return label_smoothing(y, self.num_classes, self.label_smoothing, self.label2category)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = self.fc(x)

        self._log_accuracy_on_batch(cos, y)

        sin = torch.sqrt(1.0 - torch.pow(cos, 2))

        cos_w_margin = cos * self.cos_m - sin * self.sin_m
        cos_w_margin = torch.where(cos > self.th, cos_w_margin, cos - self.mm)

        ohe = F.one_hot(y, self.num_classes)
        logit = torch.where(ohe.bool(), cos_w_margin, cos) * self.rescale

        if self.label_smoothing:
            y = self.smooth_labels(y)

        return self.criterion(logit, y)

    def _log_accuracy_on_batch(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        accuracy = torch.mean((y == torch.argmax(logits, 1)).to(torch.float32))
        self.last_logs.update(
            {
                ACCURACY_KEY: float(accuracy),
            }
        )


class ArcFaceLossWithMLP(nn.Module):
    """
    Almost the same as ``ArcFaceLoss``, but also has MLP projector before the loss.
    """

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
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            mlp_features: Layers sizes for MLP before ArcFace
            criterion: Something like CrossEntropyLoss for margin-adjusted OHE
            label2category: Optional, for label smoothing. If you will not provide it label smoothing will be global and
                not category-wise
            label_smoothing: Label smoothing effect strength
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
        """
        super().__init__()
        self.mlp = MLP(in_features, mlp_features)
        self.arcface = ArcFaceLoss(mlp_features[-1], num_classes, criterion, label2category, label_smoothing, m, s)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.arcface(self.mlp(x), y)
