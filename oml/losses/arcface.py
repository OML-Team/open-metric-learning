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
    ArcFace loss from `paper <https://arxiv.org/abs/1801.07698>`_ with possibility to use label smoothing.
    It contains projection (num_features x num_classes) inside itself so you don't have to produce output of
    ``num_classes`` yourself. Please make sure that class labels started with 0 and ended as ``num_classes - 1``.
    """

    criterion_name = "arcface"  # for better logging

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        label2category: Optional[Dict[Any, Any]] = None,
        label_smoothing: Optional[float] = None,
        m: float = 0.5,
        s: float = 64,
    ):
        """
        Args:
            in_features: Input feature size
            num_classes: Number of classes in train set
            label2category: Optional, for label smoothing. If you will not provide it label smoothing will be global and
                not category-wise
            label_smoothing: Label smoothing effect strength
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
        """
        super(ArcFaceLoss, self).__init__()

        assert (
            label_smoothing is None or 0 < label_smoothing < 1
        ), f"Choose another label_smoothing parametrization, got {label_smoothing}"

        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        if label2category is not None:
            mapper = {l: i for i, l in enumerate(sorted(list(set(label2category.values()))))}
            label2category = {k: mapper[v] for k, v in label2category.items()}
            self.label2category = torch.arange(num_classes).apply_(label2category.get)
        else:
            self.label2category = None
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
        if self.label2category is not None:
            self.label2category = self.label2category.to(self.weight.device)
        return label_smoothing(y, self.num_classes, self.label_smoothing, self.label2category)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert torch.all(y < self.num_classes), "You should provide labels between 0 and num_classes - 1."

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

    @torch.no_grad()
    def _log_accuracy_on_batch(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.last_logs[ACCURACY_KEY] = torch.mean((y == torch.argmax(logits, 1)).to(torch.float32))


class ArcFaceLossWithMLP(nn.Module):
    """
    Almost the same as ``ArcFaceLoss``, but also has MLP projector before the loss.
    We need this because all layers will be removed from loss at test time.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        mlp_features: List[int],
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
            label2category: Optional, for label smoothing. If you will not provide it label smoothing will be global and
                not category-wise
            label_smoothing: Label smoothing effect strength
            m: Margin parameter for ArcFace loss. Usually you should use 0.3-0.5 values for it
            s: Scaling parameter for ArcFace loss. Usually you should use 30-64 values for it
        """
        super().__init__()
        self.mlp = MLP(in_features, mlp_features)
        self.arcface = ArcFaceLoss(mlp_features[-1], num_classes, label2category, label_smoothing, m, s)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.arcface(self.mlp(x), y)
