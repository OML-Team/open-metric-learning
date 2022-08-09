from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from oml.interfaces.models import IHead


class SimpleLinearHead(IHead):
    def __init__(
        self,
        weights: Union[Path, str],
        in_features: int,
        num_classes: int,
        bias: bool = True,
        strict_load: bool = False,
    ):
        super(SimpleLinearHead, self).__init__()

        self._num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

        if weights == "random":
            return
        else:
            state_dict = torch.load(weights, map_location="cpu")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict
        self.fc.load_state_dict(state_dict, strict=strict_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.fc(x))

    @property
    def num_classes(self) -> int:
        return self._num_classes


class ArcFaceHead(IHead):
    def __init__(
        self,
        weights: Union[Path, str],
        in_features: int,
        num_classes: int,
        margin: float = 0.1,
        s: float = 16,
        eps: float = 1e-8,
    ):
        super(ArcFaceHead, self).__init__()

        self._num_classes = num_classes
        self._eps = eps
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        self.rescale = nn.Parameter(torch.ones(1, dtype=torch.float) * s)

        if weights == "random":
            nn.init.xavier_uniform_(self.weight)
            return
        else:
            state_dict = torch.load(weights, map_location="cpu")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict
        self.weight.load_state_dict(state_dict)

    def fc(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(x, p=2), F.normalize(self.weight, p=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angle = torch.acos(self.fc(x).clip(-1 + self._eps, 1 - self._eps))
        logit = torch.cos(angle + self.margin) * self.rescale
        return F.softmax(logit)

    @property
    def num_classes(self) -> int:
        return self._num_classes


class ArcFaceFancy(IHead):
    def __init__(
        self,
        weights: Union[Path, str],
        in_features: int,
        num_classes: int,
        inter_scale: float = 1.5,
        margin: float = 0.1,
        eps: float = 1e-8,
    ):
        super(ArcFaceFancy, self).__init__()

        intermidiate_fsize = int(in_features * inter_scale)

        self.linear = nn.Linear(in_features, intermidiate_fsize)
        self.mish = nn.Mish()
        self.arcface = ArcFaceHead(
            weights="random",
            in_features=intermidiate_fsize,
            num_classes=num_classes,
            margin=margin,
            eps=eps,
        )

        if weights == "random":
            return
        else:
            state_dict = torch.load(weights, map_location="cpu")

        state_dict = state_dict["state_dict"] if "state_dict" in state_dict.keys() else state_dict
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.arcface(self.mish(self.linear(x)))
