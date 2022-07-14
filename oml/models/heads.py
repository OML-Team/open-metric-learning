from pathlib import Path
from typing import Union

import torch
from torch import nn

from oml.interfaces.models import IHead


class SimpleLinearHead(IHead):
    def __init__(
        self,
        weights: Union[Path, str],
        in_features: int,
        num_classes: int,
        bias: bool,
        strict_load: bool,
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
        return self.fc(x)

    @property
    def num_classes(self) -> int:
        return self._num_classes
