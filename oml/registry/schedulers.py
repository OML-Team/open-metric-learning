from typing import Any, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

from oml.utils.misc import TCfg

SCHEDULERS_REGISTRY = {
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CyclicLR": CyclicLR,
    "OneCycleLR": OneCycleLR,
}


def get_scheduler(name: str, optimizer: Optimizer, kwargs: Dict[str, Any]) -> _LRScheduler:
    return SCHEDULERS_REGISTRY[name](optimizer=optimizer, **kwargs)


def get_scheduler_by_cfg(cfg: TCfg, optimizer: Optimizer) -> _LRScheduler:
    return get_scheduler(name=cfg["name"], optimizer=optimizer, kwargs=cfg["args"])
