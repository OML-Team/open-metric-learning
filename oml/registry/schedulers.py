from typing import Any, Dict

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

from oml.utils.misc import TCfg, dictconfig_to_dict

SCHEDULERS_REGISTRY = {
    "lambda": LambdaLR,
    "multiplicative": MultiplicativeLR,
    "step": StepLR,
    "multi_step": MultiStepLR,
    "exponential": ExponentialLR,
    "cosine_annealing": CosineAnnealingLR,
    "cyclic": CyclicLR,
    "one_cycle": OneCycleLR,
}


def get_scheduler(name: str, **kwargs: Dict[str, Any]) -> _LRScheduler:
    return SCHEDULERS_REGISTRY[name](**kwargs)


def get_scheduler_by_cfg(cfg: TCfg, **kwargs_runtime: Dict[str, Any]) -> _LRScheduler:
    cfg = dictconfig_to_dict(cfg)
    cfg["args"].update(kwargs_runtime)
    return get_scheduler(name=cfg["name"], **cfg["args"])


__all__ = ["SCHEDULERS_REGISTRY", "get_scheduler", "get_scheduler_by_cfg"]
