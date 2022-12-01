from typing import Any, Dict

import torch.optim as opt

from oml.utils.misc import TCfg, dictconfig_to_dict

OPTIMIZERS_REGISTRY = {
    "adadelta": opt.Adadelta,
    "adagrad": opt.Adagrad,
    "adam": opt.Adam,
    "adamw": opt.AdamW,
    "adamax": opt.Adamax,
    "asgd": opt.ASGD,
    "lbfgs": opt.LBFGS,
    "rmsprop": opt.RMSprop,
    "rprop": opt.Rprop,
    "sgd": opt.SGD,
}


def get_optimizer(name: str, **kwargs: Dict[str, Any]) -> opt.Optimizer:
    return OPTIMIZERS_REGISTRY[name](**kwargs)


def get_optimizer_by_cfg(cfg: TCfg, **kwargs_runtime: Dict[str, Any]) -> opt.Optimizer:
    cfg = dictconfig_to_dict(cfg)
    cfg["args"].update(kwargs_runtime)
    return get_optimizer(name=cfg["name"], **cfg["args"])


__all__ = ["OPTIMIZERS_REGISTRY", "get_optimizer", "get_optimizer_by_cfg"]
