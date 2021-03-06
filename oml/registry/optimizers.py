from typing import Any, Dict

import torch.optim as opt

from oml.utils.misc import TCfg, dictconfig_to_dict

OPTIMIZERS_REGISTRY = {
    "sgd": opt.SGD,
    "adam": opt.Adam,
}


def get_optimizer(name: str, **kwargs: Dict[str, Any]) -> opt.Optimizer:
    return OPTIMIZERS_REGISTRY[name](**kwargs)


def get_optimizer_by_cfg(cfg: TCfg, **kwargs_runtime: Dict[str, Any]) -> opt.Optimizer:
    cfg = dictconfig_to_dict(cfg)
    cfg["args"].update(kwargs_runtime)
    return get_optimizer(name=cfg["name"], **cfg["args"])
