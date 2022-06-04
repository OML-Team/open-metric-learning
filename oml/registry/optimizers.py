from typing import Any, Dict

import torch.optim as opt

from oml.utils.misc import TCfg, dictconfig_to_dict

OPTIMIZERS_REGISTRY = {
    "sgd": opt.SGD,
    "adam": opt.Adam,
}


def get_optimizer(name: str, kwargs: Dict[str, Any], params: Any) -> opt.Optimizer:
    return OPTIMIZERS_REGISTRY[name](params=params, **kwargs)


def get_optimizer_by_cfg(cfg: TCfg, params: Any) -> opt.Optimizer:
    cfg = dictconfig_to_dict(cfg)
    return get_optimizer(name=cfg["name"], kwargs=cfg["args"], params=params)
