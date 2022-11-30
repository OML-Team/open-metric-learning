from typing import Any, Dict

from torch import nn

from oml.losses.surrogate_precision import SurrogatePrecision
from oml.losses.triplet import TripletLoss, TripletLossPlain, TripletLossWithMiner
from oml.registry.miners import get_miner_by_cfg
from oml.utils.misc import TCfg, dictconfig_to_dict

LOSSES_REGISTRY = {
    "triplet": TripletLoss,
    "triplet_plain": TripletLossPlain,
    "triplet_with_miner": TripletLossWithMiner,
    "surrogate_precision": SurrogatePrecision
}


def get_criterion(name: str, **kwargs: Dict[str, Any]) -> nn.Module:
    if "miner" in kwargs:
        miner = get_miner_by_cfg(kwargs.pop("miner"))
        return LOSSES_REGISTRY[name](miner=miner, **kwargs)
    else:
        return LOSSES_REGISTRY[name](**kwargs)


def get_criterion_by_cfg(cfg: TCfg, **kwargs_runtime: Dict[str, Any]) -> nn.Module:
    cfg = dictconfig_to_dict(cfg)
    cfg.setdefault("args", {})
    cfg["args"].update(**kwargs_runtime)
    return get_criterion(name=cfg["name"], **cfg["args"])


__all__ = ["LOSSES_REGISTRY", "get_criterion", "get_criterion_by_cfg"]
