from typing import Any, Dict, Optional

from torch import nn

from oml.losses.arcface import ArcFaceLoss
from oml.losses.triplet import TripletLoss, TripletLossPlain, TripletLossWithMiner
from oml.registry.miners import get_miner_by_cfg
from oml.utils.misc import TCfg, dictconfig_to_dict

LOSSES_REGISTRY = {
    "triplet": TripletLoss,
    "triplet_plain": TripletLossPlain,
    "triplet_with_miner": TripletLossWithMiner,
    "ce": nn.CrossEntropyLoss,
    "arcface": ArcFaceLoss,
}


def get_criterion(name: str, **kwargs: Dict[str, Any]) -> nn.Module:
    if "miner" in kwargs:
        miner = get_miner_by_cfg(kwargs["miner"].copy())
        del kwargs["miner"]
        return LOSSES_REGISTRY[name](miner=miner, **kwargs)
    else:
        return LOSSES_REGISTRY[name](**kwargs)


def get_criterion_by_cfg(
    cfg: TCfg,
    in_features: Optional[int] = None,
    num_classes: Optional[int] = None,
    label2category: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    cfg = dictconfig_to_dict(cfg)
    cfg.setdefault("args", {})
    if cfg["name"] == "arcface":
        cfg["args"]["in_features"] = in_features
        cfg["args"]["num_classes"] = num_classes
        if "label_smoothing" in cfg["args"]:
            cfg["args"]["label2category"] = label2category

    return get_criterion(name=cfg["name"], **cfg["args"])


__all__ = ["LOSSES_REGISTRY", "get_criterion", "get_criterion_by_cfg"]
