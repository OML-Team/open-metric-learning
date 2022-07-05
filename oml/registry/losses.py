from typing import Any, Dict, Optional

from torch import nn

from oml.losses.triplet import TripletLoss, TripletLossPlain, TripletLossWithMiner
from oml.registry.miners import get_miner_by_cfg
from oml.utils.misc import TCfg, dictconfig_to_dict

LOSSES_REGISTRY = {
    "TripletLoss": TripletLoss,
    "TripletLossPlain": TripletLossPlain,
    "TripletLossWithMiner": TripletLossWithMiner,
}


def get_criterion(name: str, kwargs: Dict[str, Any], miner_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    if miner_cfg is None:
        return LOSSES_REGISTRY[name](**kwargs)
    else:
        miner = get_miner_by_cfg(miner_cfg)
        return LOSSES_REGISTRY[name](miner=miner, **kwargs)


def get_criterion_by_cfg(cfg: TCfg) -> nn.Module:
    cfg = dictconfig_to_dict(cfg)

    if "miner" in cfg["args"].keys():
        miner_cfg = cfg["args"]["miner"].copy()
        del cfg["args"]["miner"]
    else:
        miner_cfg = None

    return get_criterion(name=cfg["name"], kwargs=cfg["args"], miner_cfg=miner_cfg)
