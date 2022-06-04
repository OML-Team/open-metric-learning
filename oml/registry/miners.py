from typing import Any, Dict

from oml.interfaces.miners import ITripletsMiner
from oml.miners.among_batches import TripletMinerWithMemory
from oml.miners.inbatch import AllTripletsMiner, HardClusterMiner, HardTripletsMiner
from oml.utils.misc import TCfg, dictconfig_to_dict

MINERS_REGISTRY = {
    "AllTripletsMiner": AllTripletsMiner,
    "HardClusterMiner": HardClusterMiner,
    "HardTripletsMiner": HardTripletsMiner,
    "TripletMinerWithMemory": TripletMinerWithMemory,
}


def get_miner(name: str, kwargs: Dict[str, Any]) -> ITripletsMiner:
    return MINERS_REGISTRY[name](**kwargs)


def get_miner_by_cfg(cfg: TCfg) -> ITripletsMiner:
    cfg = dictconfig_to_dict(cfg)
    return get_miner(name=cfg["name"], kwargs=cfg["args"])
