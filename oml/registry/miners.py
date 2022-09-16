from typing import Any, Dict

from oml.interfaces.miners import ITripletsMiner
from oml.miners.cross_batch import TripletMinerWithMemory
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.miners.inbatch_hard_cluster import HardClusterMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.utils.misc import TCfg, dictconfig_to_dict

MINERS_REGISTRY = {
    "all_triplets": AllTripletsMiner,
    "hard_cluster": HardClusterMiner,
    "hard_triplets": HardTripletsMiner,
    "triplets_with_memory": TripletMinerWithMemory,
}


def get_miner(name: str, **kwargs: Dict[str, Any]) -> ITripletsMiner:
    return MINERS_REGISTRY[name](**kwargs)


def get_miner_by_cfg(cfg: TCfg) -> ITripletsMiner:
    cfg = dictconfig_to_dict(cfg)
    return get_miner(name=cfg["name"], **cfg["args"])


__all__ = ["MINERS_REGISTRY", "get_miner", "get_miner_by_cfg"]
