from typing import Any, Dict

from oml.interfaces.miners import ITripletsMiner
from oml.miners import TripletMinerWithMemory
from oml.miners import AllTripletsMiner
from oml.miners import HardClusterMiner
from oml.miners import HardTripletsMiner
from oml.miners import NHardTripletsMiner
from oml.miners import MinerWithBank
from oml.utils.misc import TCfg, dictconfig_to_dict

MINERS_REGISTRY = {
    "all_triplets": AllTripletsMiner,
    "hard_cluster": HardClusterMiner,
    "hard_triplets": HardTripletsMiner,
    "triplets_with_memory": TripletMinerWithMemory,
    "miner_with_bank": MinerWithBank,
    "n_hard_triplets": NHardTripletsMiner,
}


def get_miner(name: str, **kwargs: Dict[str, Any]) -> ITripletsMiner:
    if "miner" in kwargs:
        miner = get_miner_by_cfg(kwargs["miner"].copy())
        del kwargs["miner"]
        return MINERS_REGISTRY[name](miner=miner, **kwargs)
    else:
        return MINERS_REGISTRY[name](**kwargs)


def get_miner_by_cfg(cfg: TCfg) -> ITripletsMiner:
    cfg = dictconfig_to_dict(cfg)
    return get_miner(name=cfg["name"], **cfg["args"])


__all__ = ["MINERS_REGISTRY", "get_miner", "get_miner_by_cfg"]
