import random
from typing import Any, Dict, Iterable, List, Union

import dotenv
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from oml.const import DOTENV_PATH


def find_value_ids(it: Iterable[Any], value: Any) -> List[int]:
    """
    Args:
        it: List of any
        value: Query element

    Returns:
        Indices of the all elements equal to x0
    """
    if isinstance(it, np.ndarray):
        inds = list(np.where(it == value)[0])
    else:  # could be very slow
        inds = [i for i, el in enumerate(it) if el == value]
    return inds


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        pass
    else:
        xm.set_rng_state(seed)


def one_hot(i: int, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim)
    vector[i] = 1
    return vector


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items = []  # type: ignore
    for k, v in d.items():
        new_key = str(parent_key) + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_dotenv() -> None:
    dotenv.load_dotenv(DOTENV_PATH)


TCfg = Union[Dict[str, Any], DictConfig]


def dictconfig_to_dict(cfg: TCfg) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    ret = dict()

    for k in cfg.keys():

        if isinstance(cfg[k], DictConfig) or isinstance(cfg[k], dict):
            ret[k] = dictconfig_to_dict(cfg[k])
        else:
            ret[k] = cfg[k]

    return ret
