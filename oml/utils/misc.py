import inspect
import os
import random
from typing import Any, Dict, Iterable, List, Tuple

import dotenv
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from oml.const import DOTENV_PATH, TCfg


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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PL_SEED_WORKERS"] = str(1)

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


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/", ignored_keys: Iterable[str] = ()
) -> Dict[str, Any]:
    items = []  # type: ignore
    for k, v in d.items():
        if k in ignored_keys:
            continue
        new_key = str(parent_key) + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep, ignored_keys=ignored_keys).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_dotenv() -> None:
    dotenv.load_dotenv(DOTENV_PATH)


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


def smart_sample(array: List[Any], k: int) -> List[Any]:
    """Sample n_samples items from given list. If array contains at least n_samples items, sample without repetition;
    otherwise take all the unique items and sample n_samples - len(array) ones with repetition.

    Args:
        array: list of unique elements to sample from
        k: number of items to sample

    Returns:
        sampled_items: list of sampled items
    """
    array_size = len(array)
    if array_size < k:
        sampled = (
            np.random.choice(array, size=array_size, replace=False).tolist()
            + np.random.choice(array, size=k - array_size, replace=True).tolist()
        )
    else:
        sampled = np.random.choice(array, size=k, replace=False).tolist()
    return sampled


def clip_max(arr: Tuple[int, ...], max_el: int) -> Tuple[int, ...]:
    return tuple(min(x, max_el) for x in arr)


def remove_unused_kargs(kwargs: Dict[str, Any], constructor: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in inspect.signature(constructor).parameters}


__all__ = [
    "find_value_ids",
    "set_global_seed",
    "one_hot",
    "flatten_dict",
    "load_dotenv",
    "dictconfig_to_dict",
    "smart_sample",
    "clip_max",
]
