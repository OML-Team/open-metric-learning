import inspect
import os
import random
from typing import Any, Dict, Hashable, Iterable, List, Sequence, Tuple, Union

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


def remove_unused_kwargs(kwargs: Dict[str, Any], constructor: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in inspect.signature(constructor).parameters}


def check_if_nonempty_positive_integers(var: Union[int, Sequence[int]], name: str) -> None:
    """
    Check whether ``var`` is a positive integer or a non-empty Iterable of positive integers.

    Args:
        var: A sequence.
        name: A name of the sequence in case of exception should be raised.

    """
    if isinstance(var, Sequence):
        if not len(var) > 0 or not all([isinstance(x, int) and (x > 0) for x in var]):
            raise ValueError(f"{name} is expected to be non-empty and contain positive integers, but got {var}")
    elif isinstance(var, int):
        if var <= 0:
            raise ValueError(f"{name} is expected to be a positive integer, but got {var}")
    else:
        raise ValueError(f"Unsupported argument type. Expected int or Iterable[int], but got {type(var)}")


def compare_dicts_recursively(d1: Dict, d2: Dict) -> bool:  # type: ignore
    """
    The function compares dictionaries and prints the exact information where they differ. By using the built-in
    dictionary comparison one can get just a plain 'True' or 'False' as result of the comparison, without any hints
    on where the dictionaries differ.
    """
    assert set(d1.keys()) == set(
        d2.keys()
    ), f"The dictionaries keys are different.\nDict_1 keys: {set(d1.keys())}\nDict_2 keys: {set(d2.keys())}"
    for k, v in d1.items():
        if isinstance(v, dict):
            assert compare_dicts_recursively(
                v, d2[k]
            ), f"The dictionaries differs at key {k}.\nDict_1 value: {v}\nDict_2 value: {d2[k]}"
        else:
            assert d2[k] == v, f"Key name: {k}\nDict_1 value: {v}\nDict_2 value: {d2[k]}"
    return True


def find_first_occurrences(x: List[Hashable]) -> List[int]:
    """
    Args:
        x: Sequence of something hashable

    Returns:
        Indices of elements which are first occurrences

    """
    seen = set()
    first_ids = []
    for i, el in enumerate(x):
        if el not in seen:
            first_ids.append(i)
        seen.add(el)
    return first_ids


__all__ = [
    "find_value_ids",
    "set_global_seed",
    "one_hot",
    "flatten_dict",
    "load_dotenv",
    "dictconfig_to_dict",
    "smart_sample",
    "clip_max",
    "check_if_nonempty_positive_integers",
    "compare_dicts_recursively",
    "find_first_occurrences",
]
