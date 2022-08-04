from typing import Any, Dict

from torch.utils.data.sampler import Sampler

from oml.samplers.balance import SequentialBalanceSampler
from oml.samplers.category_balance import SequentialCategoryBalanceSampler
from oml.samplers.distinct_category_balance import (
    SequentialDistinctCategoryBalanceSampler,
)
from oml.utils.misc import TCfg, dictconfig_to_dict

SAMPLERS_CATEGORIES_BASED = {
    "SequentialCategoryBalanceSampler": SequentialCategoryBalanceSampler,
    "SequentialDistinctCategoryBalanceSampler": SequentialDistinctCategoryBalanceSampler,
}

SAMPLERS_REGISTRY = {
    **SAMPLERS_CATEGORIES_BASED,  # type: ignore
    "SequentialBalanceSampler": SequentialBalanceSampler,
}


def get_sampler(name: str, **kwargs: Dict[str, Any]) -> Sampler:
    return SAMPLERS_REGISTRY[name](**kwargs)  # type: ignore


def get_sampler_by_cfg(cfg: TCfg, **kwargs_runtime: Dict[str, Any]) -> Sampler:
    cfg = dictconfig_to_dict(cfg)
    cfg["args"].update(kwargs_runtime)
    return get_sampler(name=cfg["name"], **cfg["args"])


__all__ = ["SAMPLERS_REGISTRY", "get_sampler", "get_sampler_by_cfg"]
