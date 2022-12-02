from typing import Any, Dict

from oml.interfaces.samplers import IBatchSampler
from oml.samplers.balance import BalanceSampler
from oml.samplers.category_balance import CategoryBalanceSampler
from oml.samplers.distinct_category_balance import DistinctCategoryBalanceSampler
from oml.utils.misc import TCfg, dictconfig_to_dict, remove_unused_kargs

SAMPLERS_CATEGORIES_BASED = {
    "category_balance": CategoryBalanceSampler,
    "distinct_category_balance": DistinctCategoryBalanceSampler,
}

SAMPLERS_REGISTRY = {
    **SAMPLERS_CATEGORIES_BASED,  # type: ignore
    "balance": BalanceSampler,
}


def get_sampler(name: str, **kwargs: Dict[str, Any]) -> IBatchSampler:
    constructor = SAMPLERS_REGISTRY[name]
    kwargs = remove_unused_kargs(kwargs, constructor)
    return constructor(**kwargs)  # type: ignore


def get_sampler_by_cfg(cfg: TCfg, **kwargs_runtime: Dict[str, Any]) -> IBatchSampler:
    cfg = dictconfig_to_dict(cfg)
    cfg["args"].update(kwargs_runtime)
    return get_sampler(name=cfg["name"], **cfg["args"])


__all__ = ["SAMPLERS_REGISTRY", "SAMPLERS_CATEGORIES_BASED", "get_sampler", "get_sampler_by_cfg"]
