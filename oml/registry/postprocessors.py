from typing import Any, Dict

from oml.const import TCfg
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.registry.models import get_pairwise_model_by_cfg
from oml.retrieval.postprocessors.algo import AdaptiveThresholding, ConstantThresholding
from oml.retrieval.postprocessors.pairwise import PairwiseReranker
from oml.utils.misc import dictconfig_to_dict

POSTPROCESSORS_REGISTRY = {
    "pairwise_reranker": PairwiseReranker,
    "constant_thresholding": ConstantThresholding,
    "adaptive_thresholding": AdaptiveThresholding,
}


def get_postprocessor(name: str, **kwargs: Dict[str, Any]) -> IRetrievalPostprocessor:
    constructor = POSTPROCESSORS_REGISTRY[name]

    if "pairwise_model" in kwargs:
        kwargs["pairwise_model"] = get_pairwise_model_by_cfg(kwargs["pairwise_model"])  # type: ignore

    return constructor(**kwargs)  # type: ignore


def get_postprocessor_by_cfg(cfg: TCfg) -> IRetrievalPostprocessor:
    cfg = dictconfig_to_dict(cfg)
    postprocessor = get_postprocessor(cfg["name"], **cfg["args"])
    return postprocessor


__all__ = ["get_postprocessor", "get_postprocessor_by_cfg"]
