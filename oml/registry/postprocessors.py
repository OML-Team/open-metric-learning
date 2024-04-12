from typing import Any, Dict

from oml.const import TCfg
from oml.interfaces.retrieval import IRetrievalPostprocessor
from oml.registry.models import get_pairwise_model_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import (
    PairwiseEmbeddingsPostprocessor,
    PairwiseImagesPostprocessor,
    PairwiseReranker,
)
from oml.utils.misc import dictconfig_to_dict

POSTPROCESSORS_REGISTRY = {
    "pairwise_reranker": PairwiseReranker,
    "pairwise_images": PairwiseReranker,
    "pairwise_embeddings": PairwiseReranker,
}


def get_postprocessor(name: str, **kwargs: Dict[str, Any]) -> IRetrievalPostprocessor:
    constructor = POSTPROCESSORS_REGISTRY[name]
    if "transforms" in kwargs:
        # todo 522: warning - we dont expect transforms anymore
        kwargs["transforms"] = get_transforms_by_cfg(kwargs["transforms"])

    if "pairwise_model" in kwargs:
        kwargs["pairwise_model"] = get_pairwise_model_by_cfg(kwargs["pairwise_model"])

    return constructor(**kwargs)


def get_postprocessor_by_cfg(cfg: TCfg) -> IRetrievalPostprocessor:
    cfg = dictconfig_to_dict(cfg)
    postprocessor = get_postprocessor(cfg["name"], **cfg["args"])
    return postprocessor


__all__ = ["get_postprocessor", "get_postprocessor_by_cfg"]
