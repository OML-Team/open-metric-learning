from typing import Any, Dict

from oml.const import TCfg
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.registry.models import get_pairwise_model_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.retrieval.postprocessors.pairwise import (
    PairwiseEmbeddingsPostprocessor,
    PairwiseImagesPostprocessor,
)
from oml.utils.misc import dictconfig_to_dict

POSTPROCESSORS_REGISTRY = {
    "pairwise_images": PairwiseImagesPostprocessor,
    "pairwise_embeddings": PairwiseEmbeddingsPostprocessor,
}


def get_postprocessor(name: str, **kwargs: Dict[str, Any]) -> IDistancesPostprocessor:
    constructor = POSTPROCESSORS_REGISTRY[name]
    if "transforms" in kwargs:
        kwargs["transforms"] = get_transforms_by_cfg(kwargs["transforms"])

    if "pairwise_model" in kwargs:
        kwargs["pairwise_model"] = get_pairwise_model_by_cfg(kwargs["pairwise_model"])

    return constructor(**kwargs)


def get_postprocessor_by_cfg(cfg: TCfg) -> IDistancesPostprocessor:
    cfg = dictconfig_to_dict(cfg)
    postprocessor = get_postprocessor(cfg["name"], **cfg["args"])
    return postprocessor


__all__ = ["get_postprocessor", "get_postprocessor_by_cfg"]
