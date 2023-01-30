from typing import Any, Dict
from warnings import warn

from oml.interfaces.models import IExtractor, IPairwiseModel
from oml.models.projection import ExtractorWithMLP
from oml.models.resnet import ResnetExtractor
from oml.models.siamese import (
    ConcatSiamese,
    LinearTrivialDistanceSiamese,
    TrivialDistanceSiamese,
)
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor
from oml.utils.misc import TCfg, dictconfig_to_dict

EXTRACTORS_REGISTRY = {
    "resnet": ResnetExtractor,
    "vit": ViTExtractor,
    "vit_clip": ViTCLIPExtractor,
    "extractor_with_mlp": ExtractorWithMLP,
}

PAIRWISE_MODELS_REGISTRY = {
    "linear_siamese": LinearTrivialDistanceSiamese,
    "concat_siamese": ConcatSiamese,
    "trivial_distance_siamese": TrivialDistanceSiamese,
}


def raise_if_needed(extractor_cfg: TCfg, kwargs: Dict[str, Any], model_name: str) -> None:
    if extractor_cfg.get("weights", False) and kwargs.get("weights", False):
        raise ValueError("You should only provide one weight for extractor_with_mlp.")
    elif extractor_cfg.get("weights", "") is None and kwargs.get("weights", False):
        warn(f"There are weights provided for {model_name}. They can overwrite internal extractor's weights.")


def _get_model(model_name: str, registry: Dict[str, Any], **kwargs: Dict[str, Any]) -> IExtractor:
    if "extractor" in kwargs:  # for nested models
        extractor_cfg = kwargs.pop("extractor")
        inside_extractor = get_extractor_by_cfg(extractor_cfg)
        model = registry[model_name](extractor=inside_extractor, **kwargs)  # type: ignore
        raise_if_needed(extractor_cfg, kwargs, model_name)
    else:
        model = registry[model_name](**kwargs)  # type: ignore
    return model


def get_extractor(model_name: str, **kwargs: Dict[str, Any]) -> IExtractor:
    return _get_model(model_name=model_name, registry=EXTRACTORS_REGISTRY, **kwargs)


def get_extractor_by_cfg(cfg: TCfg) -> IExtractor:
    cfg = dictconfig_to_dict(cfg)
    extractor = get_extractor(model_name=cfg["name"], **cfg["args"])
    return extractor


def get_pairwise_model(model_name: str, **kwargs: Dict[str, Any]) -> IPairwiseModel:
    return _get_model(model_name=model_name, registry=PAIRWISE_MODELS_REGISTRY, **kwargs)


def get_pairwise_model_by_cfg(cfg: TCfg) -> IPairwiseModel:
    cfg = dictconfig_to_dict(cfg)
    pairwise_model = get_pairwise_model(model_name=cfg["name"], **cfg["args"])
    return pairwise_model


__all__ = [
    "EXTRACTORS_REGISTRY",
    "PAIRWISE_MODELS_REGISTRY",
    "get_extractor",
    "get_extractor_by_cfg",
    "get_pairwise_model",
    "get_pairwise_model_by_cfg",
]
