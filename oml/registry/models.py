from typing import Any, Dict
from warnings import warn

from oml.interfaces.models import IExtractor
from oml.models.projection import ExtractorWithMLP
from oml.models.resnet import ResnetExtractor
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor
from oml.utils.misc import TCfg, dictconfig_to_dict

MODELS_REGISTRY = {
    "resnet": ResnetExtractor,
    "vit": ViTExtractor,
    "vit_clip": ViTCLIPExtractor,
    "extractor_with_mlp": ExtractorWithMLP,
}


def get_extractor(model_name: str, **kwargs: Dict[str, Any]) -> IExtractor:
    if "extractor" in kwargs:
        inside_extractor = get_extractor_by_cfg(kwargs.pop("extractor"))
        extractor = MODELS_REGISTRY[model_name](extractor=inside_extractor, **kwargs)  # type: ignore
        if inside_extractor.get("weights", None) and kwargs.get("weights", None):
            raise ValueError("You should only provide one weight for extractor_with_mlp.")
        elif inside_extractor.get("weights", 0) is None and kwargs.get("weights", None):
            warn(f"There are weights provided for {model_name}. They can overwrite internal extractor's weights.")
    else:
        extractor = MODELS_REGISTRY[model_name](**kwargs)  # type: ignore
    return extractor


def get_extractor_by_cfg(cfg: TCfg) -> IExtractor:
    cfg = dictconfig_to_dict(cfg)
    extractor = get_extractor(model_name=cfg["name"], **cfg["args"])
    return extractor


__all__ = ["MODELS_REGISTRY", "get_extractor", "get_extractor_by_cfg"]
