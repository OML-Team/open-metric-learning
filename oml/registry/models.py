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


def raise_if_needed(extractor_cfg: TCfg, kwargs: Dict[str, Any], model_name: str) -> None:
    if extractor_cfg.get("weights", False) and kwargs.get("weights", False):
        raise ValueError("You should only provide one weight for extractor_with_mlp.")
    elif extractor_cfg.get("weights", "") is None and kwargs.get("weights", False):
        warn(f"There are weights provided for {model_name}. They can overwrite internal extractor's weights.")


def get_extractor(model_name: str, **kwargs: Dict[str, Any]) -> IExtractor:
    if "extractor" in kwargs:
        extractor_cfg = kwargs.pop("extractor")
        inside_extractor = get_extractor_by_cfg(extractor_cfg)
        extractor = MODELS_REGISTRY[model_name](extractor=inside_extractor, **kwargs)  # type: ignore
        raise_if_needed(extractor_cfg, kwargs, model_name)
    else:
        extractor = MODELS_REGISTRY[model_name](**kwargs)  # type: ignore
    return extractor


def get_extractor_by_cfg(cfg: TCfg) -> IExtractor:
    cfg = dictconfig_to_dict(cfg)
    extractor = get_extractor(model_name=cfg["name"], **cfg["args"])
    return extractor


__all__ = ["MODELS_REGISTRY", "get_extractor", "get_extractor_by_cfg"]
