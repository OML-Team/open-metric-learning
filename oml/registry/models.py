from typing import Any, Dict

from oml.interfaces.models import IExtractor
from oml.models.resnet import ResnetExtractor, ResNetWithLinearExtractor
from oml.models.vit.clip import ViTCLIPExtractor
from oml.models.vit.vit import ViTExtractor, ViTWithLinearExtractor
from oml.utils.misc import TCfg, dictconfig_to_dict

MODELS_REGISTRY = {
    "resnet": ResnetExtractor,
    "resnet_with_projection": ResNetWithLinearExtractor,
    "vit": ViTExtractor,
    "vit_with_projection": ViTWithLinearExtractor,
    "vit_clip": ViTCLIPExtractor,
}


def get_extractor(model_name: str, **kwargs: Dict[str, Any]) -> IExtractor:
    extractor = MODELS_REGISTRY[model_name](**kwargs)  # type: ignore
    return extractor


def get_extractor_by_cfg(cfg: TCfg) -> IExtractor:
    cfg = dictconfig_to_dict(cfg)
    extractor = get_extractor(model_name=cfg["name"], **cfg["args"])
    return extractor


__all__ = ["MODELS_REGISTRY", "get_extractor", "get_extractor_by_cfg"]
