from typing import Any, Dict

from omegaconf import OmegaConf

from oml.interfaces.models import IExtractor, IHead
from oml.models.heads import ArcFaceFancy, ArcFaceHead, SimpleLinearHead
from oml.models.resnet import ResnetExtractor
from oml.models.vit.vit import ViTExtractor
from oml.utils.misc import TCfg, dictconfig_to_dict

MODELS_REGISTRY = {
    "resnet": ResnetExtractor,
    "vit": ViTExtractor,
}

HEADS_REGISTRY = {
    "simple_linear": SimpleLinearHead,
    "arcface": ArcFaceHead,
    "fancy_arcface": ArcFaceFancy,
}


def get_head_by_cfg(cfg: TCfg, **kwargs: Dict[str, Any]) -> IHead:
    head_name = cfg["name"]
    cfg.setdefault("args", {})
    cfg["args"].update(kwargs)
    return HEADS_REGISTRY[head_name](**cfg["args"])  # type: ignore


def get_extractor(model_name: str, **kwargs: Dict[str, Any]) -> IExtractor:
    extractor = MODELS_REGISTRY[model_name](**kwargs)  # type: ignore
    return extractor


def get_extractor_by_cfg(cfg: TCfg) -> IExtractor:
    cfg = dictconfig_to_dict(cfg)
    extractor = get_extractor(model_name=cfg["name"], **cfg["args"])
    return extractor


__all__ = ["MODELS_REGISTRY", "get_extractor", "get_extractor_by_cfg"]
