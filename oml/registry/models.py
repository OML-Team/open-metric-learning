from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from oml.interfaces.models import IExtractor, IHead
from oml.models.heads import ArcFaceHead, SimpleLinearHead
from oml.models.resnet import ResnetExtractor
from oml.models.vit.vit import ViTExtractor
from oml.utils.misc import TCfg, dictconfig_to_dict

MODELS_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs" / "model"

MODELS_REGISTRY = {
    "resnet": ResnetExtractor,
    "vit": ViTExtractor,
}

HEADS_REGISTRY = {
    "simple_linear": SimpleLinearHead,
    "arcface": ArcFaceHead,
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


def get_extractor_by_cfg_file(cfg_path: Path) -> IExtractor:
    return get_extractor_by_cfg(OmegaConf.load(cfg_path))


def get_pretrained_resnet() -> ResnetExtractor:
    # ResNet model was trained on ImageNet vis MoCo Framework
    return get_extractor_by_cfg_file(MODELS_CONFIGS_PATH / "resnet.yaml")


def get_pretrained_vit() -> ViTExtractor:
    # ViT model was trained on ImageNet vis DiNo Framework
    return get_extractor_by_cfg_file(MODELS_CONFIGS_PATH / "vit.yaml")
