from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from oml.interfaces.models import IExtractor
from oml.models.resnet import ResnetExtractor
from oml.models.vit.vit import ViTExtractor
from oml.utils.misc import TCfg, dictconfig_to_dict

MODELS_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs" / "model"

MODELS_REGISTRY = {
    "resnet": ResnetExtractor,
    "vit": ViTExtractor,
}


def get_extractor(model_name: str, **kwargs: Dict[str, Any]) -> IExtractor:
    extractor = MODELS_REGISTRY[model_name](**kwargs)  # type: ignore
    return extractor


def get_extractor_by_cfg(cfg: TCfg) -> IExtractor:
    cfg = dictconfig_to_dict(cfg)
    extractor = get_extractor(model_name=cfg["name"], **cfg["args"])
    return extractor


def get_extractor_by_cfg_file(cfg_path: Path) -> IExtractor:
    return get_extractor_by_cfg(OmegaConf.load(cfg_path))
