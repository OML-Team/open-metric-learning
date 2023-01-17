from typing import Any, Dict

from oml.const import TCfg
from oml.interfaces.models import IPairwiseModel
from oml.models.siamese import LinearSiamese, ResNetSiamese
from oml.utils.misc import dictconfig_to_dict

PAIRWISE_MODELS_REGISTRY = {"linear_siamese": LinearSiamese, "resnet_siamese": ResNetSiamese}


def get_pairwise_model(model_name: str, **kwargs: Dict[str, Any]) -> IPairwiseModel:
    pairwise_model = PAIRWISE_MODELS_REGISTRY[model_name](**kwargs)
    return pairwise_model


def get_pairwise_model_by_cfg(cfg: TCfg) -> IPairwiseModel:
    cfg = dictconfig_to_dict(cfg)
    pairwise_model = get_pairwise_model(model_name=cfg["name"], **cfg["args"])
    return pairwise_model


__all__ = ["get_pairwise_model", "get_pairwise_model_by_cfg"]
