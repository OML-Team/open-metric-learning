from typing import Any, Dict, Union

from pytorch_lightning.loggers import LightningLoggerBase, NeptuneLogger, WandbLogger, TensorBoardLogger

from oml.const import TCfg
from oml.utils.misc import dictconfig_to_dict

LOGGERS_REGISTRY = {
    "wandb_logger": WandbLogger,
    "neptune_logger": NeptuneLogger,
    "tensorboard_logger": TensorBoardLogger
}


def get_logger(name: str, **kwargs: Dict[str, Any]) -> LightningLoggerBase:
    return LOGGERS_REGISTRY[name](**kwargs)


def get_logger_by_cfg(cfg: TCfg) -> Union[bool, LightningLoggerBase]:
    cfg = dictconfig_to_dict(cfg)
    logger = get_logger(cfg["name"], **cfg["args"])
    return logger


__all__ = ["LOGGERS_REGISTRY", "get_logger", "get_logger_by_cfg"]
