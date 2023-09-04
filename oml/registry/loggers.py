import os
from typing import Any, Dict, Union

from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.logger import Logger

from oml.const import TCfg
from oml.utils.misc import dictconfig_to_dict

LOGGERS_REGISTRY = {"wandb": WandbLogger, "neptune": NeptuneLogger, "tensorboard": TensorBoardLogger}

CLOUD_TOKEN_NAMES = {"wandb": "WANDB_API_KEY", "neptune": "NEPTUNE_API_TOKEN"}
TOKEN_ERROR_MESSAGE = (
    "{} logger is specified in your config file, "
    "but <{}> is not provided as a global environment variable."
    "Please, execute `export {}=your_token` before running the script."
)


def get_logger(name: str, **kwargs: Dict[str, Any]) -> Logger:
    if (name in CLOUD_TOKEN_NAMES) and (CLOUD_TOKEN_NAMES[name] not in os.environ):
        token_name = CLOUD_TOKEN_NAMES[name]
        raise ValueError(TOKEN_ERROR_MESSAGE.format(name.upper(), token_name, token_name))

    return LOGGERS_REGISTRY[name](**kwargs)


def get_logger_by_cfg(cfg: TCfg) -> Union[bool, Logger]:
    cfg = dictconfig_to_dict(cfg)
    logger = get_logger(cfg["name"], **cfg["args"])
    return logger


__all__ = ["LOGGERS_REGISTRY", "get_logger", "get_logger_by_cfg"]
