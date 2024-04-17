import os
from typing import Any, Dict

from oml.const import TCfg
from oml.interfaces.loggers import IPipelineLogger
from oml.lightning.pipelines.logging import (
    ClearMLPipelineLogger,
    MLFlowPipelineLogger,
    NeptunePipelineLogger,
    TensorBoardPipelineLogger,
    WandBPipelineLogger,
)
from oml.utils.misc import dictconfig_to_dict

LOGGERS_REGISTRY = {
    "wandb": WandBPipelineLogger,
    "neptune": NeptunePipelineLogger,
    "tensorboard": TensorBoardPipelineLogger,
    "mlflow": MLFlowPipelineLogger,
    "clearml": ClearMLPipelineLogger,
}

CLOUD_TOKEN_NAMES = {"wandb": "WANDB_API_KEY", "neptune": "NEPTUNE_API_TOKEN"}
TOKEN_ERROR_MESSAGE = (
    "{} logger is specified in your config file, "
    "but <{}> is not provided as a global environment variable."
    "Please, execute `export {}=your_token` before running the script."
)


def get_logger(name: str, **kwargs: Dict[str, Any]) -> IPipelineLogger:
    if (name in CLOUD_TOKEN_NAMES) and (CLOUD_TOKEN_NAMES[name] not in os.environ):
        token_name = CLOUD_TOKEN_NAMES[name]
        raise ValueError(TOKEN_ERROR_MESSAGE.format(name.upper(), token_name, token_name))

    return LOGGERS_REGISTRY[name](**kwargs)  # type: ignore


def get_logger_by_cfg(cfg: TCfg) -> IPipelineLogger:
    cfg = dictconfig_to_dict(cfg)
    logger = get_logger(cfg["name"], **cfg["args"])
    return logger


__all__ = ["LOGGERS_REGISTRY", "get_logger", "get_logger_by_cfg"]
