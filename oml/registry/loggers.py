import os
import warnings

from pathlib import Path
from typing import Any, Dict, Union

import albumentations as albu
from pytorch_lightning.loggers import LightningLoggerBase, NeptuneLogger, WandbLogger
import wandb

from oml.const import PROJECT_ROOT, TCfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import dictconfig_to_dict, flatten_dict

LOGGERS_REGISTRY = {
    "wandb_logger": WandbLogger,
    "neptune_logger": NeptuneLogger
}


def upload_files_to_neptune_cloud(logger: NeptuneLogger, cfg: TCfg) -> None:
    assert isinstance(logger, NeptuneLogger)

    # save transforms as files
    for key, val in cfg.items():
        if "transforms" in key:
            try:
                transforms = get_transforms_by_cfg(cfg[key])
                if isinstance(transforms, albu.Compose):
                    transforms_file = str(Path(".hydra/") / f"{key}.yaml") if Path(".hydra").exists() else f"{key}.yaml"
                    albu.save(filepath=transforms_file, transform=transforms, data_format="yaml")
                    logger.run[key].upload(str(transforms_file))
            except Exception:
                print(f"We are not able to interpret {key} as albumentations transforms and log them as a file.")

    # log source code
    source_files = list(map(lambda x: str(x), PROJECT_ROOT.glob("**/*.py"))) + list(
        map(lambda x: str(x), PROJECT_ROOT.glob("**/*.yaml"))
    )
    logger.run["code"].upload_files(source_files)

    # log dataset
    logger.run["dataset"].upload(str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]))


def get_logger(name: str, **kwargs: Dict[str, Any]) -> LightningLoggerBase:
    if name == "wandb_logger":
        wandb.login(key=os.environ["WANDB_API_TOKEN"])
        logger = WandbLogger(**kwargs)

    elif name == "neptune_logger":
        warnings.warn(
            "Unfortunately, in the case of using Neptune, you may experience that long experiments are"
            "stacked and not responding. It's not an issue on OML's side, so, we cannot fix it. You can use"
            "Tensorboard logger instead, for this simply leave <NEPTUNE_API_TOKEN> unfilled."
        )

        logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            log_model_checkpoints=False,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown logger: logger name can take one of the following values: {LOGGERS_REGISTRY.keys()}")

    return logger


def get_logger_by_cfg(cfg: TCfg) -> Union[bool, LightningLoggerBase]:
    if cfg["logger"] is not None:
        logger = get_logger(cfg["logger"]["name"], **cfg["logger"]["args"])
        dict_to_log = {**dictconfig_to_dict(cfg), **{"dir": Path.cwd()}}
        logger.log_hyperparams(flatten_dict(dict_to_log, sep="|"))
        if isinstance(logger, NeptuneLogger):
            upload_files_to_neptune_cloud(logger, cfg)

    else:
        print(f"Your current logger is not able to log your files, you can use {NeptuneLogger.__name__} for this.")
        logger = True

    return logger


__all__ = ["LOGGERS_REGISTRY", "get_logger", "get_logger_by_cfg"]
