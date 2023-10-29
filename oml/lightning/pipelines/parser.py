import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.strategies import DDPStrategy

from oml.const import OML_PATH, TCfg
from oml.datasets.base import DatasetWithLabels
from oml.interfaces.samplers import IBatchSampler
from oml.registry.loggers import get_logger_by_cfg
from oml.registry.samplers import SAMPLERS_CATEGORIES_BASED, get_sampler_by_cfg
from oml.registry.schedulers import get_scheduler_by_cfg
from oml.registry.transforms import save_transforms_as_files
from oml.utils.misc import dictconfig_to_dict, flatten_dict


def parse_engine_params_from_config(cfg: TCfg) -> Dict[str, Any]:
    """
    The function parses config and based on it prepares DDP parameters for PytorchLightning Trainer module.
    There are two parameters 'accelerator' and 'devices' you can configure. If one of the parameters or both of them
    are not specified, the best option based on your hardware will be automatically prepared for you.
    Possible options for 'accelerator' are 'cpu' and 'gpu'.
    You can select specific GPUs using the list 'devices=[0, 2]' or specify the number of GPUs by `devices=3`.
    An experiment might be launched in DDP with the 'cpu' accelerator. In this case, 'devices' (integer value or
    length of list) interpreted as a number of processes.
    """
    cfg = dictconfig_to_dict(cfg)

    # we want to replace possible null or no values in config for "accelerator" and "devices"
    accelerator = cfg.get("accelerator")
    if accelerator is None:
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    devices = cfg.get("devices")
    if devices is None:
        devices = torch.cuda.device_count() if (torch.cuda.is_available() and accelerator == "gpu") else 1

    if isinstance(devices, (list, tuple)) and accelerator == "cpu":
        devices = len(devices)

    if (isinstance(devices, int) and devices > 1) or (isinstance(devices, (list, tuple)) and len(devices) > 1):
        strategy = DDPStrategy(find_unused_parameters=cfg.get("find_unused_parameters", False))
    else:
        strategy = "auto"

    return {
        "devices": devices,
        "strategy": strategy,
        "accelerator": accelerator,
        "use_distributed_sampler": False,
    }


def check_is_config_for_ddp(cfg: TCfg) -> bool:
    return bool(cfg["strategy"])


def parse_logger_from_config(cfg: TCfg) -> Logger:
    logger = TensorBoardLogger(".") if cfg.get("logger", None) is None else get_logger_by_cfg(cfg["logger"])
    return logger


def initialize_logging(cfg: TCfg) -> Logger:
    logger = parse_logger_from_config(cfg)
    cwd = Path.cwd().name

    dict_to_log = flatten_dict({**dictconfig_to_dict(cfg), **{"dir": cwd}}, sep="|")

    tags = list(cfg.get("tags", [])) + [cfg.get("postfix", "")] + [cwd]

    if isinstance(logger, NeptuneLogger):
        warnings.warn(
            "Unfortunately, in the case of using Neptune, you may experience that long experiments are"
            "stacked and not responding. It's not an issue on OML's side, so, we cannot fix it."
        )
        logger.log_hyperparams(dict_to_log)
        upload_files_to_neptune_cloud(logger, cfg)
        logger.run["sys/tags"].add(tags)

    elif isinstance(logger, WandbLogger):
        logger.log_hyperparams(dict_to_log)
        upload_files_to_wandb_cloud(logger, cfg)
        logger.experiment.tags = list(filter(lambda x: len(x) > 0, tags))  # it fails in the case of empty tag

    elif isinstance(logger, TensorBoardLogger):
        pass

    else:
        raise ValueError(f"Unexpected logger {type(logger)}")

    return logger


def upload_files_to_neptune_cloud(logger: NeptuneLogger, cfg: TCfg) -> None:
    assert isinstance(logger, NeptuneLogger)

    # log transforms as files
    for key, transforms_file in save_transforms_as_files(cfg):
        logger.run[key].upload(transforms_file)

    # log source code
    source_files = list(map(lambda x: str(x), OML_PATH.glob("**/*.py"))) + list(
        map(lambda x: str(x), OML_PATH.glob("**/*.yaml"))
    )
    logger.run["code"].upload_files(source_files)

    # log dataset
    logger.run["dataset"].upload(str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]))


def upload_files_to_wandb_cloud(logger: WandbLogger, cfg: TCfg) -> None:
    assert isinstance(logger, WandbLogger)

    # log transforms as files
    keys_files = save_transforms_as_files(cfg)
    if keys_files:
        transforms = wandb.Artifact("transforms", type="transforms")
        for _, transforms_file in keys_files:
            transforms.add_file(transforms_file)
        logger.experiment.log_artifact(transforms)

    # log source code
    code = wandb.Artifact("source_code", type="code")
    code.add_dir(OML_PATH, name="oml")
    logger.experiment.log_artifact(code)

    # log dataset
    dataset = wandb.Artifact("dataset", type="dataset")
    dataset.add_file(str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]))
    logger.experiment.log_artifact(dataset)


def parse_scheduler_from_config(cfg: TCfg, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    if cfg.get("scheduling"):
        scheduler_kwargs = {
            "scheduler": get_scheduler_by_cfg(cfg["scheduling"]["scheduler"], **{"optimizer": optimizer}),
            "scheduler_interval": cfg["scheduling"]["scheduler_interval"],
            "scheduler_frequency": cfg["scheduling"]["scheduler_frequency"],
            "scheduler_monitor_metric": cfg["scheduling"].get("monitor_metric", None),
        }
    else:
        scheduler_kwargs = {"scheduler": None}

    return scheduler_kwargs


def parse_sampler_from_config(cfg: TCfg, dataset: DatasetWithLabels) -> Optional[IBatchSampler]:
    if (not dataset.categories_key) and cfg["sampler"]["name"] in SAMPLERS_CATEGORIES_BASED.keys():
        raise ValueError(
            "NOTE! You are trying to use Sampler which works with the information related"
            "to categories, but there is no <categories_key> in your Dataset."
        )

    sampler_runtime_args = {"labels": dataset.get_labels(), "label2category": dataset.get_label2category()}
    sampler = get_sampler_by_cfg(cfg["sampler"], **sampler_runtime_args) if cfg["sampler"] is not None else None

    return sampler


def parse_ckpt_callback_from_config(cfg: TCfg) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=Path.cwd() / "checkpoints",
        monitor=cfg["metric_for_checkpointing"],
        mode="max",
        save_top_k=1,
        verbose=True,
        filename="best",
    )


__all__ = [
    "parse_engine_params_from_config",
    "check_is_config_for_ddp",
    "initialize_logging",
    "upload_files_to_neptune_cloud",
    "upload_files_to_wandb_cloud",
    "parse_scheduler_from_config",
    "parse_sampler_from_config",
    "parse_ckpt_callback_from_config",
]
