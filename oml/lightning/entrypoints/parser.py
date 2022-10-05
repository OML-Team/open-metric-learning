from typing import Any, Dict

import torch
from pytorch_lightning.plugins import DDPPlugin

from oml.const import TCfg
from oml.utils.misc import dictconfig_to_dict


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
        strategy = DDPPlugin()
    else:
        strategy = None

    return {
        "devices": devices,
        "strategy": strategy,
        "accelerator": accelerator,
        "gpus": None,
        "replace_sampler_ddp": False,
    }


def check_is_config_for_ddp(cfg: TCfg) -> bool:
    return bool(cfg["strategy"])


__all__ = ["parse_engine_params_from_config", "check_is_config_for_ddp"]
