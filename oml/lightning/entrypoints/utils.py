from typing import Any, Dict

import torch
from pytorch_lightning.plugins import DDPPlugin

from oml.const import TCfg
from oml.utils.misc import dictconfig_to_dict


def parse_runtime_params_from_config(cfg: TCfg) -> Dict[str, Any]:
    cfg = dictconfig_to_dict(cfg.copy())

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
