from typing import Any, Dict

import torch
from pytorch_lightning.plugins import DDPPlugin

from oml.const import TCfg
from oml.utils.misc import dictconfig_to_dict


def parse_runtime_params_from_config(cfg: TCfg) -> Dict[str, Any]:
    """
    Function parses cfg and based on it prepares DDP parameters for PytorchLightning Trainer module.
    There are two parameters 'accelerator' and 'devices' you can configure. If one of the parameters or both of them
    are not specified, the best option based on your hardware will be automatically prepared for you.
    Posible options for 'accelerator' are 'cpu' and 'gpu'.
    You can select specific gpus using list 'devices=[0, 2]' or specify the number of gpus by `devices=3`.
    Experiment might be launched in DDP with 'cpu' accelerator. In this case, 'devices' (integer value or len of list)
    interpreted as number of processes.

    TODO: now DDP works in wrong way. We temporarily force to specify 'devices=1'
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

    # TODO: we will remove it after merging PR with proper DDP
    if not (strategy is None):
        raise RuntimeError("Now DDP works in wrong way. Please, specifiy explicitly 'devices=1' in your config")

    return {
        "devices": devices,
        "strategy": strategy,
        "accelerator": accelerator,
        "gpus": None,
        "replace_sampler_ddp": False,
    }
