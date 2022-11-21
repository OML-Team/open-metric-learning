# type: ignore
from typing import Any, Dict

import pytest
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Optimizer

from oml.const import CONFIGS_PATH, TCfg
from oml.registry.losses import LOSSES_REGISTRY, get_criterion
from oml.registry.miners import MINERS_REGISTRY, get_miner
from oml.registry.models import MODELS_REGISTRY, get_extractor, raise_if_needed
from oml.registry.optimizers import (
    OPTIMIZERS_REGISTRY,
    get_optimizer,
    get_optimizer_by_cfg,
)
from oml.registry.samplers import SAMPLERS_REGISTRY, get_sampler
from oml.registry.schedulers import SCHEDULERS_REGISTRY, get_scheduler
from oml.registry.transforms import TRANSFORMS_REGISTRY, get_transforms
from oml.utils.misc import dictconfig_to_dict


def get_sampler_kwargs_runtime() -> Any:
    return {"label2category": {0: 0, 1: 0, 2: 1, 3: 1}, "labels": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]}


def get_params() -> Any:
    return list(nn.Linear(3, 6).parameters())


def get_opt() -> Optimizer:
    return get_optimizer_by_cfg({"name": "sgd", "args": {"lr": 0.001, "params": get_params()}})


@pytest.mark.parametrize(
    "folder_name,registry,factory_fun,runtime_args",
    [
        ("model", MODELS_REGISTRY, get_extractor, None),
        ("criterion", LOSSES_REGISTRY, get_criterion, None),
        ("miner", MINERS_REGISTRY, get_miner, None),
        ("optimizer", OPTIMIZERS_REGISTRY, get_optimizer, {"params": get_params()}),
        ("sampler", SAMPLERS_REGISTRY, get_sampler, get_sampler_kwargs_runtime()),
        ("scheduler", SCHEDULERS_REGISTRY, get_scheduler, {"optimizer": get_opt()}),
        ("transforms", TRANSFORMS_REGISTRY, get_transforms, None),
    ],
)
def test_registry(folder_name, registry, factory_fun, runtime_args) -> None:
    for obj_name in registry.keys():
        cfg = dictconfig_to_dict(OmegaConf.load(CONFIGS_PATH / folder_name / f"{obj_name}.yaml"))
        args = cfg["args"]

        # this case is special since only 2 schedulers have "lr_lambda" param which is not in defaults
        if (folder_name == "scheduler") and (obj_name == "lambda" or obj_name == "multiplicative"):
            args["lr_lambda"] = lambda epoch: 0.9

        if runtime_args is not None:
            args = dict(**args, **runtime_args)

        factory_fun(cfg["name"], **args)

    assert True


@pytest.mark.parametrize(
    "extractor_cfg,kwargs,raises",
    [
        ({"weights": None}, {"weights": "some_weights"}, False),
        ({"weights": None}, {"weights": None}, False),
        ({"weights": "some_weights"}, {"weights": None}, False),
        ({"weights": "some_weights"}, {"weights": "some_other_weights"}, True),
    ],
)
def test_model_raises(extractor_cfg: TCfg, kwargs: Dict[str, Any], raises: bool, model_name: str = "default") -> None:
    if raises:
        with pytest.raises(ValueError):
            raise_if_needed(extractor_cfg=extractor_cfg, kwargs=kwargs, model_name=model_name)
    else:
        raise_if_needed(extractor_cfg=extractor_cfg, kwargs=kwargs, model_name=model_name)
