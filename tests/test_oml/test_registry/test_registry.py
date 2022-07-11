from pathlib import Path
from typing import Any, Callable, Dict

import pytest
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Optimizer

from oml.const import CONFIGS_PATH
from oml.registry.losses import get_criterion_by_cfg
from oml.registry.miners import get_miner_by_cfg
from oml.registry.models import get_extractor_by_cfg
from oml.registry.optimizers import get_optimizer_by_cfg
from oml.registry.samplers import get_sampler_by_cfg
from oml.registry.schedulers import get_scheduler_by_cfg
from oml.utils.misc import TCfg


@pytest.mark.parametrize(
    "folder_name,factory_fun",
    [
        ("model", get_extractor_by_cfg),
        ("criterion", get_criterion_by_cfg),
        ("miner", get_miner_by_cfg),
    ],
)
def test_registry(folder_name: str, factory_fun: Callable[[TCfg], Any]) -> None:
    for cfg_path in (CONFIGS_PATH / folder_name).glob("**/*.yaml"):
        with open(cfg_path, "r") as f:
            factory_fun(OmegaConf.load(f))
    assert True


def get_sampler_kwargs_runtime() -> Any:
    return {"label2category": {0: 0, 1: 0, 2: 1, 3: 1}, "labels": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]}


def get_params() -> Any:
    return list(nn.Linear(3, 6).parameters())


def get_opt() -> Optimizer:
    return get_optimizer_by_cfg({"name": "sgd", "args": {"lr": 0.001, "params": get_params()}})


@pytest.mark.parametrize(
    "cfg_path,factory_fun,runtime_params",
    [
        ("sampler/sequential_balance.yaml", get_sampler_by_cfg, get_sampler_kwargs_runtime()),
        ("sampler/sequential_category_balance.yaml", get_sampler_by_cfg, get_sampler_kwargs_runtime()),
        ("sampler/sequential_distinct_category_balance.yaml", get_sampler_by_cfg, get_sampler_kwargs_runtime()),
        ("scheduler/lambda.yaml", get_scheduler_by_cfg, {"optimizer": get_opt(), "lr_lambda": lambda e: 0.9}),
        ("scheduler/multiplicative.yaml", get_scheduler_by_cfg, {"optimizer": get_opt(), "lr_lambda": lambda e: 0.9}),
        ("scheduler/step.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("scheduler/multi_step.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("scheduler/exponential.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("scheduler/cosine_annealing.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("scheduler/reduce_on_plateau.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("scheduler/cyclic.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("scheduler/one_cycle.yaml", get_scheduler_by_cfg, {"optimizer": get_opt()}),
        ("optimizer/adam.yaml", get_optimizer_by_cfg, {"params": get_params()}),
        ("optimizer/sgd.yaml", get_optimizer_by_cfg, {"params": get_params()}),
    ],
)
def test_registry_with_runtime_args(
    cfg_path: Path, factory_fun: Callable[[TCfg], Any], runtime_params: Dict[str, Any]
) -> None:
    with open(CONFIGS_PATH / cfg_path, "r") as f:
        factory_fun(OmegaConf.load(f), **runtime_params)  # type: ignore
    assert True
