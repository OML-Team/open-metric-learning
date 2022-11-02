from typing import List

import pytest
import torch
from pytorch_lightning.plugins import DDPPlugin

from oml.const import TCfg
from oml.lightning.entrypoints.parser import parse_engine_params_from_config


def compare_params(cfg: TCfg, expected_params: TCfg) -> None:
    parsed_params = parse_engine_params_from_config(cfg)

    assert set(parsed_params.keys()) == set(expected_params.keys())

    for key in expected_params.keys():
        if key == "strategy":
            assert isinstance(expected_params[key], type(parsed_params[key]))
        else:

            assert expected_params[key] == parsed_params[key]


@pytest.mark.parametrize("list_devices", [[1], [1, 2], [0, 3], [0, 1, 2, 5]])
def test_list_devices_cpu_accelerator(list_devices: List[int]) -> None:
    cfg = {"accelerator": "cpu", "devices": list_devices}
    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": "cpu",
        "devices": len(list_devices),
        "strategy": DDPPlugin() if len(list_devices) > 1 else None,
    }
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("int_devices", [1, 2, 4])
def test_int_devices_cpu_accelerator(int_devices: int) -> None:
    cfg = {"accelerator": "cpu", "devices": int_devices}
    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": "cpu",
        "devices": int_devices,
        "strategy": DDPPlugin() if int_devices > 1 else None,
    }
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("list_devices", [[1], [1, 2], [0, 3], [0, 1, 2, 5]])
def test_list_devices_gpu_accelerator(list_devices: List[int]) -> None:
    cfg = {"accelerator": "gpu", "devices": list_devices}
    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": "gpu",
        "devices": list_devices,
        "strategy": DDPPlugin() if len(list_devices) > 1 else None,
    }
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("int_devices", [1, 2, 4])
def test_int_devices_gpu_accelerator(int_devices: int) -> None:
    cfg = {"accelerator": "gpu", "devices": int_devices}
    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": "gpu",
        "devices": int_devices,
        "strategy": DDPPlugin() if int_devices > 1 else None,
    }
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("no_devices", [{}, {"devices": None}])
@pytest.mark.parametrize("no_accelerator", [{}, {"accelerator": None}])
def test_no_devices_no_accelerator(no_devices: TCfg, no_accelerator: TCfg) -> None:
    cfg = {}  # type: ignore
    cfg.update(no_devices)
    cfg.update(no_accelerator)

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        if devices > 1:
            strategy = DDPPlugin()
        else:
            strategy = None
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None

    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
    }

    compare_params(cfg, expected_params)


@pytest.mark.parametrize("no_devices", [{}, {"devices": None}])
def test_no_devices_cpu_accelerator(no_devices: TCfg) -> None:
    cfg = {"accelerator": "cpu"}
    cfg.update(no_devices)

    expected_params = {"gpus": None, "replace_sampler_ddp": False, "accelerator": "cpu", "devices": 1, "strategy": None}
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("no_devices", [{}, {"devices": None}])
def test_no_devices_gpu_accelerator(no_devices: TCfg) -> None:
    cfg = {"accelerator": "gpu"}
    cfg.update(no_devices)

    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": "gpu",
        "devices": devices,
        "strategy": DDPPlugin() if devices > 1 else None,
    }
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("no_accelerator", [{}, {"accelerator": None}])
@pytest.mark.parametrize("int_devices", [1, 2, 4])
def case_int_devices_no_accelerator(no_accelerator: TCfg, int_devices: int) -> None:
    cfg = {"devices": int_devices}
    cfg.update(no_accelerator)

    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": int_devices,
        "strategy": DDPPlugin() if int_devices > 1 else None,
    }
    compare_params(cfg, expected_params)


@pytest.mark.parametrize("no_accelerator", [{}, {"accelerator": None}])
@pytest.mark.parametrize("list_devices", [[1], [1, 2], [0, 3], [0, 1, 2, 5]])
def case_list_devices_no_accelerator(no_accelerator: TCfg, list_devices: List[int]) -> None:
    cfg = {"devices": list_devices}
    cfg.update(no_accelerator)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = len(list_devices) if accelerator == "cpu" else list_devices
    expected_params = {
        "gpus": None,
        "replace_sampler_ddp": False,
        "accelerator": accelerator,
        "devices": devices,
        "strategy": DDPPlugin() if len(list_devices) > 1 else None,
    }
    compare_params(cfg, expected_params)
