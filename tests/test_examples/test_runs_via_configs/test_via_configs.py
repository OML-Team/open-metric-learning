import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytest
import torch
import yaml  # type: ignore
from omegaconf import OmegaConf

from oml.const import PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_examples/test_runs_via_configs/"


def accelerator_devices_pairs() -> List[Tuple[str, str]]:  # type: ignore
    pairs = [("cpu", "1"), ("cpu", "[0, 2]")]

    if torch.cuda.is_available():
        pairs += [("gpu", "1")]
        if torch.cuda.device_count() > 1:
            pairs += [("gpu", "[0, 1]")]

    return pairs  # type: ignore


def rm_logs(cfg_name: Path) -> None:
    with open(SCRIPTS_PATH / "configs" / cfg_name, "r") as f:
        cfg = yaml.safe_load(f)

    if ("logs_root" in cfg) and Path(cfg["logs_root"]).exists():
        shutil.rmtree(cfg["logs_root"])


def run(file: str, accelerator: str, devices: str) -> None:
    cmd = f"python {str(SCRIPTS_PATH / file)} ++accelerator='{accelerator}' ++devices='{devices}'"
    subprocess.run(cmd, check=True, shell=True)

    rm_logs(cfg_name=SCRIPTS_PATH / "configs" / file.replace(".py", ".yaml"))


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train(accelerator: str, devices: str) -> None:
    run("train_mock.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_val(accelerator: str, devices: str) -> None:
    run("val_mock.py", accelerator, devices)
