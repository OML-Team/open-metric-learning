import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import yaml  # type: ignore

from oml.const import PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_runs/test_pipelines/"


def accelerator_devices_pairs() -> List[Tuple[str, int]]:
    pairs = [("cpu", 1), ("cpu", 2)]

    if torch.cuda.is_available():
        pairs += [("gpu", 1)]
        if torch.cuda.device_count() > 1:
            pairs += [("gpu", 2)]

    return pairs


def rm_logs(cfg_name: Path) -> None:
    with open(SCRIPTS_PATH / "configs" / cfg_name, "r") as f:
        cfg = yaml.safe_load(f)

    if ("logs_root" in cfg) and Path(cfg["logs_root"]).exists():
        shutil.rmtree(cfg["logs_root"])


def run(file: str, accelerator: str, devices: int, rm_log=True) -> None:
    cmd = f"python {str(SCRIPTS_PATH / file)} ++accelerator='{accelerator}' ++devices='{devices}'"
    subprocess.run(cmd, check=True, shell=True)
    if rm_log:
        rm_logs(cfg_name=SCRIPTS_PATH / "configs" / file.replace(".py", ".yaml"))
        
        
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train(accelerator: str, devices: int) -> None:
    run("train.py", accelerator, devices)


@pytest.mark.long
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_with_bboxes(accelerator: str, devices: int) -> None:
    run("train_with_bboxes.py", accelerator, devices)


@pytest.mark.long
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_with_categories(accelerator: str, devices: int) -> None:
    run("train_with_categories.py", accelerator, devices)


@pytest.mark.long
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_arcface_with_categories(accelerator: str, devices: int) -> None:
    run("train_arcface_with_categories.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_postprocessor(accelerator: str, devices: int) -> None:
    run("train_postprocessor.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_validation(accelerator: str, devices: int) -> None:
    run("validate.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_predict(accelerator: str, devices: int) -> None:
    run("predict.py", accelerator, devices)

                #   test train->load piplines arface   #
#resnet
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_arface_resnet(accelerator: str, devices: int) -> None:
    run("train_arc_resnet.py", accelerator, devices, rm_log=False)

@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_valid_arface_resnet(accelerator: str, devices: int) -> None:
    run("valid_arc_resnet.py", accelerator, devices)

#dino
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_arface_dino(accelerator: str, devices: int) -> None:
    run("train_arc_dino.py", accelerator, devices, rm_log=False)

@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_valid_arface_dino(accelerator: str, devices: int) -> None:
    file = "valid_arc_dino.py"
    run(file, accelerator, devices)
