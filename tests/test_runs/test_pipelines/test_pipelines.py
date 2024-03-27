import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List, Tuple

import dotenv
import pytest
import torch
import yaml  # type: ignore

from oml.const import DOTENV_PATH, PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_runs/test_pipelines/"

dotenv.load_dotenv(DOTENV_PATH)  # we need to load tokens for cloud loggers (Neptune, W & B)


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


def run(file: str, accelerator: str, devices: int, need_rm_logs: bool = True) -> None:
    cmd = f"python {str(SCRIPTS_PATH / file)} ++accelerator='{accelerator}' ++devices='{devices}'"
    subprocess.run(cmd, check=True, shell=True)

    if need_rm_logs:
        rm_logs(cfg_name=SCRIPTS_PATH / "configs" / file.replace(".py", ".yaml"))


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_and_validate(accelerator: str, devices: int) -> None:
    run("train.py", accelerator, devices, need_rm_logs=False)
    # it takes checkpoints from the train stage
    run("validate.py", accelerator, devices, need_rm_logs=False)

    for file in ["train.py", "validate.py"]:
        rm_logs(cfg_name=SCRIPTS_PATH / "configs" / file.replace(".py", ".yaml"))


@pytest.mark.long
@pytest.mark.needs_optional_dependency
@pytest.mark.skipif(os.getenv("TEST_CLOUD_LOGGERS") != "yes", reason="To have more control.")
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_with_bboxes(accelerator: str, devices: int) -> None:
    run("train_with_bboxes.py", accelerator, devices)


@pytest.mark.long
@pytest.mark.needs_optional_dependency
@pytest.mark.skipif(os.getenv("TEST_CLOUD_LOGGERS") != "yes", reason="To have more control.")
@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_with_sequence(accelerator: str, devices: int) -> None:
    run("train_with_sequence.py", accelerator, devices)


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
def test_predict(accelerator: str, devices: int) -> None:
    run("predict.py", accelerator, devices)
