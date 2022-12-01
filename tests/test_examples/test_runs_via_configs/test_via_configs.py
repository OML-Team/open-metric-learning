import shutil
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch
import yaml  # type: ignore

from oml.const import MOCK_DATASET_PATH, PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_examples/test_runs_via_configs/"


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


def run(file: str, accelerator: str, devices: int, cfg_path: Optional[Path] = None) -> None:
    cmd = f"python {str(SCRIPTS_PATH / file)} ++accelerator='{accelerator}' ++devices='{devices}'"
    subprocess.run(cmd, check=True, shell=True)

    if cfg_path is None:
        cfg_path = SCRIPTS_PATH / "configs" / file.replace(".py", ".yaml")
    rm_logs(cfg_name=cfg_path)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train(accelerator: str, devices: int) -> None:
    run("train.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_with_bboxes(accelerator: str, devices: int) -> None:
    run("train_with_bboxes.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_train_with_categories(accelerator: str, devices: int) -> None:
    run("train_with_categories.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_val(accelerator: str, devices: int) -> None:
    run("val_mock.py", accelerator, devices)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_inference_error_two_sources_provided(accelerator: str, devices: int) -> None:
    try:
        run("inference_images_mock.py", accelerator, devices)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("This test should raise exeption for two input sources provided")


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_inference_images_list(accelerator: str, devices: int) -> None:
    run(
        f"inference_images_mock.py ~dataframe_name dataset_root={MOCK_DATASET_PATH}",
        accelerator,
        devices,
        cfg_path=SCRIPTS_PATH / "configs" / "inference_images_mock.yaml",
    )
