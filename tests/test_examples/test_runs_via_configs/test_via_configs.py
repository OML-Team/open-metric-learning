import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
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
    with pytest.raises(subprocess.CalledProcessError, match=r".returned non-zero exit status 1."):
        run("inference_images_mock.py", accelerator, devices)


@pytest.fixture()
def params_for_inference_on_folder(
    accelerator: str, devices: int, cleanup: bool = True
) -> Iterator[Tuple[str, Path, Path]]:
    features_path: Path = MOCK_DATASET_PATH / "features1.json"
    command = (
        f"inference_images_mock.py ~dataframe_name dataset_root={MOCK_DATASET_PATH} " f"features_file={features_path}"
    )
    cfg_path = SCRIPTS_PATH / "configs" / "inference_images_mock.yaml"
    yield command, features_path, cfg_path
    features_path.unlink()


@pytest.fixture()
def params_for_inference_on_df(
    accelerator: str, devices: int, cleanup: bool = True
) -> Iterator[Tuple[str, Path, Path]]:
    cfg_path = SCRIPTS_PATH / "configs" / "inference_images_mock.yaml"
    features_path = MOCK_DATASET_PATH / "features2.json"

    command = (
        "inference_images_mock.py dataframe_name={df_name} "
        f"dataset_root={MOCK_DATASET_PATH} ~images_folder features_file={str(features_path)}"
    )
    yield command, features_path, cfg_path
    features_path.unlink()


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_inference_images_list(
    accelerator: str, devices: int, params_for_inference_on_folder: Tuple[str, Path, Path]
) -> None:
    command, features_path, cfg_path = params_for_inference_on_folder
    run(command, accelerator, devices, cfg_path=cfg_path)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_inference_dataframe_w_boxes(
    accelerator: str, devices: int, params_for_inference_on_df: Tuple[str, Path, Path]
) -> None:
    command, features_path, cfg_path = params_for_inference_on_df
    command = command.format(df_name=MOCK_DATASET_PATH / "df_with_bboxes.csv")
    run(
        command,
        accelerator,
        devices,
        cfg_path=cfg_path,
    )


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_inference_dataframe_without_boxes(
    accelerator: str, devices: int, params_for_inference_on_df: Tuple[str, Path, Path]
) -> None:
    command, features_path, cfg_path = params_for_inference_on_df
    command = command.format(df_name=MOCK_DATASET_PATH / "df.csv")
    run(command, accelerator, devices, cfg_path=cfg_path)


@pytest.mark.parametrize("accelerator, devices", accelerator_devices_pairs())
def test_inference_compare_features_from_df_and_path(
    accelerator: str,
    devices: int,
    params_for_inference_on_df: Tuple[str, Path, Path],
    params_for_inference_on_folder: Tuple[str, Path, Path],
) -> None:
    command_df, features_df_path, cfg_df_path = params_for_inference_on_df
    command_df = command_df.format(df_name=MOCK_DATASET_PATH / "df.csv")
    command_w_path, features_with_paths_location, cfg_w_path_location = params_for_inference_on_folder
    run(command_df, accelerator, devices, cfg_df_path)
    run(command_w_path, accelerator, devices, cfg_w_path_location)

    with features_df_path.open() as f_df, features_with_paths_location.open() as f_folder:
        features_df = json.load(f_df)
        features_from_path = json.load(f_folder)

    sorted_paths_feats_df = sorted(
        zip(features_df["filenames"], features_df["features"]), key=lambda path_feat: path_feat[0]
    )
    sorted_paths_feats_from_path = sorted(
        zip(features_from_path["filenames"], features_from_path["features"]), key=lambda path_feat: path_feat[0]
    )

    for (_, feats1), (_, feats2) in zip(sorted_paths_feats_df, sorted_paths_feats_from_path):
        assert np.isclose(feats1, feats2).all()
