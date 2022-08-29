import shutil
import subprocess
import warnings
from pathlib import Path

import yaml  # type: ignore

from oml.const import PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_examples/test_runs_via_configs/"


def rm_logs(cfg_name: Path) -> None:
    with open(SCRIPTS_PATH / "configs" / cfg_name, "r") as f:
        cfg = yaml.safe_load(f)

    if ("logs_root" in cfg) and Path(cfg["logs_root"]).exists():
        shutil.rmtree(cfg["logs_root"])


def run(file: str) -> None:
    subprocess.run(["python", str(SCRIPTS_PATH / file)], check=True)

    rm_logs(cfg_name=SCRIPTS_PATH / "configs" / file.replace(".py", ".yaml"))


def test_train() -> None:
    run("train.py")


def test_train_with_bboxes() -> None:
    run("train_with_bboxes.py")


def test_train_with_categories() -> None:
    run("train_with_categories.py")


def test_val() -> None:
    run("val_mock.py")
