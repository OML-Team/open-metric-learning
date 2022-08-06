import subprocess
import warnings
from pathlib import Path

import pytest

from oml.const import PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_examples/test_runs_via_configs/"


@pytest.mark.parametrize(
    "file",
    [
        SCRIPTS_PATH / "train_mock.py",
        SCRIPTS_PATH / "val_mock.py",
    ],
)
def test_mock_examples(file: Path) -> None:
    subprocess.run(["python", str(file)], check=True)
