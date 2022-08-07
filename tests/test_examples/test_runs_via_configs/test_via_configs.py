import subprocess
import warnings
from pathlib import Path

import pytest

from oml.const import PROJECT_ROOT

warnings.filterwarnings("ignore")

SCRIPTS_PATH = PROJECT_ROOT / "tests/test_examples/test_runs_via_configs/"


@pytest.mark.parametrize("file", ["train_mock.py", "val_mock.py"])
def test_mock_examples(file: str) -> None:
    file = SCRIPTS_PATH / file
    subprocess.run(["python", str(file)], check=True)
