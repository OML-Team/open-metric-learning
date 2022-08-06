import subprocess
import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.mark.parametrize(
    "file",
    [
        "tests/test_examples/test_runs_via_configs/train_mock.py",
        "tests/test_examples/test_runs_via_configs/val_mock.py",
    ],
)
def test_mock_examples(file: str) -> None:
    subprocess.run(["python", file], check=True)
