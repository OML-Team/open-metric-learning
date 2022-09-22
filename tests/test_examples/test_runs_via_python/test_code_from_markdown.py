from pathlib import Path
from typing import List

import pytest

from oml.const import PROJECT_ROOT


def find_code_in_file(file: Path, start_indicator: str, end_indicator: str) -> List[str]:
    with open(file) as f:
        text = f.readlines()

    i = text.index(start_indicator) + 2
    j = text.index(end_indicator) - 1

    code_lines = text[i:j]

    return code_lines


@pytest.mark.parametrize(
    "filename,start_indicator,end_indicator",
    [
        ("python_examples.md", "[comment]:vanilla-train-start\n", "[comment]:vanilla-train-end\n"),
        ("python_examples.md", "[comment]:vanilla-validation-start\n", "[comment]:vanilla-validation-end\n"),
        ("python_examples.md", "[comment]:lightning-start\n", "[comment]:lightning-end\n"),
        ("zoo.md", "[comment]:checkpoint-start\n", "[comment]:checkpoint-end\n"),
    ],
)
def test_code_blocks_in_docs(filename: str, start_indicator: str, end_indicator: str) -> None:
    code_from_readme = find_code_in_file(PROJECT_ROOT / "docs" / "readme" / filename, start_indicator, end_indicator)

    code_to_exec = "".join(code_from_readme)
    print(code_to_exec)
    exec(code_to_exec)
    assert True
