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


def find_code_in_readme(start_indicator: str, end_indicator: str) -> List[str]:
    return find_code_in_file(PROJECT_ROOT / "README.md", start_indicator, end_indicator)


def find_code_in_docs(start_indicator: str, end_indicator: str) -> List[str]:
    n_spaces = 4  # number of leading white spaces in rst code blocks
    start_indicator = " " * n_spaces + start_indicator
    end_indicator = " " * n_spaces + end_indicator

    rst_file = PROJECT_ROOT / "docs" / "source" / "examples" / "python.rst"
    code_from_docs = find_code_in_file(rst_file, start_indicator, end_indicator)

    code_from_docs = code_from_docs[2:-1]  # in rst files few have few extra lines around the code block
    code_from_docs = list(map(lambda x: x[n_spaces:] if x.startswith(" " * n_spaces) else x, code_from_docs))

    return code_from_docs


@pytest.mark.parametrize(
    "start_indicator,end_indicator",
    [
        ("[comment]:vanilla-train-start\n", "[comment]:vanilla-train-end\n"),
        ("[comment]:vanilla-validation-start\n", "[comment]:vanilla-validation-end\n"),
        ("[comment]:lightning-start\n", "[comment]:lightning-end\n"),
        ("[comment]:checkpoint-start\n", "[comment]:checkpoint-end\n"),
    ],
)
def test_code_blocks_in_docs(start_indicator: str, end_indicator: str) -> None:
    code_from_readme = find_code_in_readme(start_indicator, end_indicator)
    code_from_docs = find_code_in_docs(start_indicator, end_indicator)

    assert (
        code_from_docs == code_from_readme
    ), "Code snippets in the documentation file and in the readme have to be equal"

    code_to_exec = "".join(code_from_readme)
    print(code_to_exec)
    exec(code_to_exec)
    assert True
