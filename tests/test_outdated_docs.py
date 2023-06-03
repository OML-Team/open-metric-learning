import ast
from pathlib import Path
from typing import List, Optional, Tuple

from oml.const import PROJECT_ROOT


def parse_arg_in_docstring_row(row: str) -> Tuple[Optional[str], Optional[int]]:
    # We assume Google docstring format
    if ":" not in row:
        return None, None
    else:
        prefix = row.split(":")[0]
        offset = len(prefix) - len(prefix.lstrip(" "))
        return prefix[offset:], offset


def parse_args_in_docstring(docstring: str) -> List[str]:
    # 1. Find first docstring row if exists
    row_first_arg = None
    docstring_rows = docstring.split("\n")

    for i, row in enumerate(docstring_rows):
        if "Args:" in row:
            row_first_arg = i + 1
            break

    parsed_args: List[str] = []

    if row_first_arg is None:
        return parsed_args

    # 2. Parse the row that has to contain the first argument and check its offset
    first_arg, first_offset = parse_arg_in_docstring_row(docstring_rows[row_first_arg])
    assert first_arg
    parsed_args.append(first_arg)

    # 3. Parse rest of the rows, check the corresponding offsets
    for row in docstring_rows[row_first_arg + 1 :]:
        if ("Returns:" in row) or ("Raises:" in row):
            break

        arg, offset = parse_arg_in_docstring_row(row)
        if arg and (offset == first_offset):
            parsed_args.append(arg)

    return parsed_args


def check_docstrings_in_file(filename: Path) -> None:
    with open(filename, "r") as file:
        content = file.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)

            if docstring is not None:
                actual_args = set([arg.arg for arg in node.args.args]) - {"self", "cls"}
                docstring_args = set(parse_args_in_docstring(docstring))

                if docstring_args and (actual_args != docstring_args) and ("*_" not in docstring_args):
                    raise ValueError(
                        f"Incorrect docstring for {node.name} in {filename}."
                        f"Actual args are: {actual_args}\n"
                        f"Docstring args are: {docstring_args}"
                    )


def check_docstrings_in_dir(directory: Path) -> None:
    for fname in Path(directory).glob("**/*.py"):
        check_docstrings_in_file(fname)


def test_docstrings() -> None:
    check_docstrings_in_dir(PROJECT_ROOT / "oml")
