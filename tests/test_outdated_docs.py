import ast
from pathlib import Path

from oml.const import PROJECT_ROOT


def check_docstrings_in_file(filename: Path) -> None:
    with open(filename, "r") as file:
        content = file.read()

    tree = ast.parse(content)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)

            if docstring is not None:
                actual_args = set([arg.arg for arg in node.args.args]) - {"self", "cls"}
                existing_indicator = [f" {arg}:" not in docstring for arg in actual_args]

                if set(existing_indicator) == {True, False}:  # means some arguments exist, but some of them not
                    # TODO: we are checking only missed arguments now
                    raise ValueError(f"Incorrect docstring for {node.name} in {filename}")


def check_docstrings_in_dir(directory: Path) -> None:
    for fname in Path(directory).glob("**/*.py"):
        check_docstrings_in_file(fname)


def test_docstrings() -> None:
    check_docstrings_in_dir(PROJECT_ROOT)
