import ast
import importlib
import json
from itertools import chain
from pathlib import Path
from typing import List

import pytest

from oml.const import PROJECT_ROOT

LIBS_TO_IGNORE = ["torch_xla"]


def get_files_with_imports() -> List[str]:
    folder_with_scripts = PROJECT_ROOT / "oml"
    scriptes_files = sorted(str(fname.relative_to(PROJECT_ROOT)) for fname in folder_with_scripts.rglob("*.py"))
    scriptes_files.remove("oml/utils/misc.py")

    folder_with_tests = PROJECT_ROOT / "tests"
    tests_files = sorted(str(fname.relative_to(PROJECT_ROOT)) for fname in folder_with_tests.rglob("*.py"))

    folder_with_examples = PROJECT_ROOT / "examples"
    examples_files = sorted(str(fname.relative_to(PROJECT_ROOT)) for fname in folder_with_tests.rglob("*.py"))

    notebooks_files = sorted(
        str(fname.relative_to(PROJECT_ROOT))
        for fname in folder_with_examples.rglob("*.ipynb")
        if fname.parent.name != ".ipynb_checkpoints"
    )

    return scriptes_files + tests_files + examples_files + notebooks_files


def find_imports_in_script(fname: Path) -> List[str]:
    with open(fname, "r") as f:
        script_code = f.read()

    imports = find_imports(script_code)

    return imports


def find_imports_in_notebook(fname: Path) -> List[str]:
    with open(fname, "r") as f:
        notebook_raw = json.load(f)

    code_lines = [cell["source"] for cell in notebook_raw["cells"] if cell["cell_type"] == "code"]
    code_lines = list(chain(*code_lines))
    # cells with single row haven't new line symbol
    code_lines = [line if line.endswith("\n") else f"{line}\n" for line in code_lines]
    # remove jupyter magic commands
    code_lines = filter(lambda x: not (x.startswith("%") or x.startswith("!")), code_lines)

    notebook_code = "".join(code_lines)
    imports = find_imports(notebook_code)

    return imports


def find_imports(code: str) -> List[str]:
    code = ast.parse(code)
    imports = set()
    for node in ast.walk(code):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imports.add(node.module)
    return list(imports)


@pytest.mark.parametrize("file", get_files_with_imports())
def test_project_imports(file: str) -> None:
    fname = PROJECT_ROOT / file
    if fname.suffix == ".py":
        imports = find_imports_in_script(fname)
    elif fname.suffix == ".ipynb":
        imports = find_imports_in_notebook(fname)
    else:
        raise ValueError

    for library in imports:
        try:
            importlib.import_module(library)
        except Exception as e:
            if library in LIBS_TO_IGNORE:
                pass
            else:
                raise e
