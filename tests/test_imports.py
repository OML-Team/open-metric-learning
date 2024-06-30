import ast
import importlib
import json
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import pytest

from oml.const import PROJECT_ROOT

LIBS_TO_IGNORE = [
    "torch_xla",
    "pytorch_grad_cam",
    "wandb",
    "neptune",
    "clearml",
    "IPython",
    "transformers",
    "torchaudio",
]

NEED_TO_TEST_NOTEBOOKS = True


def get_imports_from_files() -> List[Tuple[str, str]]:
    files = get_files_with_imports()

    file_import_pairs = []

    for file in files:
        fname = PROJECT_ROOT / file
        if fname.suffix == ".py":
            imports = find_imports_in_script(fname)
        elif fname.suffix == ".ipynb":
            imports = find_imports_in_notebook(fname)
        else:
            raise ValueError

        file_import_pairs.extend(list(zip([file] * len(imports), imports)))

    return file_import_pairs


def get_files_with_imports() -> List[str]:
    files = []

    folder_with_scripts = PROJECT_ROOT / "oml"
    scriptes_files = sorted(str(fname.relative_to(PROJECT_ROOT)) for fname in folder_with_scripts.rglob("*.py"))
    files.extend(scriptes_files)

    folder_with_tests = PROJECT_ROOT / "tests"
    tests_files = sorted(str(fname.relative_to(PROJECT_ROOT)) for fname in folder_with_tests.rglob("*.py"))
    files.extend(tests_files)

    pipelines_folder = PROJECT_ROOT / "pipelines"
    pipelines_files = sorted(str(fname.relative_to(PROJECT_ROOT)) for fname in folder_with_tests.rglob("*.py"))
    files.extend(pipelines_files)

    notebooks_files = sorted(
        str(fname.relative_to(PROJECT_ROOT))
        for fname in pipelines_folder.rglob("*.ipynb")
        if fname.parent.name != ".ipynb_checkpoints"
    )

    if NEED_TO_TEST_NOTEBOOKS:
        files.extend(notebooks_files)

    return files


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


@pytest.mark.parametrize("file,lib", get_imports_from_files())
def test_project_imports(file: str, lib: str) -> None:
    if any(lib.startswith(ignore_lib) for ignore_lib in LIBS_TO_IGNORE):
        pass
    else:
        importlib.import_module(lib)
