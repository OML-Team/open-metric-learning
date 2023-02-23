import os
from pathlib import Path

from oml.const import PROJECT_ROOT


def test_readme_was_built_correctly() -> None:
    readme_tmp_file = "README_tmp"
    os.chdir(PROJECT_ROOT)
    os.system(f"make build_readme README_FILE={readme_tmp_file}.md")

    with open("README.md", "r") as f1:
        readme = f1.read()

    with open(f"{readme_tmp_file}.md", "r") as f2:
        readme_tmp = f2.read()

    assert readme == readme_tmp, (
        "Automatically generated readme and yours are not the same. The reason is that"
        "you tried to change the README.md inplace, instead of changing its source components"
        "in docs/readme folder."
    )

    Path(f"{readme_tmp_file}.md").unlink()
