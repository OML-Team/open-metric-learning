import subprocess
from pathlib import Path

import pytest

from oml.const import PROJECT_ROOT


def find_code_block(file: Path, start_indicator: str, end_indicator: str) -> str:
    with open(file) as f:
        text = f.readlines()

    i = text.index(start_indicator) + 2
    j = text.index(end_indicator) - 1

    code_block = "".join(text[i:j])

    print(code_block)

    return code_block


@pytest.mark.parametrize(
    "fname,start_indicator,end_indicator",
    [
        ("extractor/train.md", "[comment]:vanilla-train-start\n", "[comment]:vanilla-train-end\n"),
        ("extractor/val.md", "[comment]:vanilla-validation-start\n", "[comment]:vanilla-validation-end\n"),
        ("extractor/train_val_pl.md", "[comment]:lightning-start\n", "[comment]:lightning-end\n"),
        ("extractor/train_val_pl_ddp.md", "[comment]:lightning-ddp-start\n", "[comment]:lightning-ddp-end\n"),
        ("extractor/retrieval_usage.md", "[comment]:usage-retrieval-start\n", "[comment]:usage-retrieval-end\n"),
        ("zoo/models_usage.md", "[comment]:zoo-start\n", "[comment]:zoo-end\n"),
        ("postprocessing/train_val.md", "[comment]:postprocessor-start\n", "[comment]:postprocessor-end\n"),
    ],
)
def test_code_blocks_in_readme(fname: str, start_indicator: str, end_indicator: str) -> None:
    code = find_code_block(PROJECT_ROOT / "docs/readme/examples_source" / fname, start_indicator, end_indicator)
    tmp_fname = "tmp.py"

    with open(tmp_fname, "w") as f:
        f.write(code)

    try:
        subprocess.run(f"python {tmp_fname}", check=True, shell=True)
    finally:
        Path(tmp_fname).unlink()
