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


@pytest.mark.long
@pytest.mark.parametrize(
    "fname,start_indicator,end_indicator",
    [
        ("extractor/retrieval_usage.md", "[comment]:usage-retrieval-start\n", "[comment]:usage-retrieval-end\n"),
        ("extractor/retrieval_usage_streaming.md", "[comment]:usage-streaming-retrieval-start\n", "[comment]:usage-streaming-retrieval-end\n"),  # noqa
        ("extractor/train_2loaders_val.md", "[comment]:lightning-2loaders-start\n", "[comment]:lightning-2loaders-end\n"),  # noqa
        ("extractor/train_val_img_txt.md", "[comment]:train-val-img-start\n", "[comment]:train-val-img-end\n"),
        pytest.param("extractor/train_val_img_txt.md", "[comment]:train-val-txt-start\n", "[comment]:train-val-txt-end\n", marks=pytest.mark.needs_optional_dependency),  # noqa
        ("extractor/train_val_pl.md", "[comment]:lightning-start\n", "[comment]:lightning-end\n"),
        pytest.param("extractor/train_val_pl_ddp.md", "[comment]:lightning-ddp-start\n", "[comment]:lightning-ddp-end\n", marks=pytest.mark.needs_optional_dependency),  # noqa
        ("extractor/handling_categories.md", "[comment]:categories-start\n", "[comment]:categories-end\n"),
        ("extractor/val_with_sequence.md", "[comment]:val-with-sequence-start\n", "[comment]:val-with-sequence-end\n"),
        ("postprocessing/predict.md", "[comment]:postprocessor-pred-start\n", "[comment]:postprocessor-pred-end\n"),
        ("postprocessing/train_val.md", "[comment]:postprocessor-start\n", "[comment]:postprocessor-end\n"),
        ("postprocessing/algo.md", "[comment]:algo-pp-start\n", "[comment]:algo-pp-end\n"),
        ("zoo/models_usage.md", "[comment]:zoo-image-start\n", "[comment]:zoo-image-end\n"),
        pytest.param("zoo/models_usage.md", "[comment]:zoo-text-start\n", "[comment]:zoo-text-end\n", marks=pytest.mark.needs_optional_dependency),  # noqa
        ("retrieval_format.md", "[comment]:dataset-start\n", "[comment]:dataset-end\n")
    ],
)  # fmt: skip
def test_code_blocks_in_readme(fname: str, start_indicator: str, end_indicator: str) -> None:
    code = find_code_block(PROJECT_ROOT / "docs/readme/examples_source" / fname, start_indicator, end_indicator)
    tmp_fname = "tmp.py"

    with open(tmp_fname, "w") as f:
        f.write(code)

    try:
        subprocess.run(f"python {tmp_fname}", check=True, shell=True)
    finally:
        Path(tmp_fname).unlink()


def test_minimal_pipeline_example() -> None:
    readme_file = PROJECT_ROOT / "pipelines" / "README.md"
    registry_text = find_code_block(readme_file, "[comment]:registry-start\n", "[comment]:registry-end\n")
    pipeline_text = find_code_block(readme_file, "[comment]:pipeline-start\n", "[comment]:pipeline-end\n")
    config_text = find_code_block(readme_file, "[comment]:config-start\n", "[comment]:config-end\n")
    validate_text = find_code_block(readme_file, "[comment]:script-start\n", "[comment]:script-end\n")
    command_text = find_code_block(readme_file, "[comment]:shell-start\n", "[comment]:shell-end\n")

    with open("registry.py", "w") as r:
        r.write(registry_text)

    with open("pipeline.py", "w") as r:
        r.write(pipeline_text)

    with open("config.yaml", "w") as c:
        c.write(config_text)

    with open("validate.py", "w") as r:
        r.write(validate_text)

    try:
        subprocess.run(command_text, check=True, shell=True)
    except Exception as e:
        raise e
    finally:
        for fname in ["registry.py", "config.yaml", "validate.py", "pipeline.py"]:
            Path(fname).unlink()
