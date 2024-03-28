import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import (
    MLFlowLogger,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)

from oml.const import OML_PATH, TCfg
from oml.interfaces.loggers import IPipelineLogger
from oml.registry.transforms import save_transforms_as_files
from oml.utils.images.images import figure_to_nparray
from oml.utils.misc import dictconfig_to_dict, flatten_dict


def prepare_config_to_logging(cfg: TCfg) -> Dict[str, Any]:
    cwd = Path.cwd().name
    flattened_dict = flatten_dict({**dictconfig_to_dict(cfg), **{"dir": cwd}}, sep="|")
    return flattened_dict


def prepare_tags(cfg: TCfg) -> List[str]:
    cwd = Path.cwd().name
    tags = list(cfg.get("tags", [])) + [cfg.get("postfix", "")] + [cwd]
    tags = list(filter(lambda x: len(x) > 0, tags))
    return tags


class NeptunePipelineLogger(NeptuneLogger, IPipelineLogger):
    def log_pipeline_info(self, cfg: TCfg) -> None:
        warnings.warn(
            "Unfortunately, in the case of using Neptune, you may experience that long experiments are "
            "stacked and not responding. It's not an issue on OML's side, so, we cannot fix it."
        )
        self.log_hyperparams(prepare_config_to_logging(cfg))

        tags = prepare_tags(cfg)
        self.run["sys/tags"].add(tags)

        # log transforms as files
        for key, transforms_file in save_transforms_as_files(cfg):
            self.run[key].upload(transforms_file)

        # log source code
        source_files = list(map(lambda x: str(x), OML_PATH.glob("**/*.py"))) + list(
            map(lambda x: str(x), OML_PATH.glob("**/*.yaml"))
        )
        self.run["code"].upload_files(source_files)

        # log dataframe
        self.run["dataset"].upload(str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]))

    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        from neptune.types import File  # this is the optional dependency

        self.experiment[title].log(File.as_image(fig))


class WandBPipelineLogger(WandbLogger, IPipelineLogger):
    def log_pipeline_info(self, cfg: TCfg) -> None:
        # this is the optional dependency
        import wandb

        self.log_hyperparams(prepare_config_to_logging(cfg))

        tags = prepare_tags(cfg)
        self.experiment.tags = tags

        # log transforms as files
        keys_files = save_transforms_as_files(cfg)
        if keys_files:
            transforms = wandb.Artifact("transforms", type="transforms")
            for _, transforms_file in keys_files:
                transforms.add_file(transforms_file)
            self.experiment.log_artifact(transforms)

        # log source code
        code = wandb.Artifact("source_code", type="code")
        code.add_dir(OML_PATH, name="oml")
        self.experiment.log_artifact(code)

        # log dataset
        dataset = wandb.Artifact("dataset", type="dataset")
        dataset.add_file(str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]))
        self.experiment.log_artifact(dataset)

    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        fig_img = figure_to_nparray(fig)
        self.log_image(images=[fig_img], key=title)


class TensorBoardPipelineLogger(TensorBoardLogger, IPipelineLogger):
    def log_pipeline_info(self, cfg: TCfg) -> None:
        pass

    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        fig_img = figure_to_nparray(fig)
        self.experiment.add_image(title, np.transpose(fig_img, (2, 0, 1)), idx)


class MLFlowPipelineLogger(MLFlowLogger, IPipelineLogger):
    def log_pipeline_info(self, cfg: TCfg) -> None:
        pass

    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        pass
        # self.experiment.log_figure(figure=fig, artifact_file=title, run_id="todo")


__all__ = [
    "IPipelineLogger",
    "TensorBoardPipelineLogger",
    "WandBPipelineLogger",
    "NeptunePipelineLogger",
    "MLFlowPipelineLogger",
]
