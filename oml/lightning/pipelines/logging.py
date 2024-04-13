import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from lightning_fabric.utilities.logger import _flatten_dict
from lightning_fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import (
    Logger,
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


def prepare_config_to_logging(cfg: TCfg, sep: str = "|") -> Dict[str, Any]:
    cwd = Path.cwd().name
    flattened_dict = flatten_dict({**dictconfig_to_dict(cfg), **{"dir": cwd}}, sep=sep)
    return flattened_dict


def prepare_tags(cfg: TCfg) -> List[str]:
    cwd = Path.cwd().name
    tags = list(cfg.get("tags", [])) + [cfg.get("postfix", "")] + [cwd]
    tags = list(filter(lambda x: len(x) > 0, tags))
    return tags


class ClearMLLogger(Logger):
    def __init__(self, **kwargs: Any):
        try:
            from clearml import Task
        except ImportError as e:
            raise ModuleNotFoundError(
                "This contrib module requires clearml to be installed. "
                "You may install clearml using: \n pip install clearml \n"
            ) from e

        experiment_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("project_name", "task_name", "task_type", "offline_mode")
        }

        if kwargs.get("offline_mode", True):
            Task.set_offline(offline_mode=True)
            warnings.warn("ClearMLSaver: running in offline mode")

        # Try to retrieve current the ClearML Task before trying to create a new one
        self.task = Task.current_task()
        if self.task is None:
            self.task = Task.init(
                project_name=kwargs.get("project_name"),
                task_name=kwargs.get("task_name"),
                task_type=kwargs.get("task_type", Task.TaskTypes.training),
                **experiment_kwargs,
            )

        self.logger = self.task.get_logger()

    @property
    def name(self) -> str:
        return "ClearMLLogger"

    @property
    def version(self) -> Union[int, str]:
        return self.task.id

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.logger.flush()

    @rank_zero_only
    def log_hyperparams(self, params: Optional[Union[Dict[str, Any], Namespace]]) -> None:
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}
        params = _flatten_dict(params)

        self.task.connect(params)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"  # type: ignore
        for k, v in metrics.items():
            self.logger.report_scalar(title=k, series=k, iteration=step, value=v)


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
        # log config
        self.log_hyperparams(prepare_config_to_logging(cfg, sep="/"))

        # log tags
        for tag in prepare_tags(cfg):
            self.experiment.set_tag(run_id=self.run_id, key=tag, value="tag")

        # log transforms as files
        names_files = save_transforms_as_files(cfg)
        if names_files:
            for name, transforms_file in names_files:
                self.experiment.log_artifact(run_id=self.run_id, local_path=transforms_file, artifact_path=name)

        # log code
        self.experiment.log_artifacts(run_id=self.run_id, local_dir=OML_PATH, artifact_path="code")

        # log dataframe
        self.experiment.log_artifact(
            run_id=self.run_id,
            local_path=str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]),
            artifact_path="dataset",
        )

    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        self.experiment.log_figure(figure=fig, artifact_file=f"{title}.png", run_id=self.run_id)


class ClearMLPipelineLogger(ClearMLLogger, IPipelineLogger):
    def log_pipeline_info(self, cfg: TCfg) -> None:
        # log config
        self.log_hyperparams(prepare_config_to_logging(cfg))

        # log tags
        self.task.add_tags(prepare_tags(cfg))

        # log transforms as files
        names_files = save_transforms_as_files(cfg)
        if names_files:
            for name, transforms_file in names_files:
                self.task.upload_artifact(name=name, artifact_object=transforms_file)

        # log code
        self.task.upload_artifact(name="code", artifact_object=OML_PATH)

        # log dataframe
        self.task.upload_artifact(
            name="dataset",
            artifact_object=str(Path(cfg["dataset_root"]) / cfg["dataframe_name"]),
        )

    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        self.logger.report_matplotlib_figure(
            title=title,
            series="",
            figure=fig,
            iteration=idx,
            report_image=True,
        )


__all__ = [
    "IPipelineLogger",
    "TensorBoardPipelineLogger",
    "WandBPipelineLogger",
    "NeptunePipelineLogger",
    "MLFlowPipelineLogger",
    "ClearMLPipelineLogger",
]
