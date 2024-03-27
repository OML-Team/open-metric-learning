from abc import abstractmethod

from matplotlib import pyplot as plt
from pytorch_lightning.loggers import Logger as LightningLogger

from oml.const import TCfg


class IFigureLogger:
    @abstractmethod
    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        raise NotImplementedError()


class IPipelineLogger(LightningLogger, IFigureLogger):
    @abstractmethod
    def log_pipeline_info(self, cfg: TCfg) -> None:
        raise NotImplementedError()
