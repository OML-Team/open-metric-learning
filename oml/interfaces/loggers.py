from abc import abstractmethod

from matplotlib import pyplot as plt
from pytorch_lightning.loggers import Logger as LightningLogger

from oml.const import TCfg


class IPipelineLogger(LightningLogger):
    @abstractmethod
    def log_experiment_info(self, cfg: TCfg) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_figure(self, fig: plt.Figure, title: str, idx: int) -> None:
        raise NotImplementedError()
