from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from torch import LongTensor

from oml.const import OVERALL_CATEGORIES_KEY

TIndices = Union[LongTensor, List[int]]


class IBasicMetric(ABC):
    """
    This is a base interface for the objects that calculate metrics.

    """

    metric_name = "BASE"
    overall_categories_key = OVERALL_CATEGORIES_KEY

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> Any:
        """
        Method for preparing metrics to work: memory allocation, placeholder preparation, etc.
        Has to be called before the first call of ``self.update_data()``.

        """
        raise NotImplementedError()

    @abstractmethod
    def update_data(self, data: Dict[str, Any], indices: TIndices) -> None:
        """
        Method for passing data to calculate the metrics later on.

        """
        raise NotImplementedError()

    @abstractmethod
    def compute_metrics(self, *args: Any, **kwargs: Any) -> Any:
        """
        The output must be in the following format:

        .. code-block:: python

            {
                "self.overall_categories_key": {"metric1": ..., "metric2": ...},
                "category1": {"metric1": ..., "metric2": ...},
                "category2": {"metric1": ..., "metric2": ...}
            }

        Where ``category1`` and ``category2`` are optional.
        """

        raise NotImplementedError()


class IMetricVisualisable(IBasicMetric):
    """
    This is an interface for all metrics which can visualize themselves.

    """

    @abstractmethod
    def visualize(self) -> Tuple[Collection[plt.Figure], Collection[str]]:
        """
        Method which returns results of visualization and titles for logging.

        """
        raise NotImplementedError()

    @abstractmethod
    def ready_to_visualize(self) -> bool:
        """
        Method which checks if visualization can be done.

        """
        raise NotImplementedError()


__all__ = ["IBasicMetric", "IMetricVisualisable"]
