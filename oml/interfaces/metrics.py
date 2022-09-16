from abc import ABC, abstractmethod
from typing import Any

from oml.const import OVERALL_CATEGORIES_KEY


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
        Has to be called before the first call of

        >>> self.update_data()

        """
        raise NotImplementedError()

    @abstractmethod
    def update_data(self, *args: Any, **kwargs: Any) -> Any:
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


class IBasicMetricDDP(IBasicMetric):
    """
    This is an extension of a base metric interface to work in DDP mode

    """

    def sync(self) -> None:
        """
        Method aggregates data in DDP mode before metric calculations
        """
        raise NotImplementedError()


__all__ = ["IBasicMetric", "IBasicMetricDDP"]
