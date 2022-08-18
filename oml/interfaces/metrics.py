from abc import ABC, abstractmethod
from typing import Any

from oml.const import MAIN_CATEGORY_KEY


class IBasicMetric(ABC):
    metric_name = "BASE"
    main_category_key = MAIN_CATEGORY_KEY

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> Any:
        """
        Method for preparing metrics for work: memory allocation, placeholder preparation, etc.
        Called before the first call of 'update_data'.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_data(self, *args: Any, **kwargs: Any) -> Any:
        """
        Method for passing data to calculate the metric.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_metrics(self, *args: Any, **kwargs: Any) -> Any:
        """
        Method for obtaining dictionary with metric on categories level
        The main metric calculated for all of the categories is available by the key:
        >>> self.main_category_key
        """
        raise NotImplementedError()


__all__ = ["IBasicMetric"]
