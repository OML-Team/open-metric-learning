from abc import ABC, abstractmethod
from typing import Any, List

from oml.const import OVERALL_CATEGORIES_KEY


class IBasicMetric(ABC):
    metric_name = "BASE"
    overall_categories_key = OVERALL_CATEGORIES_KEY

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
        The output must be in the following format:
        {
            "self.overall_categories_key": {"metric1": v1, "metric2": v2},
            "category1": {"metric1": v1, "metric2": v2},
            "category2": {"metric1": v1, "metric2": v2}
        }
        Where "category1" and "category2" are optional.

        """

        raise NotImplementedError()

    @abstractmethod
    def get_keys_for_metric(self) -> List[str]:
        """
        Method return list of keys which is neccesary to get from batch to calculate metrics. We use it to reduce
        overhead of gathering data from different devices
        """
        raise NotImplementedError()

    def sync(self) -> None:
        """
        Method aggregate data in DDP mode before metric calculations
        """
        raise NotImplementedError()


__all__ = ["IBasicMetric"]
