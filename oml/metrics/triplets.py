from collections import defaultdict
from typing import Any, Dict, Optional

from oml.const import EMBEDDINGS_KEY
from oml.functional.metrics import calculate_accuracy_on_triplets
from oml.interfaces.metrics import IBasicMetric
from oml.utils.misc_torch import OnlineAvgDict


class AccuracyOnTriplets(IBasicMetric):
    metric_name = "accuracy_on_triplets"

    def __init__(self, embeddings_key: str = EMBEDDINGS_KEY, categories_key: Optional[str] = None):
        """

        Args:
            embeddings_key: key to get embeddings which are tensors with the following structure:
                            [anchor, positive, negative, anchor, positive, negative...]
                             with the shape of [n_triplets * 3, features_dim]
            categories_key: key to get categories, which are integers or strings
        """
        self.embeddings_key = embeddings_key
        self.categories_key = categories_key

        self.avg_online = defaultdict(OnlineAvgDict)  # type: ignore

    def setup(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
        self.avg_online = defaultdict(OnlineAvgDict)  # type: ignore

    def update_data(self, data_dict: Dict[str, Any]) -> None:  # type: ignore
        acc_batch = calculate_accuracy_on_triplets(data_dict[self.embeddings_key], reduce_mean=False)

        self.avg_online[self.overall_categories_key].update({self.metric_name: list(map(float, acc_batch))})

        if self.categories_key is not None:
            for acc, category in zip(acc_batch, data_dict[self.categories_key]):
                self.avg_online[category].update({self.metric_name: float(acc)})

    def compute_metrics(self) -> Dict[str, Any]:  # type: ignore
        return {metric: avg_online.get_dict_with_results() for metric, avg_online in self.avg_online.items()}


__all__ = ["AccuracyOnTriplets"]
