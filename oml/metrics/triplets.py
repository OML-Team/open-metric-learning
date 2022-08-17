from typing import Any, Dict, Optional

from oml.const import T_Str2Int_or_Int2Str
from oml.functional.metrics import calculate_accuracy_on_triplets
from oml.interfaces.metrics import IBasicMetric
from oml.utils.misc_torch import OnlineAvgDict


class AccuracyOnTriplets(IBasicMetric):
    metric_name = "accuracy_on_triplets"

    def __init__(
        self,
        embeddings_key: str = "embeddings",
        categories_key: Optional[str] = None,
        categories_names_mapping: Optional[T_Str2Int_or_Int2Str] = None,
    ):
        """

        Args:
            embeddings_key: key to get embeddings which are tensors with the following structure:
                            [anchor, positive, negative, anchor, positive, negative...]
                             with the shape of [n_triplets * 3, features_dim]
            categories_key: key to get categories, which are integers or strings
            categories_names_mapping: mapping from the category id to its name or vice versa
        """
        if (categories_names_mapping is not None) and (categories_key is None):
            raise ValueError(
                "You have not specified category key but specified the mapping for " "the categories at the same time."
            )

        self.embeddings_key = embeddings_key
        self.categories_key = categories_key
        self.categories_names_mapping = categories_names_mapping

        self.avg_online = OnlineAvgDict()

    def setup(self, *args: Any, **kwargs: Any) -> None:  # type: ignore
        self.avg_online = OnlineAvgDict()

    def update_data(self, data_dict: Dict[str, Any]) -> None:  # type: ignore
        acc_batch = calculate_accuracy_on_triplets(data_dict[self.embeddings_key], reduce_mean=False)

        self.avg_online.update({f"{self.metric_name}/OVERALL": list(map(float, acc_batch))})

        if self.categories_key is not None:
            categories = data_dict[self.categories_key]

            for acc, category in zip(acc_batch, categories):
                if self.categories_names_mapping is not None:
                    category = self.categories_names_mapping[category]
                self.avg_online.update({f"{self.metric_name}/{category}": float(acc)})

    def compute_metrics(self) -> Dict[str, Any]:  # type: ignore
        return self.avg_online.get_dict_with_results()


__all__ = ["AccuracyOnTriplets"]
