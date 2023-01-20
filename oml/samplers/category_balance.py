import math
from collections import Counter
from typing import Dict, Iterator, List, Union

import numpy as np

from oml.interfaces.samplers import IBatchSampler
from oml.utils.misc import smart_sample


class CategoryBalanceSampler(IBatchSampler):
    """
    This sampler takes ``n_instances`` for each of the ``n_labels`` for each of the
    ``n_categories`` to form the batches.
    Thus, the batch size is ``n_instances x n_labels x n_categories``.

    Note, to form an epoch of batches we simply sample ``L / n_labels`` batches with repetition.

    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        label2category: Dict[int, Union[str, int]],
        n_categories: int,
        n_labels: int,
        n_instances: int,
        resample_labels: bool = False,
        weight_categories: bool = True,
    ):
        """

        Args:
            labels: Labels to sample from
            label2category: Mapping from label to category
            n_categories: The desired number of categories to sample for each batch
            n_labels: The desired number of labels to sample for each category in batch
            n_instances: The desired number of samples to sample for each label in batch
            resample_labels: If ``True`` sample with repetition otherwise, otherwise
                raise an error in case of the labels lack in any category
            weight_categories: If ``True`` sample categories for each batch with weights proportional
                to the number of unique labels in the categories

        """
        unique_labels = set(labels)
        unique_categories = set(label2category.values())
        category2labels = {
            category: {label for label, cat in label2category.items() if category == cat}
            for category in unique_categories
        }
        for param in [n_categories, n_labels, n_instances]:
            if not isinstance(param, int):
                raise TypeError(f"{param.__name__} must be int, {type(param)} given")
        if not 1 <= n_categories <= len(unique_categories):
            raise ValueError(f"must be 1 <= n_categories <= {len(unique_categories)}, {n_categories} given")
        if not 1 < n_labels <= len(unique_labels):
            raise ValueError(f"must be 1 < n_labels <= {len(unique_labels)}, {n_labels} given")
        if n_instances <= 1:
            raise ValueError(f"must be not less than 1, {n_instances} given")
        if any(label not in label2category.keys() for label in unique_labels):
            raise ValueError("All the labels must have category")
        if any(label not in unique_labels for label in label2category.keys()):
            raise ValueError("All the labels from label2category mapping must be in the labels")
        if any(n <= 1 for n in Counter(labels).values()):
            raise ValueError("Each class must contain at least 2 instances to fit")
        if not resample_labels:
            if any(len(list(labs)) < n_labels for labs in category2labels.values()):
                raise ValueError(f"All the categories must have at least {n_labels} unique labels")

        unique_categories = sorted(list(unique_categories))

        self._labels = np.array(labels)
        self._label2category = label2category
        self.n_categories = n_categories
        self.n_labels = n_labels
        self.n_instances = n_instances

        self._batch_size = self.n_categories * self.n_labels * self.n_instances
        self._weight_categories = weight_categories

        self._label2index = {
            label: np.arange(len(self._labels))[self._labels == label].tolist() for label in sorted(list(unique_labels))
        }
        self._category2labels = {
            category: {label for label, cat in self._label2category.items() if category == cat}
            for category in unique_categories
        }
        category_weights = {cat: len(labels) / len(unique_labels) for cat, labels in self._category2labels.items()}
        self._category_weights = (
            [category_weights[cat] for cat in unique_categories] if self._weight_categories else None
        )
        self._batch_number = math.ceil(len(unique_labels) / self.n_labels)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._batch_number

    def __iter__(self) -> Iterator[List[int]]:
        epoch_indices = []
        for _ in range(self._batch_number):
            categories = np.random.choice(
                list(self._category2labels.keys()),
                size=self.n_categories,
                replace=False,
                p=self._category_weights,
            )
            batch_indices = []
            for category in categories:
                labels_available = list(self._category2labels[category])
                labels = smart_sample(array=labels_available, k=self.n_labels)
                for label in labels:
                    indices = self._label2index[label]
                    samples_indices = smart_sample(array=indices, k=self.n_instances)
                    batch_indices.extend(samples_indices)
            epoch_indices.append(batch_indices)
        return iter(epoch_indices)


__all__ = ["CategoryBalanceSampler"]
