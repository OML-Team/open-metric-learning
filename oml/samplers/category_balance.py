import math
from collections import Counter
from typing import Dict, Iterator, List, Union

import numpy as np
from torch.utils.data import Sampler

from oml.utils.misc import smart_sample


class CategoryBalanceBatchSampler(Sampler):
    """
    Let C is a set of categories in dataset, L is a set of labels in dataset:
    - select n_categories for the 1st batch from C
    - select n_labels for each of chosen categories for the 1st batch
    - select n_instances for each label for the 1st batch
    - select n_categories from C for the 2nd batch
    - select n_labels for each category for the 2nd batch
    - select n_instances for each label for the 2nd batch
    ...

    Behavior in corner cases:
    - If a label does not contain n_instances, a choice will be made with repetition.
    - If P % n_labels == 1 then one of the labels should be dropped
    - If a category does not contain n_labels, a choice will be made with repetition
    or an error will be raised depend on few_labels_number_policy
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        label2category: Dict[int, int],
        n_categories: int,
        n_labels: int,
        n_instances: int,
        resample_labels: bool = False,
        weight_categories: bool = True,
    ):
        """Init CategoryBalanceBatchSampler.

        Args:
            labels: Labels to sample from
            label2category: Mapping from label to category
            n_categories: Number of categories to sample for each batch
            n_labels: Number of labels to sample for each category in batch
            n_instances: Number of samples to sample for each label in batch
            resample_labels: If False raise an error in case of lack of labels in any categories,
                sample with repetition otherwise
            weight_categories: If True sample categories for each batch with weights proportional to the number of
                unique labels in the categories
        """
        super().__init__(self)
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
            raise ValueError(f"c must be 1 <= c <= {len(unique_categories)}, {n_categories} given")
        if not 1 < n_labels <= len(unique_labels):
            raise ValueError(f"p must be 1 < p <= {len(unique_labels)}, {n_labels} given")
        if n_instances <= 1:
            raise ValueError(f"k must be not less than 1, {n_instances} given")
        if any(label not in label2category.keys() for label in unique_labels):
            raise ValueError("All the labels must have category")
        if any(label not in unique_labels for label in label2category.keys()):
            raise ValueError("All the labels from label2category mapping must be in the labels")
        if any(n <= 1 for n in Counter(labels).values()):
            raise ValueError("Each class must contain at least 2 instances to fit")
        if not resample_labels:
            if any(len(list(labs)) < n_labels for labs in category2labels.values()):
                raise ValueError(f"All the categories must have at least {n_labels} unique labels")
        self._resample_labels = resample_labels
        self._labels = np.array(labels)
        self._label2category = label2category
        self._n_categories = n_categories
        self._n_labels = n_labels
        self._n_instances = n_instances

        self._batch_size = self._n_categories * self._n_labels * self._n_instances
        self._unique_labels = list(unique_labels)
        self._unique_categories = list(unique_categories)
        self._weight_categories = weight_categories

        self._label2index = {
            label: np.arange(len(self._labels))[self._labels == label].tolist() for label in self._unique_labels
        }
        self._category2labels = {
            category: {label for label, cat in self._label2category.items() if category == cat}
            for category in self._unique_categories
        }
        category_weights = {cat: len(labels) / len(unique_labels) for cat, labels in self._category2labels.items()}
        self._category_weights = (
            [category_weights[cat] for cat in self._unique_categories] if self._weight_categories else None
        )
        self._batch_number = math.ceil(len(self._unique_labels) / self._n_labels)

    @property
    def batch_size(self) -> int:
        """
        Returns:
            This value should be used in DataLoader as batch size

        """
        return self._batch_size

    @property
    def batches_in_epoch(self) -> int:
        """
        Returns:
            Number of batches in an epoch

        """
        return self._batch_number

    def __len__(self) -> int:
        """
        Returns:
            Number of batches in an epoch
        """
        return self.batches_in_epoch

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns:
            Indexes for sampling dataset elements during an epoch
        """
        epoch_indices = []
        for _ in range(self.batches_in_epoch):
            categories = np.random.choice(
                self._unique_categories,
                size=self._n_categories,
                replace=False,
                p=self._category_weights,
            )
            batch_indices = []
            for category in categories:
                labels_available = list(self._category2labels[category])
                labels = smart_sample(array=labels_available, k=self._n_labels)
                for label in labels:
                    indices = self._label2index[label]
                    samples_indices = smart_sample(array=indices, k=self._n_instances)
                    batch_indices.extend(samples_indices)
            epoch_indices.append(batch_indices)
        return iter(epoch_indices)


class SequentialCategoryBalanceSampler(CategoryBalanceBatchSampler):
    """
    Almost the same as
    >>> CategoryBalanceBatchSampler
    but indexes will be returned in a flattened way

    """

    def __iter__(self) -> Iterator[int]:  # type: ignore
        ids_flatten = []
        for ids in super().__iter__():
            ids_flatten.extend(ids)
        return iter(ids_flatten)

    def __len__(self) -> int:
        return self.batches_in_epoch * self._batch_size
