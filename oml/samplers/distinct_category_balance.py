from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Set, Union

import numpy as np
from torch.utils.data import Sampler

from oml.utils.misc import smart_sample


class DistinctCategoryBalanceBatchSampler(Sampler):
    """
    Let C is a set of categories in dataset, L is a set of labels in dataset:
    - select n_categories for the 1st batch from C
    - select n_labels for each of chosen categories for the 1st batch
    - select n_instances for each label for the 1st batch
    - define set of available for the 2nd batch labels L*: all the labels from L except the ones
    chosen for the 1st batch
    - define set of available categories C*: all the categories corresponding to labels from L*
    - select n_categories from C* for the 2nd batch
    - select n_labels for each category from L* for the 2nd batch
    - select n_instances for each label for the 2nd batch
    ...
    If all the categories were chosen sampler resets its state and goes on sampling from the first step.

    Behavior in corner cases:
    - If a class does not contain n_instances, a choice will be made with repetition.
    - If chosen category does not contain unused n_labels, all the unused labels will be added
    to a batch and missing ones will be sampled from used labels without repetition.
    - If L % n_labels == 1 then one of the classes should be dropped
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        label2category: Dict[int, int],
        n_categories: int,
        n_labels: int,
        n_instances: int,
        epoch_size: int,
    ):
        """Init DistinctCategoryBalanceBatchSampler.

        Args:
            labels: Labels to sample from
            label2category: Mapping from label to category
            n_categories: Number of categories to sample for each batch
            n_labels: Number of labels to sample for each category in batch
            n_instances: Number of samples to sample for each label in batch
            epoch_size: Number of batches in epoch
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
        if any(len(list(labs)) < n_labels for labs in category2labels.values()):
            raise ValueError(f"All the categories must have at least {n_labels} unique labels")

        self._labels = np.array(labels)
        self._label2category = label2category
        self._n_categories = n_categories
        self._n_labels = n_labels
        self._n_instances = n_instances
        self._epoch_size = epoch_size

        self._batch_size = self._n_categories * self._n_labels * self._n_instances
        self._unique_labels = unique_labels
        self._unique_categories = unique_categories

        self._label2index = {
            label: np.arange(len(self._labels))[self._labels == label].tolist() for label in self._unique_labels
        }
        self._category2labels = {
            category: {label for label, cat in self._label2category.items() if category == cat}
            for category in self._unique_categories
        }

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
        return self._epoch_size

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
        category2labels = deepcopy(self._category2labels)
        used_labels: Dict[int, Set[int]] = defaultdict(set)
        epoch_indices = []
        for _ in range(self.batches_in_epoch):
            if len(category2labels) < self._n_categories:
                category2labels = deepcopy(self._category2labels)
                used_labels = defaultdict(set)
            categories_available = list(category2labels.keys())
            categories = np.random.choice(
                categories_available, size=min(self._n_categories, len(categories_available)), replace=False
            )
            batch_indices = []
            for category in categories:
                labels_available = list(category2labels[category])
                labels_available_number = len(labels_available)
                if self._n_labels <= labels_available_number:
                    labels = np.random.choice(labels_available, size=self._n_labels, replace=False).tolist()
                else:
                    labels = (
                        labels_available
                        + np.random.choice(
                            list(used_labels[category]), size=self._n_labels - labels_available_number, replace=False
                        ).tolist()
                    )
                for label in labels:
                    indices = self._label2index[label]
                    samples_indices = smart_sample(array=indices, k=self._n_instances)
                    batch_indices.extend(samples_indices)
                category2labels[category] -= set(labels)
                used_labels[category].update(labels)
                if not category2labels[category]:
                    category2labels.pop(category)
            epoch_indices.append(batch_indices)
        return iter(epoch_indices)


class SequentialDistinctCategoryBalanceSampler(DistinctCategoryBalanceBatchSampler):
    """
    Almost the same as
    >>> DistinctCategoryBalanceBatchSampler
    but indexes will be returned in a flattened way
    """

    def __iter__(self) -> Iterator[int]:  # type: ignore
        ids_flatten = []
        for ids in super().__iter__():
            ids_flatten.extend(ids)
        return iter(ids_flatten)

    def __len__(self) -> int:
        return self.batches_in_epoch * self.batch_size
