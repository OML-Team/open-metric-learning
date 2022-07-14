from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Set, Union

import numpy as np
from torch.utils.data import Sampler

from oml.utils.misc import smart_sample


class DistinctCategoryBalanceBatchSampler(Sampler):
    """Let C is a set of categories in dataset, P is a set of labels in dataset:
    - select c categories for the 1st batch from C
    - select n_labels labels for each of chosen categories for the 1st batch (total labels available P)
    - select n_instances samples for each label for the 1st batch
    - define set of available for the 2nd batch labels P*: all the labels from P except the ones
    chosen for the 1st batch
    - define set of available categories C*: all the categories corresponding to labels from P*
    - select c categories from C* for the 2nd batch
    - select n_labels labels for each category from P* for the 2nd batch
    - select n_instances samples for each label for the 2nd batch
    ...
    If all the categories were chosen sampler resets its state and goes on sampling from the first step.

    Behavior in corner cases:
    - If a class does not contain n_instances instances, a choice will be made with repetition.
    - If chosen category does not contain n_labels unused labels, all the unused labels will be added
    to a batch and missing ones will be sampled from used labels without repetition.
    - If P % n_labels == 1 then one of the classes should be dropped
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        label2category: Dict[int, int],
        c: int,
        p: int,
        k: int,
        epoch_size: int,
    ):
        """Init DistinctCategoryBalanceBatchSampler.

        Args:
            labels: Labels to sample from
            label2category: Mapping from label to category
            c: Number of categories to sample for each batch
            p: Number of labels to sample for each category in batch
            k: Number of samples to sample for each label in batch
            epoch_size: Number of batches in epoch
        """
        super().__init__(self)
        unique_labels = set(labels)
        unique_categories = set(label2category.values())
        category2labels = {
            category: {label for label, cat in label2category.items() if category == cat}
            for category in unique_categories
        }

        for param in [c, p, k]:
            if not isinstance(param, int):
                raise TypeError(f"{param.__name__} must be int, {type(param)} given")
        if not 1 <= c <= len(unique_categories):
            raise ValueError(f"c must be 1 <= c <= {len(unique_categories)}, {c} given")
        if not 1 < p <= len(unique_labels):
            raise ValueError(f"n_labels must be 1 < n_labels <= {len(unique_labels)}, {p} given")
        if k <= 1:
            raise ValueError(f"n_instances must be not less than 1, {k} given")
        if any(label not in label2category.keys() for label in unique_labels):
            raise ValueError("All the labels must have category")
        if any(label not in unique_labels for label in label2category.keys()):
            raise ValueError("All the labels from label2category mapping must be in the labels")
        if any(n <= 1 for n in Counter(labels).values()):
            raise ValueError("Each class must contain at least 2 instances to fit")
        if any(len(list(labs)) < p for labs in category2labels.values()):
            raise ValueError(f"All the categories must have at least {p} unique labels")

        self._labels = np.array(labels)
        self._label2category = label2category
        self._c = c
        self._p = p
        self._k = k
        self._epoch_size = epoch_size

        self._batch_size = self._c * self._p * self._k
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
            if len(category2labels) < self._c:
                category2labels = deepcopy(self._category2labels)
                used_labels = defaultdict(set)
            categories_available = list(category2labels.keys())
            categories = np.random.choice(
                categories_available, size=min(self._c, len(categories_available)), replace=False
            )
            batch_indices = []
            for category in categories:
                labels_available = list(category2labels[category])
                labels_available_number = len(labels_available)
                if self._p <= labels_available_number:
                    labels = np.random.choice(labels_available, size=self._p, replace=False).tolist()
                else:
                    labels = (
                        labels_available
                        + np.random.choice(
                            list(used_labels[category]), size=self._p - labels_available_number, replace=False
                        ).tolist()
                    )
                for label in labels:
                    indices = self._label2index[label]
                    samples_indices = smart_sample(array=indices, k=self._k)
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
