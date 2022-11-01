from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Set, Union

import numpy as np

from oml.interfaces.samplers import IBatchSampler
from oml.utils.misc import smart_sample


class DistinctCategoryBalanceSampler(IBatchSampler):
    """
    This sampler takes ``n_instances`` for each of the ``n_labels`` for each of the
    ``n_categories`` to form the batches.
    Thus, the batch size is ``n_instances x n_labels x n_categories``.

    The strategy for the dataset with ``L`` unique labels and ``C`` unique categories is the following:

    - Select ``n_categories`` of ``C`` for the 1st batch

    - Select ``n_labels`` for each of the chosen categories for the 1st batch

    - Select ``n_instances`` for each of the chosen labels for the 1st batch

    - Define the set of available for the 2nd batch labels ``L^``: these are all the labels ``L`` except the ones
      chosen for the 1st batch

    - Define set of available categories ``C^``: these are all the categories corresponding to labels from ``L^``

    - Select ``n_categories`` from ``C^`` for the 2nd batch

    - Select ``n_labels`` for each category from ``L^`` for the 2nd batch

    - Select ``n_instances`` for each label for the 2nd batch

    - ...

    - Epoch ends after ``epoch_size`` steps

    Behavior in corner cases:

    - If all the categories were chosen before ``epoch_size`` steps, the sampler resets its state and goes on sampling
     from the first step.

    - If some class does not contain ``n_instances``, a choice will be made with repetition.

    - If the chosen category does not contain unused ``n_labels``, all the unused labels will be added to a batch and
     the missing ones will be sampled from the used labels without repetition.

    - If ``L % n_labels == 1`` then one of the labels must be dropped because we always want to have more than 1 label
     in a batch to be able to form positive pairs later on.

    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        label2category: Dict[int, Union[str, int]],
        n_categories: int,
        n_labels: int,
        n_instances: int,
        epoch_size: int,
    ):
        """

        Args:
            labels: Labels to sample from
            label2category: Mapping from label to category
            n_categories: The desired number of categories to sample for each batch
            n_labels: The desired number of labels to sample for each category in batch
            n_instances: The desired number of samples to sample for each label in batch
            epoch_size: The desired number of batches in epoch

        """
        unique_labels = set(labels)
        unique_categories = set(label2category.values())
        category2labels = {
            category: {label for label, cat in label2category.items() if category == cat}
            for category in sorted(list(unique_categories))
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
        self.n_categories = n_categories
        self.n_labels = n_labels
        self.n_instances = n_instances
        self._epoch_size = epoch_size

        self._batch_size = self.n_categories * self.n_labels * self.n_instances

        self._label2index = {
            label: np.arange(len(self._labels))[self._labels == label].tolist() for label in sorted(list(unique_labels))
        }
        self._category2labels = category2labels

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._epoch_size

    def __iter__(self) -> Iterator[List[int]]:
        category2labels = deepcopy(self._category2labels)
        used_labels: Dict[int, Set[int]] = defaultdict(set)
        epoch_indices = []
        for _ in range(self._epoch_size):
            if len(category2labels) < self.n_categories:
                category2labels = deepcopy(self._category2labels)
                used_labels = defaultdict(set)
            categories_available = list(category2labels.keys())
            categories = np.random.choice(
                categories_available, size=min(self.n_categories, len(categories_available)), replace=False
            )
            batch_indices = []
            for category in categories:
                labels_available = list(category2labels[category])
                labels_available_number = len(labels_available)
                if self.n_labels <= labels_available_number:
                    labels = np.random.choice(labels_available, size=self.n_labels, replace=False).tolist()
                else:
                    labels = (
                        labels_available
                        + np.random.choice(
                            list(used_labels[category]), size=self.n_labels - labels_available_number, replace=False
                        ).tolist()
                    )
                for label in labels:
                    indices = self._label2index[label]
                    samples_indices = smart_sample(array=indices, k=self.n_instances)
                    batch_indices.extend(samples_indices)
                category2labels[category] -= set(labels)
                used_labels[category].update(labels)
                if not category2labels[category]:
                    category2labels.pop(category)
            epoch_indices.append(batch_indices)
        return iter(epoch_indices)


__all__ = ["DistinctCategoryBalanceSampler"]
