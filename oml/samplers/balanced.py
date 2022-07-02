import math
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, Iterator, List, Set, Union

import numpy as np
from torch.utils.data.sampler import Sampler

from oml.utils.misc import smart_sample


class BalanceBatchSampler(Sampler):
    """
    This kind of sampler can be used for both metric learning and
    classification task.

    Sampler with the given strategy for the dataset with L unique labels:
    - Selection P of L labels for the 1st batch
    - Selection K instances for each label for the 1st batch
    - Selection P of L - P remaining labels for 2nd batch
    - Selection K instances for each label for the 2nd batch
    - ...

    The epoch ends when there are no labels left.
    So, the batch size is P * K except the last one.
    Thus, in each epoch, all the labels will be selected once, but this
    does not mean that all the instances will be selected during the epoch.
    One of the purposes of this sampler is to be used for
    forming triplets and pos/neg pairs inside the batch.
    To guarante existing of these pairs in the batch,
    P and K should be > 1. (1)

    Behavior in corner cases:
    - If a label does not contain K instances,
    a choice will be made with repetition.
    - If L % P == 1 then one of the labels should be dropped
    otherwise statement (1) will not be met.

    This type of sampling can be found in the classical paper of Person Re-Id,
    where P equals 32 and K equals 4:
    `In Defense of the Triplet Loss for Person Re-Identification`_.

    Args:
        labels: list of labels labels for each elem in the dataset
        p: number of labels in a batch, should be > 1
        k: number of instances of each label in a batch, should be > 1

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737

    """

    def __init__(self, labels: Union[List[int], np.ndarray], p: int, k: int):
        """Sampler initialisation."""
        super().__init__(self)
        unq_labels = set(labels)

        assert isinstance(p, int) and isinstance(k, int)
        assert (1 < p <= len(unq_labels)) and (1 < k)
        assert all(n > 1 for n in Counter(labels).values()), "Each label should contain at least 2 samples to fit (1)"

        self._labels = np.array(labels)
        self._p = p
        self._k = k

        self._batch_size = self._p * self._k
        self._unq_labels = unq_labels

        # to satisfy statement (1)
        n_labels = len(self._unq_labels)
        if n_labels % self._p == 1:
            self._labels_per_epoch = n_labels - 1
        else:
            self._labels_per_epoch = n_labels

        labels = np.array(labels)
        self.lbl2idx = {label: np.arange(len(labels))[labels == label].tolist() for label in set(labels)}

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
        return int(np.ceil(self._labels_per_epoch / self._p))

    def __len__(self) -> int:
        return self.batches_in_epoch

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns:
            Indexes for sampling dataset elements during an epoch
        """
        inds_epoch = []

        labels_rest = self._unq_labels.copy()

        for _ in range(self.batches_in_epoch):
            ids_batch = []

            labels_for_batch = set(
                np.random.choice(list(labels_rest), size=min(self._p, len(labels_rest)), replace=False)
            )
            labels_rest -= labels_for_batch

            for cls in labels_for_batch:
                cls_ids = self.lbl2idx[cls]
                selected_inds = smart_sample(array=cls_ids, n_samples=self._k)
                ids_batch.extend(selected_inds)

            inds_epoch.append(ids_batch)

        return iter(inds_epoch)  # type: ignore


class SequentialBalanceSampler(BalanceBatchSampler):
    """
    Almost the same as
    >>> BalanceBatchSampler
    but indexes will be returned in a flattened way

    """

    def __iter__(self) -> Iterator[int]:  # type: ignore
        ids_flatten = []
        for ids in super().__iter__():
            ids_flatten.extend(ids)
        return iter(ids_flatten)

    def __len__(self) -> int:
        return self._labels_per_epoch * self._k


class CategoryBalanceBatchSampler(Sampler):
    """Let C is a set of categories in dataset, P is a set of labels in dataset:
    - select c categories for the 1st batch from C
    - select p labels for each of chosen categories for the 1st batch
    - select k samples for each label for the 1st batch
    - select c categories from C for the 2nd batch
    - select p labels for each category for the 2nd batch
    - select k samples for each label for the 2nd batch
    ...

    Behavior in corner cases:
    - If a label does not contain k instances, a choice will be made with repetition.
    - If P % p == 1 then one of the labels should be dropped
    - If a category does not contain p labels, a choice will be made with repetition or an error will be raised
    depend on few_labels_number_policy
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        label2category: Dict[int, int],
        c: int,
        p: int,
        k: int,
        resample_labels: bool = False,
        weight_categories: bool = True,
    ):
        """Init CategoryBalanceBatchSampler.

        Args:
            labels: Labels to sample from
            label2category: Mapping from label to category
            c: Number of categories to sample for each batch
            p: Number of labels to sample for each category in batch
            k: Number of samples to sample for each label in batch
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
        for param in [c, p, k]:
            if not isinstance(param, int):
                raise TypeError(f"{param.__name__} must be int, {type(param)} given")
        if not 1 <= c <= len(unique_categories):
            raise ValueError(f"c must be 1 <= c <= {len(unique_categories)}, {c} given")
        if not 1 < p <= len(unique_labels):
            raise ValueError(f"p must be 1 < p <= {len(unique_labels)}, {p} given")
        if k <= 1:
            raise ValueError(f"k must be not less than 1, {k} given")
        if any(label not in label2category.keys() for label in unique_labels):
            raise ValueError("All the labels must have category")
        if any(label not in unique_labels for label in label2category.keys()):
            raise ValueError("All the labels from label2category mapping must be in the labels")
        if any(n <= 1 for n in Counter(labels).values()):
            raise ValueError("Each class must contain at least 2 instances to fit")
        if not resample_labels:
            if any(len(list(labs)) < p for labs in category2labels.values()):
                raise ValueError(f"All the categories must have at least {p} unique labels")
        self._resample_labels = resample_labels
        self._labels = np.array(labels)
        self._label2category = label2category
        self._c = c
        self._p = p
        self._k = k

        self._batch_size = self._c * self._p * self._k
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
        self._batch_number = math.ceil(len(self._unique_labels) / self._p)

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
                size=self._c,
                replace=False,
                p=self._category_weights,
            )
            batch_indices = []
            for category in categories:
                labels_available = list(self._category2labels[category])
                labels = smart_sample(array=labels_available, n_samples=self._p)
                for label in labels:
                    indices = self._label2index[label]
                    samples_indices = smart_sample(array=indices, n_samples=self._k)
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


class DistinctCategoryBalanceBatchSampler(Sampler):
    """Let C is a set of categories in dataset, P is a set of labels in dataset:
    - select c categories for the 1st batch from C
    - select p labels for each of chosen categories for the 1st batch (total labels available P)
    - select k samples for each label for the 1st batch
    - define set of available for the 2nd batch labels P*: all the labels from P except the ones
    chosen for the 1st batch
    - define set of available categories C*: all the categories corresponding to labels from P*
    - select c categories from C* for the 2nd batch
    - select p labels for each category from P* for the 2nd batch
    - select k samples for each label for the 2nd batch
    ...
    If all the categories were chosen sampler resets its state and goes on sampling from the first step.

    Behavior in corner cases:
    - If a class does not contain k instances, a choice will be made with repetition.
    - If chosen category does not contain p unused labels, all the unused labels will be added
    to a batch and missing ones will be sampled from used labels without repetition.
    - If P % p == 1 then one of the classes should be dropped
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
            raise ValueError(f"p must be 1 < p <= {len(unique_labels)}, {p} given")
        if k <= 1:
            raise ValueError(f"k must be not less than 1, {k} given")
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
                    samples_indices = smart_sample(array=indices, n_samples=self._k)
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
