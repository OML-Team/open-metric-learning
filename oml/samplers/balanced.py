import math
import random
from collections import Counter
from typing import Dict, Iterator, List, Union

import numpy as np
from torch.utils.data.sampler import Sampler


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

            labels_for_batch = set(random.sample(labels_rest, min(self._p, len(labels_rest))))
            labels_rest -= labels_for_batch

            for cls in labels_for_batch:
                cls_ids = self.lbl2idx[cls]

                # we've checked in __init__ that this value must be > 1
                n_samples = len(cls_ids)

                if n_samples < self._k:
                    selected_inds = random.sample(cls_ids, k=n_samples) + random.choices(cls_ids, k=self._k - n_samples)
                else:
                    selected_inds = random.sample(cls_ids, k=self._k)

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
        few_labels_number_policy: str = "raise",
    ):
        """Sampler initialisation."""
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
        if few_labels_number_policy not in ["raise", "resample"]:
            raise ValueError(f"Only 'raise' and 'resample' policies are allowed, {few_labels_number_policy} given.")
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
        if few_labels_number_policy == "raise":
            if any(len(list(labs)) < p for labs in category2labels.values()):
                raise ValueError(f"All the categories must have at least {p} unique labels")
        self._few_labels_number_policy = few_labels_number_policy
        self._labels = np.array(labels)
        self._label2category = label2category
        self._c = c
        self._p = p
        self._k = k

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
            categories = random.sample(
                self._unique_categories,
                k=self._c,
            )
            batch_indices = []
            for category in categories:
                labels_available = list(self._category2labels[category])
                labels_available_number = len(labels_available)
                if self._p <= labels_available_number:
                    labels = random.sample(
                        labels_available,
                        k=self._p,
                    )
                else:
                    labels = labels_available + random.choices(labels_available, k=self._p - labels_available_number)
                for label in labels:
                    indices = self._label2index[label]
                    samples_number = len(indices)

                    if samples_number < self._k:
                        samples_indices = indices + random.choices(
                            indices,
                            k=self._k - samples_number,
                        )
                    else:
                        samples_indices = random.sample(
                            indices,
                            k=self._k,
                        )
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
