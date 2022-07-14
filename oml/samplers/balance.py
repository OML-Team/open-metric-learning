from collections import Counter
from typing import Any, Dict, Iterator, List, Union

import numpy as np
from torch.utils.data.sampler import Sampler

from oml.utils.misc import smart_sample


class BalanceBatchSampler(Sampler):
    """
    This kind of sampler can be used for both metric learning and
    classification task.

    Sampler with the given strategy for the dataset with L unique labels:
    - Selection n_labels of L labels for the 1st batch
    - Selection n_instances instances for each label for the 1st batch
    - Selection n_labels of L - n_labels remaining labels for 2nd batch
    - Selection n_instances instances for each label for the 2nd batch
    - ...

    The epoch ends when there are no labels left.
    So, the batch size is n_instances * n_labels except the last one.
    Thus, in each epoch, all the labels will be selected once, but this
    does not mean that all the instances will be selected during the epoch.
    One of the purposes of this sampler is to be used for
    forming triplets and pos/neg pairs inside the batch.
    To guarante existing of these pairs in the batch,
    n_labels and n_instances should be > 1. (1)

    Behavior in corner cases:
    - If a label does not contain n_instances instances,
    a choice will be made with repetition.
    - If total_number_of_labels % n_labels == 1 then one of the labels should be dropped
    otherwise statement (1) will not be met.

    This type of sampling can be found in the classical paper of Person Re-Id,
    where n_labels equals 32 and n_instances equals 4:
    `In Defense of the Triplet Loss for Person Re-Identification`_.

    Args:
        labels: list of labels labels for each elem in the dataset
        n_labels: number of labels in a batch, should be > 1
        n_instances: number of instances of each label in a batch, should be > 1

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737

    """

    def __init__(self, labels: Union[List[int], np.ndarray], n_labels: int, n_instances: int, **kwargs: Dict[str, Any]):
        """Sampler initialisation."""
        super().__init__(self)
        unq_labels = set(labels)

        assert isinstance(n_labels, int) and isinstance(n_instances, int)
        assert (1 < n_labels <= len(unq_labels)) and (1 < n_instances)
        assert all(n > 1 for n in Counter(labels).values()), "Each label should contain at least 2 samples to fit (1)"

        self._labels = np.array(labels)
        self._n_labels = n_labels
        self._n_instances = n_instances

        self._batch_size = self._n_labels * self._n_instances
        self._unq_labels = unq_labels

        # to satisfy statement (1)
        n_labels = len(self._unq_labels)
        if n_labels % self._n_labels == 1:
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
        return int(np.ceil(self._labels_per_epoch / self._n_labels))

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
                np.random.choice(list(labels_rest), size=min(self._n_labels, len(labels_rest)), replace=False)
            )
            labels_rest -= labels_for_batch

            for cls in labels_for_batch:
                cls_ids = self.lbl2idx[cls]
                selected_inds = smart_sample(cls_ids, self._n_instances)
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
        return self._labels_per_epoch * self._n_instances
