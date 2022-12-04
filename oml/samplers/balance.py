from collections import Counter
from typing import Iterator, List, Union

import numpy as np

from oml.interfaces.samplers import IBatchSampler
from oml.utils.misc import smart_sample


class BalanceSampler(IBatchSampler):
    """
    This sampler takes ``n_instances`` for each of the ``n_labels`` to form the batches.
    Thus, the batch size is ``n_instances x n_labels``. This type of sampling can be found
    in the classical Person Re-Id paper -
    `In Defense of the Triplet Loss for Person Re-Identification`_.

    .. _In Defense of the Triplet Loss for Person Re-Identification:
        https://arxiv.org/abs/1703.07737

    The strategy for the dataset with ``L`` unique labels is the following:

    - Select ``n_labels`` of ``L`` labels for the 1st batch

    - Select ``n_instances`` for each label for the 1st batch

    - Select ``n_labels`` of ``L - n_labels`` remaining labels for 2nd batch

    - Select ``n_instances`` instances for each label for the 2nd batch

    - ...

    - The epoch ends after ``L // n_labels``.

    Thus, in each epoch, all the labels will be selected once, but this
    does not mean that all the instances will be picked.

    Behavior in corner cases:

    - If some label does not contain ``n_instances``, a choice will be made with repetition.

    - If ``L % n_labels != 0`` then we drop the last batch.

    """

    def __init__(self, labels: Union[List[int], np.ndarray], n_labels: int, n_instances: int):
        """
        Args:
            labels: List of the labels for each element in the dataset
            n_labels: The desired number of labels in a batch, should be > 1
            n_instances: The desired number of instances of each label in a batch, should be > 1

        """
        unq_labels = set(labels)

        assert isinstance(n_labels, int) and isinstance(n_instances, int)
        assert (1 < n_labels <= len(unq_labels)) and (1 < n_instances)
        assert all(n > 1 for n in Counter(labels).values()), "Each label should contain at least 2 samples"

        self._labels = np.array(labels)
        self.n_labels = n_labels
        self.n_instances = n_instances

        self._batch_size = self.n_labels * self.n_instances
        self._unq_labels = unq_labels

        labels = np.array(labels)
        self.lbl2idx = {label: np.arange(len(labels))[labels == label].tolist() for label in set(labels)}

        self._batches_in_epoch = len(self._unq_labels) // self.n_labels

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return self._batches_in_epoch

    def __iter__(self) -> Iterator[List[int]]:
        inds_epoch = []

        labels_rest = self._unq_labels.copy()

        for _ in range(len(self)):
            ids_batch = []

            labels_for_batch = set(
                np.random.choice(list(labels_rest), size=min(self.n_labels, len(labels_rest)), replace=False)
            )
            labels_rest -= labels_for_batch

            for cls in labels_for_batch:
                cls_ids = self.lbl2idx[cls]
                selected_inds = smart_sample(cls_ids, self.n_instances)
                ids_batch.extend(selected_inds)

            inds_epoch.append(ids_batch)

        return iter(inds_epoch)  # type: ignore


__all__ = ["BalanceSampler"]
