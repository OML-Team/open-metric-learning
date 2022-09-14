from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple, Union

from torch import Tensor

TTriplets = Tuple[Tensor, Tensor, Tensor]
TTripletsIds = Tuple[List[int], List[int], List[int]]
TLabels = Union[List[int], Tensor]


def labels2list(labels: TLabels) -> List[int]:
    if isinstance(labels, Tensor):
        labels = labels.squeeze()
        labels_list = labels.tolist()
    elif isinstance(labels, list):
        labels_list = labels.copy()
    else:
        raise TypeError(f"Unexpected type of labels: {type(labels)}).")

    return labels_list


class ITripletsMiner(ABC):
    """
    An abstraction of triplet miner.

    """

    @abstractmethod
    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        """
        This method includes the logic of mining/sampling triplets.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Batch of triplets

        """
        raise NotImplementedError()


class ITripletsMinerInBatch(ITripletsMiner):
    """
    We expect that the child instances of this class
    will be used for mining triplets inside the batches.
    The batches must contain at least 2 samples for
    each class and at least 2 different labels,
    such behaviour can be guarantee via using samplers from
    our registry.

    But you are not limited to using it in any other way.

    """

    @staticmethod
    def _check_input_labels(labels: List[int]) -> None:
        """
        Args:
            labels: Labels of the samples in the batch

        """
        labels_counter = Counter(labels)
        assert all(n > 1 for n in labels_counter.values())
        assert len(labels_counter) > 1

    @abstractmethod
    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method includes the logic of mining triplets
        inside the batch. It can be based on information about
        the distance between the features, or the
        choice can be performed randomly.

        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
            Indices of the batch samples to form the triplets

        """
        raise NotImplementedError()

    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        """
        Args:
            features: Features with the shape of ``[batch_size, feature_size]``
            labels: Labels with the size of ``batch_size``

        Returns:
             Batch of triplets

        """
        # Convert labels to list
        labels = labels2list(labels)
        self._check_input_labels(labels=labels)

        ids_anchor, ids_pos, ids_neg = self._sample(features, labels=labels)

        return features[ids_anchor], features[ids_pos], features[ids_neg]


__all__ = ["TTriplets", "TTripletsIds", "TLabels", "labels2list", "ITripletsMiner", "ITripletsMinerInBatch"]
