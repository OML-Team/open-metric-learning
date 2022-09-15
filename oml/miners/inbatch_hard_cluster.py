from collections import Counter
from typing import List

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMiner, TLabels, TTriplets, labels2list
from oml.utils.misc import find_value_ids
from oml.utils.misc_torch import pairwise_dist


class HardClusterMiner(ITripletsMiner):
    """
    This miner selects the hardest triplets based on the distance to mean vectors:
    anchor is a mean vector of features of i-th label in the batch, the hardest positive sample
    is the most distant from the anchor sample of anchor's label, the hardest negative
    sample is the closest mean vector of other labels.

    The batch must contain ``n_instances`` for ``n_labels`` where both values higher than 1.

    """

    def _check_input_labels(self, labels: List[int]) -> None:
        """
        Check if the labels list is valid: contains ``n_instances`` for each of ``n_labels``.

        Args:
            labels: Labels with the size of ``batch_size``

        Raises:
            ValueError: If the batch is invalid (contains different samples
                for labels, contains only one label, or only one sample for each label)

        """
        labels_counter = Counter(labels)
        n_instances = labels_counter[labels[0]]
        if not all(n == n_instances for n in labels_counter.values()):
            raise ValueError("Expected equal number of samples for each label")
        if len(labels_counter) <= 1:
            raise ValueError("Expected at least 2 labels in the batch")
        if n_instances == 1:
            raise ValueError("Expected more than one sample for each label")

    @staticmethod
    def _get_labels_mask(labels: List[int]) -> Tensor:
        """
        Generate matrix with the shape of ``[n_unique_labels, batch_size]``,
        where ``n_unique_labels`` is a number of unique labels
        in the batch; ``matrix[i, j]`` is ``True`` if j-th element of
        the batch relates to i-th label and ``False`` otherwise.

        Args:
            labels: Labels with the size of ``batch_size``

        Returns:
            Matrix of indices of labels in batch

        """
        unique_labels = sorted(np.unique(labels))
        labels_number = len(unique_labels)
        labels_mask = torch.zeros(size=(labels_number, len(labels)))
        for label_idx, label in enumerate(unique_labels):
            label_indices = find_value_ids(labels, label)
            labels_mask[label_idx][label_indices] = 1
        return labels_mask.type(torch.bool)

    @staticmethod
    def _count_intra_label_distances(embeddings: Tensor, mean_vectors: Tensor) -> Tensor:
        """
        Count matrix of distances from mean vector of each label to it's
        samples embeddings.

        Args:
            embeddings: Tensor with the shape of ``[n_labels, n_instances, embed_dim]``
            mean_vectors: Tensor with the shape of ``[n_labels, embed_dim]`` -- mean vectors
                of each label in the batch

        Returns:
            Tensor with the shape of ``[n_labels, n_instances]`` -- matrix of distances from mean vectors to
                related samples in the batch

        """
        n_labels, n_instances, embed_dim = embeddings.shape
        # Create (n_labels, n_instances, embed_dim) tensor of mean vectors for each label
        mean_vectors = mean_vectors.unsqueeze(1).repeat((1, n_instances, 1))
        # Count euclidean distance between embeddings and mean vectors
        distances = torch.pow(embeddings - mean_vectors, 2).sum(2)
        return distances

    @staticmethod
    def _count_inter_label_distances(mean_vectors: Tensor) -> Tensor:
        """
        Count matrix of distances from mean vectors of labels to each other

        Args:
            mean_vectors: Tensor with the shape of ``[n_labels, embed_dim]`` -- mean vectors
                of the labels

        Returns:
            Tensor with the shape of ``[n_labels, n_labels]`` -- matrix of distances between mean vectors

        """
        distance = pairwise_dist(x1=mean_vectors, x2=mean_vectors, p=2)
        return distance

    @staticmethod
    def _fill_diagonal(matrix: Tensor, value: float) -> Tensor:
        """
        Set diagonal elements with the value.

        Args:
            matrix: Tensor with the shape of ``[n_labels, n_labels]``
            value: Value that diagonal should be filled with

        Returns:
            Modified matrix with inf on diagonal

        """
        n_labels, _ = matrix.shape
        indices = torch.diag(torch.ones(n_labels)).type(torch.bool)
        matrix[indices] = value
        return matrix

    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        """
        This method samples the hardest triplets in the batch.

        Args:
            features: Tensor with the shape of ``[batch_size, embed_dim]`` that contains
                ``n_instances`` for each of ``n_labels``
            labels: Labels with the size of ``batch_size``

        Returns:
            ``n_labels`` triplets in the form of ``(mean_vector, positive, negative_mean_vector)``

        """
        # Convert labels to list
        labels = labels2list(labels)
        self._check_input_labels(labels)

        # Get matrix of indices of labels in batch
        labels_mask = self._get_labels_mask(labels)
        n_labels = labels_mask.shape[0]

        embed_dim = features.shape[-1]
        # Reshape embeddings to groups of (n_labels, n_instances, embed_dim) ones,
        # each i-th group contains embeddings of i-th label.
        features = features.repeat((n_labels, 1, 1))
        features = features[labels_mask].view((n_labels, -1, embed_dim))

        # Count mean vectors for each label in batch
        mean_vectors = features.mean(1)

        d_intra = self._count_intra_label_distances(features, mean_vectors)
        # Count the distances to the sample farthest from mean vector
        # for each label.
        pos_indices = d_intra.max(1).indices
        # Count matrix of distances from mean vectors to each other
        d_inter = self._count_inter_label_distances(mean_vectors)
        # For each label mean vector get the closest mean vector
        d_inter = self._fill_diagonal(d_inter, float("inf"))
        neg_indices = d_inter.min(1).indices
        positives = torch.stack([features[idx][pos_idx] for idx, pos_idx in enumerate(pos_indices)])

        return mean_vectors, positives, mean_vectors[neg_indices]


__all__ = ["HardClusterMiner"]
