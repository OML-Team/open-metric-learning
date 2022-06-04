from collections import Counter
from itertools import combinations, product
from random import sample
from sys import maxsize
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from oml.interfaces.miners import (
    InBatchTripletsMiner,
    ITripletsMiner,
    TLabels,
    TTriplets,
    TTripletsIds,
    labels2list,
)
from oml.utils.misc import find_value_ids


class AllTripletsMiner(InBatchTripletsMiner):
    """
    This miner selects all the possible triplets for the given labels

    """

    def __init__(self, max_output_triplets: int = maxsize):
        """
        Args:
            max_output_triplets: Number of all of the possible triplets
              in the batch can be very large,
              so we can sample only some part of them,
              determined by this parameter.

        """
        self._max_out_triplets = max_output_triplets

    def _sample(self, *_: Tensor, labels: List[int]) -> TTripletsIds:  # type: ignore
        """
        Args:
            labels: Labels of the samples in the batch
            *_: Note, that we ignore features argument

        Returns:
            Indices of the triplets

        """
        num_labels = len(labels)

        triplets = []
        for label in set(labels):
            ids_pos_cur = find_value_ids(labels, label)
            ids_neg_cur = set(range(num_labels)) - set(ids_pos_cur)

            # (l0, l1, n) and (l1, l0, n) are 2 different triplets
            # and we want both of them
            pos_pairs = list(combinations(ids_pos_cur, r=2)) + list(combinations(ids_pos_cur[::-1], r=2))

            tri = [(a, p, n) for (a, p), n in product(pos_pairs, ids_neg_cur)]
            triplets.extend(tri)

        triplets = sample(triplets, min(len(triplets), self._max_out_triplets))
        ids_anchor, ids_pos, ids_neg = zip(*triplets)

        return list(ids_anchor), list(ids_pos), list(ids_neg)


class HardTripletsMiner(InBatchTripletsMiner):
    """
    This miner selects hardest triplets based on distances between features:
    the hardest positive sample has the maximal distance to the anchor sample,
    the hardest negative sample has the minimal distance to the anchor sample.

    """

    def __init__(self, norm_required: bool = False):
        """
        Args:
            norm_required: Set True if features normalisation is needed

        """
        self._norm_required = norm_required

    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets inside the batch.

        Args:
            features: Features with the shape of [batch_size, feature_size]
            labels: Labels of the samples in the batch

        Returns:
            The batch of the triplets in the order below:
            (anchor, positive, negative)

        """
        assert features.shape[0] == len(labels)

        if self._norm_required:
            features = F.normalize(features.detach(), p=2, dim=1)

        dist_mat = torch.cdist(x1=features, x2=features, p=2)

        ids_anchor, ids_pos, ids_neg = self._sample_from_distmat(distmat=dist_mat, labels=labels)

        return ids_anchor, ids_pos, ids_neg

    @staticmethod
    def _sample_from_distmat(distmat: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets based on the given
        distances matrix. It chooses each sample in the batch as an
        anchor and then finds the hardest positive and negative pair.

        Args:
            distmat: Matrix of distances between the features
            labels: Labels of the samples

        Returns:
            The batch of triplets (with the size equals to the original bs)
            in the following order: (anchor, positive, negative)

        """
        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []

        for i_anch, label in enumerate(labels):
            ids_label = set(find_value_ids(it=labels, value=label))

            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)

            i_pos = ids_pos_cur[distmat[i_anch, ids_pos_cur].argmax()]
            i_neg = ids_neg_cur[distmat[i_anch, ids_neg_cur].argmin()]

            ids_anchor.append(i_anch)
            ids_pos.append(i_pos)
            ids_neg.append(i_neg)

        return ids_anchor, ids_pos, ids_neg


class HardClusterMiner(ITripletsMiner):
    """
    This miner selects hardest triplets based on distance to mean vectors:
    anchor is a mean vector of features of i-th label in the batch,
    the hardest positive sample is the most distant from anchor sample of
    anchor's label, the hardest negative sample is the closest mean vector
    of another labels.

    The batch must contain k samples for p labels in it (k > 1, p > 1).

    """

    def _check_input_labels(self, labels: List[int]) -> None:
        """
        Check if the labels list is valid: contains k occurrences
        for each of p labels.

        Args:
            labels: Labels in the batch

        Raises:
            ValueError: If batch is invalid (contains different samples
                for labels, contains only one label or only one sample for
                each label)

        """
        labels_counter = Counter(labels)
        k = labels_counter[labels[0]]
        if not all(n == k for n in labels_counter.values()):
            raise ValueError("Expected equal number of samples for each label")
        if len(labels_counter) <= 1:
            raise ValueError("Expected at least 2 labels in the batch")
        if k == 1:
            raise ValueError("Expected more than one sample for each label")

    @staticmethod
    def _get_labels_mask(labels: List[int]) -> Tensor:
        """
        Generate matrix of bool of shape (n_unique_labels, batch_size),
        where n_unique_labels is a number of unique labels
        in the batch; matrix[i, j] is True if j-th element of
        the batch relates to i-th label and False otherwise.

        Args:
            labels: Labels of the batch, shape (batch_size)

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
            embeddings: Tensor of shape (p, k, embed_dim) where p is a number
                of labels in the batch, k is a number of samples for each label
            mean_vectors: Tensor of shape (p, embed_dim) -- mean vectors
                of each label in the batch

        Returns:
            Tensor of shape (p, k) -- matrix of distances from mean vectors to
                related samples in the batch

        """
        p, k, embed_dim = embeddings.shape
        # Create (p, k, embed_dim) tensor of mean vectors for each label
        mean_vectors = mean_vectors.unsqueeze(1).repeat((1, k, 1))
        # Count euclidean distance between embeddings and mean vectors
        distances = torch.pow(embeddings - mean_vectors, 2).sum(2)
        return distances

    @staticmethod
    def _count_inter_label_distances(mean_vectors: Tensor) -> Tensor:
        """
        Count matrix of distances from mean vectors of labels to each other

        Args:
            mean_vectors: Tensor of shape (p, embed_dim) -- mean vectors
                of labels

        Returns:
            Tensor of shape (p, p) -- matrix of distances between mean vectors

        """
        distance = torch.cdist(x1=mean_vectors, x2=mean_vectors, p=2)
        return distance

    @staticmethod
    def _fill_diagonal(matrix: Tensor, value: float) -> Tensor:
        """
        Set diagonal elements with the value.

        Args:
            matrix: Tensor of shape (p, p)
            value: Value that diagonal should be filled with

        Returns:
            Modified matrix with inf on diagonal

        """
        p, _ = matrix.shape
        indices = torch.diag(torch.ones(p)).type(torch.bool)
        matrix[indices] = value
        return matrix

    def sample(self, features: Tensor, labels: TLabels) -> TTriplets:
        """
        This method samples the hardest triplets in the batch.

        Args:
            features: Tensor of shape (batch_size; embed_dim) that contains
                k samples for each of p labels
            labels: Labels of the batch, list or tensor of size (batch_size)

        Returns:
            p triplets of (mean_vector, positive, negative_mean_vector)

        """
        # Convert labels to list
        labels = labels2list(labels)
        self._check_input_labels(labels)

        # Get matrix of indices of labels in batch
        labels_mask = self._get_labels_mask(labels)
        p = labels_mask.shape[0]

        embed_dim = features.shape[-1]
        # Reshape embeddings to groups of (p, k, embed_dim) ones,
        # each i-th group contains embeddings of i-th label.
        features = features.repeat((p, 1, 1))
        features = features[labels_mask].view((p, -1, embed_dim))

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
