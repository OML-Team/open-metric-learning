from random import randint
from typing import List, Tuple

import pytest
import torch

from oml.interfaces.miners import TLabels
from oml.miners.inbatch_hard_cluster import HardClusterMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner


@pytest.mark.parametrize(
    ["labels", "expected"],
    [
        [
            [0, 0, 1, 2, 2, 1],
            torch.tensor(
                [
                    [True, True, False, False, False, False],
                    [False, False, True, False, False, True],
                    [False, False, False, True, True, False],
                ]
            ),
        ],
        [
            [1, 2, 3],
            torch.tensor([[True, False, False], [False, True, False], [False, False, True]]),
        ],
        [
            [1, 1, 1, 1, 2, 2, 2, 2],
            torch.tensor(
                [
                    [True, True, True, True, False, False, False, False],
                    [False, False, False, False, True, True, True, True],
                ]
            ),
        ],
    ],
)
def test_cluster_get_labels_mask(labels: List[int], expected: torch.Tensor) -> None:
    """
    Test _get_labels_mask method of HardClusterMiner.

    Args:
        labels: List of labels -- input data for method _skip_diagonal
        expected: Correct answer for labels input

    """
    miner = HardClusterMiner()
    labels_mask = miner._get_labels_mask(labels)
    assert (labels_mask == expected).all()


@pytest.mark.parametrize(
    ["features", "expected"],
    [
        [
            torch.tensor(
                [
                    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 3]],
                    [[0, 3, 0, 1], [0, 6, 0, 1], [0, 3, 0, 1]],
                ],
                dtype=torch.float,
            ),
            torch.tensor([[1, 1, 4], [1, 4, 1]]),
        ],
        [
            torch.tensor(
                [[[1, 1, 1], [1, 3, 1]], [[2, 2, 6], [2, 6, 2]], [[3, 3, 3], [3, 3, 9]]],
                dtype=torch.float,
            ),
            torch.tensor([[[1, 1], [8, 8], [9, 9]]]),
        ],
    ],
)
def test_cluster_count_intra_label_distances(features: torch.Tensor, expected: torch.Tensor) -> None:
    """
    Test _count_intra_label_distances method of HardClusterMiner.

    Args:
        features: Tensor of shape (n_labels, n_instances, embed_dim)
        embed_dim is an embedding size -- features grouped by labels
        expected: Tensor of shape (n_labels, n_instances) -- expected distances from mean
        vectors of labels to corresponding features
    """
    miner = HardClusterMiner()
    mean_vectors = features.mean(1)
    distances = miner._count_intra_label_distances(features, mean_vectors)
    assert (distances == expected).all()


@pytest.mark.parametrize(
    ["mean_vectors", "expected"],
    [
        [
            torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.float),
            torch.tensor([[0, 1, 3], [1, 0, 2], [3, 2, 0]], dtype=torch.float) ** 0.5,
        ],
        [
            torch.tensor(
                [[0, 0, 0, 0], [3, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 5]],
                dtype=torch.float,
            ),
            torch.tensor(
                [[0, 9, 16, 25], [9, 0, 25, 34], [16, 25, 0, 41], [25, 34, 41, 0]],
                dtype=torch.float,
            )
            ** 0.5,
        ],
    ],
)
def test_cluster_count_inter_label_distances(mean_vectors, expected) -> None:  # type: ignore
    """
    Test _count_inter_label_distances method of HardClusterMiner.

    Args:
        mean_vectors: Tensor of shape (n_labels, embed_dim) -- mean vectors of
        labels in the batch
        expected: Tensor of shape (n_labels, n_labels) -- expected distances from mean
        vectors of labels

    """
    miner = HardClusterMiner()
    distances = miner._count_inter_label_distances(mean_vectors)
    assert (distances == expected).all()


@pytest.mark.parametrize(
    ["embed_dim", "labels", "expected_shape"],
    [
        [128, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [(4, 128), (4, 128), (4, 128)]],
        [32, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5], [(5, 32), (5, 32), (5, 32)]],
        [16, torch.tensor([0, 0, 1, 1]), [(2, 16), (2, 16), (2, 16)]],
    ],
)
def test_cluster_sample_shapes(embed_dim: int, labels: TLabels, expected_shape: List[Tuple[int]]) -> None:
    """
    Test output shapes in sample method of HardClusterMiner.

    Args:
        embed_dim: Size of embedding
        labels: List of labels for samples in batch
        expected_shape: Expected shape of output triplet

    """
    miner = HardClusterMiner()
    batch_size = len(labels)
    features = torch.rand(size=(batch_size, embed_dim))
    anchor, positive, negative = miner.sample(features, labels)
    anchor_shape, pos_shape, neg_shape = expected_shape

    assert anchor.shape == anchor_shape
    assert positive.shape == pos_shape
    assert negative.shape == neg_shape


def test_triplet_cluster_edge_case() -> None:
    """
    Check an edge case of trivial samples for labels:
    expected HardTripletsMiner and HardClusterMiner to
    generate the same triplets.

    """
    features_dim = 128
    n_labels, n_instances = randint(2, 32), randint(2, 32)

    # Create a list of random labels
    unique_labels = torch.tensor(list(range(n_labels)))
    # Create a list of random features for all the labels
    unique_features = torch.rand(size=(n_labels, features_dim), dtype=torch.float)

    labels = unique_labels.repeat(n_instances)
    features = unique_features.repeat((n_instances, 1))

    hard_triplet_miner = HardTripletsMiner()
    hard_cluster_miner = HardClusterMiner()

    triplets = hard_triplet_miner.sample(features, labels)
    cluster_triplets = hard_cluster_miner.sample(features, labels)

    # Concatenates tensors from triplets to use torch.unique for comparison
    triplets = torch.cat(triplets, dim=1)
    cluster_triplets = torch.cat(cluster_triplets, dim=1)

    triplets = torch.unique(triplets, dim=0)
    cluster_triplets = torch.unique(cluster_triplets, dim=0)

    assert torch.allclose(triplets, cluster_triplets, atol=1e-10)
