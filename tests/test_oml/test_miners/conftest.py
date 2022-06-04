from random import randint, shuffle
from typing import List, Tuple

import pytest
import torch
from torch import Tensor

TLabelsPK = List[Tuple[List[int], int, int]]


def generate_valid_labels(num: int) -> TLabelsPK:
    """
    This function generates some valid inputs for miners.
    It generates k instances for p labels.

    Args:
        num: Number of generated samples

    Returns:
        Samples in the following order: (labels, p, k)

    """
    labels_pk = []

    for _ in range(num):
        p, k = randint(2, 12), randint(2, 12)
        labels_list = [[label] * randint(2, 12) for label in range(p)]
        labels = [el for sublist in labels_list for el in sublist]

        shuffle(labels)
        labels_pk.append((labels, p, k))

    return labels_pk


@pytest.fixture()
def features_and_labels() -> List[Tuple[Tensor, List[int]]]:
    """
    Returns: list of features and valid labels

    """
    num_batches = 100
    features_dim = 10

    labels_pk = generate_valid_labels(num=num_batches)
    labels_list, _, _ = zip(*labels_pk)

    features = []
    for labels in labels_list:
        features.append(torch.rand(size=(len(labels), features_dim)))

    return list(zip(features, labels_list))
