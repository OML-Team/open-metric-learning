from random import randint, shuffle
from typing import List, Tuple

import pytest
import torch
from torch import Tensor

TLabelsLI = List[Tuple[List[int], int, int]]


def generate_valid_labels(num: int) -> TLabelsLI:
    """
    This function generates some valid inputs for miners.
    It generates n_instances for n_labels.

    Args:
        num: Number of generated samples

    Returns:
        Samples in the following order: (labels, n_labels, n_instances)

    """
    labels_li = []

    for _ in range(num):
        n_labels, n_instances = randint(2, 12), randint(2, 12)
        labels_list = [[label] * randint(2, 12) for label in range(n_labels)]
        labels = [el for sublist in labels_list for el in sublist]

        shuffle(labels)
        labels_li.append((labels, n_labels, n_instances))

    return labels_li


@pytest.fixture()
def features_and_labels() -> List[Tuple[Tensor, List[int]]]:
    """
    Returns: list of features and valid labels

    """
    num_batches = 100
    features_dim = 10

    labels_li = generate_valid_labels(num=num_batches)
    labels_list, _, _ = zip(*labels_li)

    features = []
    for labels in labels_list:
        features.append(torch.rand(size=(len(labels), features_dim)))

    return list(zip(features, labels_list))
