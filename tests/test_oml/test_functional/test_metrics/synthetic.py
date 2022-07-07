import random
from itertools import chain
from typing import List, Tuple, Union

import torch

from oml.functional.metrics import calc_mask_to_ignore


def generate_retrieval_case(
    max_labels: int,
    max_samples_per_label: int,
    is_query_all: bool = False,
    is_gallery_all: bool = False,
    return_desired_correct_positions: bool = False,
) -> Tuple[Union[List[List[int]], torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function generates set of parameters `is_label`, `is_query`, `is_gallery` as well as `distance_matrix` or
    `desired_correct_positions` of GT

    Args:
        max_labels: maximum number of generated labels (minimum is 2)
        max_samples_per_label: maximum number of generated samples per label (minimum is 2). Each unique label
            can have different number of samples
        is_query_all: Set each generated label as query
        is_gallery_all: Set each generated label as gallery
        return_desired_correct_positions: If False return distance matrix instead of GT positions matrix

    Example:
        max_labels = 4  # 2 - 4 unqiue labels
        max_samples_per_label = 4  # 2 - 4 samples per label

        =>

        labels =
        [0, 0,
        1, 1, 1, 1,
        2, 2, 2]
        is_query =
        [True, False,
        False, True, True, False,
        True, False, True]
        is_gallery =
        [True, True,
        True, False, True, False,
        False, True, True]
    """

    assert max_labels >= 2
    assert max_samples_per_label >= 2

    range_num_labels = (2, max_labels)
    range_num_samples_per_label = (2, max_samples_per_label)

    labels = []
    is_query = []
    is_gallery = []

    num_samples = [random.randint(*range_num_samples_per_label) for _ in range(random.randint(*range_num_labels))]

    for label, num_samples_for_label in enumerate(num_samples):
        labels.extend([label] * num_samples_for_label)

        if is_query_all:
            is_query_for_label = [True] * num_samples_for_label
        else:
            is_query_for_label = [bool(random.randint(0, 1)) for _ in range(num_samples_for_label)]
        guaranted_query = random.randrange(num_samples_for_label)
        is_query_for_label[guaranted_query] = True

        if is_gallery_all:
            is_gallery_for_label = [True] * num_samples_for_label
        else:
            is_gallery_for_label = [bool(random.randint(0, 1)) for _ in range(num_samples_for_label)]

            # we guarantee that for each query there is at least one gallery which is not equal to query
            if sum(is_query_for_label) == 1:
                # set another available idx as gallery
                guaranted_galleries = random.sample(set(range(num_samples_for_label)) - {guaranted_query}, k=1)
            else:
                # We have at least 2 queries in this section
                # To calculate metric for each query it's necessary to have minimum 1 gallery with another idx.
                # With k=2 one gallery can be from same or another idx, but second is guaranteed with another idx
                # E.g. is_query = [True, True, True] and is_gallery = [True, False, True]
                guaranted_galleries = random.sample(range(num_samples_for_label), k=2)

            for guaranted_g in guaranted_galleries:
                is_gallery_for_label[guaranted_g] = True

        is_query.append(is_query_for_label)
        is_gallery.append(is_gallery_for_label)

    num_galleries_total = sum(chain(*is_gallery))

    def _gen_correct_positions(_is_gallery_for_label: List[bool], q_is_g: bool) -> List[int]:
        return sorted(random.sample(range(num_galleries_total - q_is_g), k=sum(_is_gallery_for_label) - q_is_g))

    desired_correct_positions = []
    for is_query_for_label, is_gallery_for_label in zip(is_query, is_gallery):
        for q, g in zip(is_query_for_label, is_gallery_for_label):
            if q:
                desired_correct_positions.append(_gen_correct_positions(is_gallery_for_label, q == g))

    labels = torch.tensor(labels)
    is_query = torch.tensor(list(chain(*is_query)))
    is_gallery = torch.tensor(list(chain(*is_gallery)))

    if return_desired_correct_positions:
        return desired_correct_positions, labels, is_query, is_gallery
    else:
        distances = generate_distance_matrix(
            desired_correct_positions, labels=labels, is_query=is_query, is_gallery=is_gallery
        )
        return distances, labels, is_query, is_gallery


def generate_distance_matrix(
    correct_positions_array: List[List[int]], labels: torch.Tensor, is_query: torch.Tensor, is_gallery: torch.Tensor
) -> torch.Tensor:
    """
    Function generates distance matrix based on desired correct positions.
    For cases query == gallery coresponded distances are equal to zero

    Example:

        >>> desired_correct_positions = [[0, 1, 4], [0, 2, 4]]
        >>> labels = torch.tensor([0, 0, 0, 0, 1, 1])
        >>> is_query = torch.tensor([False, True, True, False, False, False])
        >>> is_gallery = torch.tensor([True] * len(labels))

        gives a distance matrix which, after sorting and applying `mask_to_ignore`, has the correct answers at
        the corresponding positions

        VVXXV
        VXVXV

        where V - correct label, X - incorrect label
    """
    num_query = is_query.sum()
    num_gallery = is_gallery.sum()

    assert num_query == len(correct_positions_array)

    mask_to_ignore = calc_mask_to_ignore(is_query=is_query, is_gallery=is_gallery)
    q_in_g_array = torch.any(mask_to_ignore, dim=1).tolist()
    query_ids = torch.nonzero(is_query).squeeze()

    assert all(len(pos) < num_gallery for pos in correct_positions_array)

    for idx, (pos, q_in_g) in enumerate(zip(correct_positions_array, q_in_g_array)):
        if not all(idx < num_gallery - int(q_in_g) for idx in pos):
            raise ValueError(
                f"Desired ids are greater than available number of galleries for "
                f"label_idx={query_ids[idx].item()}, desired_positions={pos}"
            )
        if not (len(pos) < (num_gallery - int(q_in_g))):
            raise ValueError(
                f"Desired number of ids are greater than available number of galleries for "
                f"label_idx={query_ids[idx].item()}, desired_positions={pos}"
            )

    gallery_ids = torch.nonzero(is_gallery).squeeze()
    gallery_labels = labels[is_gallery]

    distances = -1 * torch.ones(num_query, num_gallery, dtype=torch.float)

    for row, (query_idx, correct_positions, q_in_g) in enumerate(zip(query_ids, correct_positions_array, q_in_g_array)):
        same_labels = gallery_labels == labels[query_idx]
        idx_q_is_g_in_same_labels = torch.nonzero(gallery_ids == query_idx).squeeze()

        if q_in_g:
            same_labels[idx_q_is_g_in_same_labels] = False

        distances[row, same_labels] = torch.tensor(correct_positions).float()

        if q_in_g:
            same_labels[idx_q_is_g_in_same_labels] = True

        rest_of_distances = set(range(num_gallery - int(q_in_g))) - set(correct_positions)
        rest_of_distances = torch.tensor(sorted(rest_of_distances)).float()

        distances[row, ~same_labels] = rest_of_distances

    distances += 1
    distances[mask_to_ignore] = 0

    return distances
