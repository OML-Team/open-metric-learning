# flake8: noqa
from oml.miners.inbatch_all_tri import AllTripletsMiner
from tests.test_oml.test_miners.shared_checkers import (
    check_all_triplets_number,
    check_triplets_consistency,
)


def test_all_triplets_miner(features_and_labels) -> None:  # type: ignore
    """
    Args:
        features_and_labels: Features and valid labels

    """
    max_tri = 512
    miner = AllTripletsMiner(max_output_triplets=max_tri)

    for _, labels in features_and_labels:
        ids_a, ids_p, ids_n = miner._sample(labels=labels)

        check_all_triplets_number(labels=labels, max_tri=max_tri, num_selected_tri=len(ids_a))

        check_triplets_consistency(ids_anchor=ids_a, ids_pos=ids_p, ids_neg=ids_n, labels=labels)
