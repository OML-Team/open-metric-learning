from torch import FloatTensor, LongTensor

from oml.retrieval.postprocessors.thresholding import (
    AdaptiveThresholding,
    ConstantThresholding,
)
from oml.retrieval.retrieval_results import RetrievalResults
from tests.utils import check_if_sequence_of_tensors_are_equal


def test_thresholding() -> None:
    th = 0.2

    rr = RetrievalResults(
        distances=[
            FloatTensor([0.0, 0.05, 0.07, 0.1, 0.12, 0.2, 0.2, 0.4]),
            FloatTensor([]),
            FloatTensor([0.21, 0.25]),
        ],
        retrieved_ids=[LongTensor([0, 1, 2, 3, 4, 5, 6, 7]), LongTensor([]), LongTensor([3, 5])],
        gt_ids=[
            LongTensor([0, 1]),
            LongTensor([1, 500]),
            LongTensor([10, 20]),
        ],
    )

    rr_upd_expected = RetrievalResults(
        distances=[FloatTensor([0.0, 0.05, 0.07, 0.1, 0.12]), FloatTensor([]), FloatTensor([])],
        retrieved_ids=[LongTensor([0, 1, 2, 3, 4]), LongTensor([]), LongTensor([])],
        gt_ids=[
            LongTensor([0, 1]),
            LongTensor([1, 500]),
            LongTensor([10, 20]),
        ],
    )

    processor = ConstantThresholding(th=th)

    rr_upd = processor.process(rr)

    assert check_if_sequence_of_tensors_are_equal(rr_upd_expected.distances, rr_upd.distances)
    assert check_if_sequence_of_tensors_are_equal(rr_upd_expected.retrieved_ids, rr_upd.retrieved_ids)
    assert check_if_sequence_of_tensors_are_equal(rr_upd_expected.gt_ids, rr_upd.gt_ids)


def test_adaptive_thresholding() -> None:
    n_std = 3

    rr = RetrievalResults(
        distances=[
            FloatTensor([1, 2, 3, 4, 10, 11, 12, 13]),
            FloatTensor([]),
            FloatTensor([4, 5, 6, 7]),
            FloatTensor([3, 4, 5, 12, 13, 14, 20, 21, 22]),
            FloatTensor([100]),
        ],
        retrieved_ids=[
            LongTensor([1, 2, 3, 4, 10, 11, 12, 13]),
            LongTensor([]),
            LongTensor([4, 5, 6, 7]),
            LongTensor([3, 4, 5, 12, 13, 14, 20, 21, 22]),
            LongTensor([100]),
        ],
        gt_ids=[
            LongTensor([0, 1]),
            LongTensor([1, 0]),
            LongTensor([10, 20]),
            LongTensor([30, 20]),
            LongTensor([1]),
        ],
    )

    rr_upd_expected = RetrievalResults(
        distances=[
            FloatTensor([1, 2, 3, 4]),
            FloatTensor([]),
            FloatTensor([4, 5, 6, 7]),
            FloatTensor(
                [
                    3,
                    4,
                    5,
                ]
            ),
            FloatTensor([100]),
        ],
        retrieved_ids=[
            LongTensor([1, 2, 3, 4]),
            LongTensor([]),
            LongTensor([4, 5, 6, 7]),
            LongTensor([3, 4, 5]),
            LongTensor([100]),
        ],
        gt_ids=[
            LongTensor([0, 1]),
            LongTensor([1, 0]),
            LongTensor([10, 20]),
            LongTensor([30, 20]),
            LongTensor([1]),
        ],
    )

    processor = AdaptiveThresholding(n_std=n_std)

    rr_upd = processor.process(rr)

    assert check_if_sequence_of_tensors_are_equal(rr_upd_expected.distances, rr_upd.distances)
    assert check_if_sequence_of_tensors_are_equal(rr_upd_expected.retrieved_ids, rr_upd.retrieved_ids)
    assert check_if_sequence_of_tensors_are_equal(rr_upd_expected.gt_ids, rr_upd.gt_ids)
