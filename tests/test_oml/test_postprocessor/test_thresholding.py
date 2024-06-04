from torch import FloatTensor, LongTensor

from oml.retrieval.postprocessors.thresholding import (
    AdaptiveThresholding,
    ConstantThresholding,
)
from oml.retrieval.retrieval_results import RetrievalResults
from tests.utils import check_if_sequence_of_tensors_are_equal


def test_adaptive_thresholding() -> None:
    AdaptiveThresholding(th=0.5)
    assert True


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
