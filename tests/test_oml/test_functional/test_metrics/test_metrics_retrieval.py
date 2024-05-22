import numpy as np
from torch import FloatTensor, LongTensor

from oml.const import OVERALL_CATEGORIES_KEY
from oml.functional.metrics import calc_retrieval_metrics
from oml.utils.misc import compare_dicts_recursively


def test_categories_in_metrics() -> None:
    args = {
        "retrieved_ids": [LongTensor([0, 5, 4]), LongTensor([2, 1, 5]), LongTensor([2, 1, 5])],
        "gt_ids": [LongTensor([0, 10, 4]), LongTensor([2, 3]), LongTensor([2, 3])],
        "cmc_top_k": (1,),
        "precision_top_k": (3, 5),
        "map_top_k": tuple(),
    }

    # TEST WITHOUT CATEGORIES
    metrics_overall = {"cmc": {1: 1.0}, "precision": {3: (2 / 3 + 2 * 1 / 2) / 3, 5: (2 / 3 + 2 * 1 / 2) / 3}}
    metrics = calc_retrieval_metrics(query_categories=None, **args)  # type: ignore
    assert compare_dicts_recursively(metrics_overall, metrics)

    # TEST WITH CATEGORIES
    metrics = calc_retrieval_metrics(query_categories=np.array(["cat", "dog", "dog"]), **args)  # type: ignore
    metrics_expected = {
        "cat": {"cmc": {1: 1.0}, "precision": {3: 2 / 3, 5: 2 / 3}},
        "dog": {"cmc": {1: 1.0}, "precision": {3: 1 / 2, 5: 1 / 2}},
        OVERALL_CATEGORIES_KEY: metrics_overall,
    }
    assert compare_dicts_recursively(metrics_expected, metrics)


def test_empty_predictions() -> None:
    retrieved_ids = [LongTensor([]), LongTensor([]), LongTensor([])]
    gt_ids = [LongTensor([0]), LongTensor([]), LongTensor([10, 20])]

    metrics = calc_retrieval_metrics(
        retrieved_ids=retrieved_ids, gt_ids=gt_ids, cmc_top_k=(3,), precision_top_k=(2,), map_top_k=(1,), reduce=False
    )

    metrics_expected = {
        "cmc": {3: FloatTensor([0.0, 1.0, 0.0])},
        "precision": {2: FloatTensor([0.0, 1.0, 0.0])},
        "map": {1: FloatTensor([0.0, 1.0, 0.0])},
    }
    assert compare_dicts_recursively(metrics_expected, metrics)
