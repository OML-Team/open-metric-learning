import numpy as np
from torch import LongTensor

from oml.const import OVERALL_CATEGORIES_KEY
from oml.functional.metrics import calc_retrieval_metrics
from oml.utils.misc import compare_dicts_recursively


def test_categories_in_metrics() -> None:
    args = {
        "retrieved_ids": LongTensor([[0, 5, 4], [2, 1, 5], [2, 1, 5]]),
        "gt_ids": [LongTensor([0, 10, 4]), LongTensor([2, 3]), LongTensor([2, 3])],
        "cmc_top_k": (1,),
        "precision_top_k": (3, 5),
        "map_top_k": tuple(),
    }

    # TEST WITH METRICS
    metrics_overall = {"cmc": {1: 1.0}, "precision": {3: (2 / 3 + 2 * 1 / 2) / 3, 5: (2 / 3 + 2 * 1 / 2) / 3}}
    metrics = calc_retrieval_metrics(query_categories=None, **args)
    assert compare_dicts_recursively(metrics_overall, metrics)

    # TEST WITHOUT METRICS
    metrics = calc_retrieval_metrics(query_categories=np.array(["cat", "dog", "dog"]), **args)
    expected_metrics = {
        "cat": {"cmc": {1: 1.0}, "precision": {3: 2 / 3, 5: 2 / 3}},
        "dog": {"cmc": {1: 1.0}, "precision": {3: 1 / 2, 5: 1 / 2}},
        OVERALL_CATEGORIES_KEY: metrics_overall,
    }
    assert compare_dicts_recursively(expected_metrics, metrics)
