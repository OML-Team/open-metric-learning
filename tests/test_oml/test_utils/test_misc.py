from typing import Any

import numpy as np
import pytest

from oml.utils.misc import find_first_occurrences, smart_sample


@pytest.mark.long
def test_sample_enough_items() -> None:
    """Check smart_sample in case of n_sample < len(array)."""
    for _ in range(10):
        size = np.random.randint(10, 150)
        array = list(range(size))
        n_samples = np.random.randint(1, 9)
        samples = smart_sample(array=array, k=n_samples)
        assert len(set(samples)) == n_samples
        assert len(samples) == n_samples
        assert set(samples) <= set(array)


@pytest.mark.long
def test_sample_not_enough_items() -> None:
    """Check smart_sample in case of n_sample > len(array)."""
    for _ in range(10):
        size = np.random.randint(2, 25)
        array = list(range(size))
        n_samples = np.random.randint(size + 1, 50)
        samples = smart_sample(array=array, k=n_samples)
        assert len(set(samples)) == size
        assert len(samples) == n_samples
        assert set(samples) == set(array)


@pytest.fixture()
def first_occurrences_test_data() -> Any:
    data: Any = (
        ([], []),
        ([0, 1, 2, 1, 1], [0, 1, 2]),
        ([10, 10, 10], [0]),
        ([15, 20, 40, 10, 10], [0, 1, 2, 3]),
        ([0, 1, 1, 1, 1, 0], [0, 1]),
    )
    return data


@pytest.mark.long
def test_find_first_occurrences(first_occurrences_test_data) -> None:  # type: ignore
    for x, expected in first_occurrences_test_data:
        assert expected == find_first_occurrences(x)
