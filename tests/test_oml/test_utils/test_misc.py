import numpy as np
import pytest

from oml.utils.misc import pad_array_right, smart_sample


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


def test_pad_array_right() -> None:
    arr = np.array([1.5, 2, 3])
    sz = 5
    val = -100.0
    arr_pad_expected = np.array([1.5, 2, 3, val, val])
    arr_pad = pad_array_right(arr, sz, val)

    assert np.allclose(arr_pad_expected, arr_pad)
