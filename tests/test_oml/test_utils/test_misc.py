import numpy as np
import pytest
from matplotlib import pyplot as plt

from oml.const import RED
from oml.utils.misc import pad_array_right, smart_sample, visualise_text


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


def check_image_has_content(image: np.ndarray) -> bool:
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    return unique_colors.shape[0] > 1


def test_visualise_text() -> None:
    img = visualise_text(text="Hello world", color=RED, draw_bbox=False)
    plt.imshow(img)
    plt.show()
    assert check_image_has_content(img)

    # we check the function works on a single extremely huge word
    img = visualise_text(text="Hello" * 100, color=RED, draw_bbox=False)
    plt.imshow(img)
    plt.show()
    assert check_image_has_content(img)

    # the same, but there are several huge words
    img = visualise_text(text="Hello" * 50 + " HELLO " + "Hello" * 50, color=RED, draw_bbox=False)
    plt.imshow(img)
    plt.show()
    assert check_image_has_content(img)
