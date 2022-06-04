import random
from typing import Any, Callable, Type

import numpy as np
import pytest
import torch

from oml.utils.images.images_resize import (
    inverse_karesize_bboxes,
    inverse_karesize_image,
    karesize_bboxes,
    karesize_image,
)


@pytest.mark.parametrize("constructor,allclose_func", [(np.array, np.allclose), (torch.tensor, torch.allclose)])
def test_karesize_bboxes(constructor: Type, allclose_func: Callable[[Any], Any]) -> None:  # type: ignore
    min_dim = 200
    max_dim = 600
    max_bboxes = 10
    num_attempts = 100

    for _ in range(num_attempts):
        src_hw = (random.randint(min_dim, max_dim), random.randint(min_dim, max_dim))
        new_hw = (random.randint(min_dim, max_dim), random.randint(min_dim, max_dim))
        num_bboxes = random.randint(1, max_bboxes)

        left_points = [random.uniform(0, src_hw[1] - 3) for _ in range(num_bboxes)]
        right_points = [random.uniform(left + 1, src_hw[1] - 1) for left in left_points]
        top_points = [random.uniform(0, src_hw[0] - 3) for _ in range(num_bboxes)]
        bottom_points = [random.uniform(top + 1, src_hw[0] - 1) for top in top_points]

        bboxes = list(zip(left_points, top_points, right_points, bottom_points))

        raw_bboxes = constructor(bboxes)
        new_bboxes = karesize_bboxes(raw_bboxes, src_hw, new_hw)
        prev_bboxes = inverse_karesize_bboxes(new_bboxes, new_hw, src_hw)

        assert allclose_func(prev_bboxes, raw_bboxes, rtol=1e-3)  # type: ignore


@pytest.mark.parametrize("framework,ndim", [("np", 2), ("np", 3), ("torch", 2), ("torch", 2), ("torch", 4)])
def test_karesize_images(framework: str, ndim: int) -> None:
    min_dim = 200
    max_dim = 600
    num_attempts = 20
    max_extra_dims = 5

    for _ in range(num_attempts):
        src_hw = (random.randint(min_dim, max_dim), random.randint(min_dim, max_dim))
        new_hw = (random.randint(min_dim, max_dim), random.randint(min_dim, max_dim))

        src_shape = src_hw
        new_shape = new_hw

        for _ in range(ndim - 2):
            extra_dim = random.randint(1, max_extra_dims)
            if framework == "np":
                src_shape = (*src_shape, extra_dim)  # type: ignore
                new_shape = (*new_shape, extra_dim)  # type: ignore
            else:
                src_shape = (extra_dim, *src_shape)  # type: ignore
                new_shape = (extra_dim, *new_shape)  # type: ignore

        if framework == "np":
            raw_image = np.random.random(src_shape)
        else:
            raw_image = torch.randn(src_shape)

        new_image = karesize_image(raw_image, new_hw)
        assert new_image.shape == new_shape
        prev_image = inverse_karesize_image(new_image, src_hw)
        assert prev_image.shape == src_shape
