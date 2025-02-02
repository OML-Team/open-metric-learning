from typing import Any, Union

import numpy as np
import pytest
from PIL import Image

from oml.models import ResnetExtractor, ViTExtractor

IMAGE_SIZE = (224, 224, 3)


def get_image_pillow() -> Image.Image:
    img = np.random.randint(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
    return Image.fromarray(img)


def get_numpy_image() -> np.ndarray:
    return np.random.randint(0, 256, size=IMAGE_SIZE, dtype=np.uint8)


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize(
    "draw_function",
    [
        ViTExtractor(None, "vits16", normalise_features=False).draw_attention,
        ResnetExtractor(None, "resnet50", normalise_features=False, gem_p=None, remove_fc=False).draw_gradcam,
    ],
)
@pytest.mark.parametrize(
    "image",
    [
        get_numpy_image(),
        get_image_pillow(),
    ],
)
def test_visualisation(draw_function: Any, image: Union[np.ndarray, Image.Image]) -> None:
    image_modified = draw_function(image)

    assert isinstance(image_modified, type(image))

    image_modified = np.array(image_modified)

    assert 0 <= image_modified.min() <= image_modified.max() <= 255
    assert image_modified.shape == IMAGE_SIZE


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize(
    "draw_function",
    [
        ViTExtractor(None, "vits16", normalise_features=False).draw_attention,
        ResnetExtractor(None, "resnet50", normalise_features=False, gem_p=None, remove_fc=False).draw_gradcam,
    ],
)
def test_visualisation_2(draw_function: Any) -> None:
    np_image = get_numpy_image()
    pil_image = Image.fromarray(np_image)

    out1 = draw_function(np_image)
    out2 = np.array(draw_function(pil_image))

    assert (out1 == out2).all()
