import cv2
import numpy as np
import pytest

from oml.const import TMP_PATH
from oml.utils.images.images import imread_cv2, imread_pillow


@pytest.mark.parametrize("img_format, no_compression", [("png", True), ("png", False), ("jpg", False)])
@pytest.mark.parametrize("num_channels", [3, 4])
def test_readers(img_format: str, num_channels: int, no_compression: bool) -> None:
    shape_hw = (333, 257)
    dummy_image = np.random.randint(0, 255, (*shape_hw, num_channels), dtype=np.uint8)

    fname_image = str(TMP_PATH / f"img_test_readers.{img_format}")

    if img_format == "png" and no_compression:
        cv2.imwrite(fname_image, cv2.cvtColor(dummy_image, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(fname_image, cv2.cvtColor(dummy_image, cv2.COLOR_RGBA2BGRA))

    image_cv2_from_path = imread_cv2(fname_image)
    image_pil_from_path = np.array(imread_pillow(fname_image))

    with open(fname_image, "rb") as fin:
        image_bytes = fin.read()

    image_cv2_from_bytes = imread_cv2(image_bytes)
    image_pil_from_bytes = np.array(imread_pillow(image_bytes))

    for image in [image_cv2_from_path, image_pil_from_path, image_cv2_from_bytes, image_pil_from_bytes]:
        assert image.shape == (*shape_hw, 3)
        if no_compression:
            assert np.array_equal(image, dummy_image[:, :, :3]), (image == dummy_image[:, :, :3]).mean()
        else:
            assert np.array_equal(image, image_cv2_from_path), (image == image_cv2_from_path).mean()
