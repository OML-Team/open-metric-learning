from typing import Tuple

import pytest
from torch import Tensor



@pytest.fixture
def independent_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    pass


@pytest.fixture
def shared_query_gallery_case() -> Tuple[Tensor, Tensor, Tensor]:
    pass

@pytest.mark.parametrize("top_n", [])
def test_trivial_processing_does_not_change_distances_order(top_n: int) -> None:
    pass