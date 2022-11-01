from functools import partial

import matplotlib.pyplot as plt
import torch

from oml.const import (
    CATEGORIES_KEY,
    EMBEDDINGS_KEY,
    IS_GALLERY_KEY,
    IS_QUERY_KEY,
    LABELS_KEY,
    PATHS_KEY,
    get_cache_folder,
)
from oml.metrics.embeddings import EmbeddingMetrics
from oml.utils.misc import one_hot

oh = partial(one_hot, dim=8)


def test_visualization() -> None:

    dummy_image = [[0]]

    cf = get_cache_folder()
    plt.imsave(cf / "temp.png", dummy_image)

    batch1 = {
        EMBEDDINGS_KEY: torch.stack([oh(1) * 2, oh(1) * 3, oh(0)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([True, True, True]),
        IS_GALLERY_KEY: torch.tensor([False, False, False]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
        PATHS_KEY: [cf / "temp.png", cf / "temp.png", cf / "temp.png"],
    }

    batch2 = {
        EMBEDDINGS_KEY: torch.stack([oh(0), oh(1), oh(1)]),
        LABELS_KEY: torch.tensor([0, 1, 1]),
        IS_QUERY_KEY: torch.tensor([False, False, False]),
        IS_GALLERY_KEY: torch.tensor([True, True, True]),
        CATEGORIES_KEY: torch.tensor([10, 20, 20]),
        PATHS_KEY: [cf / "temp.png", cf / "temp.png", cf / "temp.png"],
    }

    calc = EmbeddingMetrics(
        embeddings_key=EMBEDDINGS_KEY,
        labels_key=LABELS_KEY,
        is_query_key=IS_QUERY_KEY,
        is_gallery_key=IS_GALLERY_KEY,
        extra_keys=(PATHS_KEY,),
        map_top_k=(2,),
    )

    num_samples = len(batch1[LABELS_KEY]) + len(batch2[LABELS_KEY])

    calc.setup(num_samples=num_samples)
    calc.update_data(batch1)
    calc.update_data(batch2)

    calc.compute_metrics()

    figs, log_strs = calc.visualize()

    (cf / "temp.png").unlink()

    assert len(figs) == len(log_strs)
