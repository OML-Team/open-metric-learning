import numpy as np
from typing import List, Any


def smart_sample(array: List[Any], n_samples: int) -> List[Any]:
    """Sample n_samples items from given list. If array contains at least n_samples items, sample without repetition;
    otherwise take all the unique items and sample n_samples - len(array) ones with repetition.

    Args:
        array: list of unique elements to sample from
        n_samples: number of items to sample

    Returns:
        sampled_items: list of sampled items
    """
    array_size = len(array)
    if array_size < n_samples:
        samples_indices = array + np.random.choice(
            array,
            size=n_samples - array_size,
            replace=True,
        ).tolist()
    else:
        samples_indices = np.random.choice(
            array,
            size=n_samples,
            replace=False,
        ).tolist()
    return samples_indices
