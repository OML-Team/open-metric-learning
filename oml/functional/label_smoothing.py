from typing import Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def label_smoothing(
    y: torch.Tensor,
    num_classes: int,
    epsilon: float = 0.2,
    categories: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    This function is doing `label smoothing <https://arxiv.org/pdf/1512.00567v3.pdf>`_.
    You can also use modified version, where the label is smoothed only for the category corresponding to sample's
    ground truth label. To use this, you should provide the ``categories`` argument: vector, for which i-th entry
    is a corresponding category for label ``i``.

    Args:
        y: Ground truth labels with the size of batch_size where each element is from 0 (inclusive) to
            num_classes (exclusive).
        num_classes: Number of classes in total
        epsilon: Power of smoothing. The biggest value in OHE-vector will be
            ``1 - epsilon + 1 / num_classes`` after the transformation
        categories: Vector for which i-th entry is a corresponding category for label ``i``. Optional, used for
            category-based label smoothing. In that case the biggest value in OHE-vector will be
            ``1 - epsilon + 1 / num_classes_of_the_same_category``, labels outside of the category will not change
    """
    assert epsilon < 1, "`epsilon` must be less than 1."
    ohe = F.one_hot(y, num_classes).float()
    if categories is not None:
        ohe *= 1 - epsilon
        same_category_mask = categories[y].tile(num_classes, 1).t() == categories
        return torch.where(same_category_mask, epsilon / same_category_mask.sum(-1).view(-1, 1), 0) + ohe
    else:
        ohe *= 1 - epsilon
        ohe += epsilon / num_classes
        return ohe
