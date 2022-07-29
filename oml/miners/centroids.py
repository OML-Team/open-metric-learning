from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from oml.interfaces.miners import ITripletsMiner
from oml.utils.misc import find_value_ids


class CentroidsTripletMiner(ITripletsMiner):
    def __init__(self, use_anchor: bool = True) -> None:
        super().__init__()
        self.use_anchor = use_anchor

    def sample(self, features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        ids_all = set(range(len(labels)))

        features_pos = torch.empty_like(features)  # TODO: check if this broke the gradients
        features_neg = torch.empty_like(features)

        for i_anch, label in enumerate(labels):
            ids_label = set(find_value_ids(it=labels, value=label))
            ids_neg = ids_all - ids_label

            if not self.use_anchor:
                ids_label -= {i_anch}

            ids_pos_cur = np.array(list(ids_label), int)
            ids_neg_cur = np.array(list(ids_neg), int)

            features_pos[i_anch] = features[ids_pos_cur].mean(0)
            features_neg[i_anch] = features[ids_neg_cur].mean(0)

        return features, features_pos, features_neg
