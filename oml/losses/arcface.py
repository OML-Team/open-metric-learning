import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor


# todo check implementation
class ArcFace(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(self, s: float = 64.0, margin: float = 0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: Tensor, labels: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        labels = labels.to(logits.device)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale

        loss = torch.nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        return loss, {}
