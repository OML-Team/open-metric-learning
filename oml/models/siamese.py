import torch
from torch import Tensor
from torchvision.models import resnet18

from oml.interfaces.models import IPairwiseDistanceModel
from oml.utils.misc_torch import elementwise_dist


class LinearSiamese(IPairwiseDistanceModel):
    """
    Model takes two embeddings as inputs, transforms them and estimates the
    corresponding *distance* (not in a strictly mathematical sense) after the transformation.

    """

    def __init__(self, feat_dim: int, identity_init: bool):
        """
        Args:
            feat_dim: Expected size of each input.
            identity_init: If ``True``, models' weights initialised in a way when
                model simply estimates L2 distance between the original embeddings.

        """
        super(LinearSiamese, self).__init__()
        self.feat_dim = feat_dim

        self.proj = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)

        if identity_init:
            self.proj.load_state_dict({"weight": torch.eye(feat_dim)})

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: Embedding with the shape of ``[batch_size, feat_dim]``
            x2: Embedding with the shape of ``[batch_size, feat_dim]``

        Returns:
            *Distance* between transformed inputs.

        """
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        y = elementwise_dist(x1, x2, p=2)
        return y


class ResNetSiamese(IPairwiseDistanceModel):
    def __init__(self, pretrained: bool) -> None:
        super(ResNetSiamese, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """

        Args:
            x1: The first input image.
            x2: The second input image.

        Returns:
            *Distance* between images (a logit of it).

        """
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x = elementwise_dist(x1, x2, p=2)
        return x


__all__ = ["LinearSiamese", "ResNetSiamese"]
