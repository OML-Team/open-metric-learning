import torch
from torch import Tensor, nn
from torchvision.models import resnet50

from oml.interfaces.models import IPairwiseDistanceModel
from oml.utils.misc_torch import elementwise_dist


class SimpleSiamese(IPairwiseDistanceModel):
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
        super(SimpleSiamese, self).__init__()
        self.feat_dim = feat_dim

        self.proj1 = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)
        self.proj2 = torch.nn.Linear(in_features=feat_dim, out_features=feat_dim, bias=False)

        if identity_init:
            self.proj1.load_state_dict({"weight": torch.eye(feat_dim)})
            self.proj2.load_state_dict({"weight": torch.eye(feat_dim)})

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: Embedding with the shape of ``[batch_size, feat_dim]``
            x2: Embedding with the shape of ``[batch_size, feat_dim]``

        Returns:
            *Distance* between transformed inputs.

        """
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        y = elementwise_dist(x1, x2, p=2)
        return y


class ImagesSiamese(IPairwiseDistanceModel):
    def __init__(self) -> None:
        super(ImagesSiamese, self).__init__()
        self.model = resnet50(pretrained=True)
        self.fc = nn.Linear(in_features=1000 * 2, out_features=1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.model(x1)
        x2 = self.model(x2)

        x = torch.concat([x1 + x2, x1 * x2], dim=1)
        x = self.fc(x)

        return x


__all__ = ["SimpleSiamese", "ImagesSiamese"]
