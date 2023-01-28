import torch
from torch import Tensor, nn
from torchvision.models import resnet18

from oml.interfaces.models import IExtractor, IFreezable, IPairwiseModel
from oml.utils.misc_torch import elementwise_dist


class LinearSiamese(IPairwiseModel):
    """
    The model takes two embeddings as inputs, transforms them linearly
    and estimates the distance between them.

    """

    def __init__(self, feat_dim: int, identity_init: bool):
        """
        Args:
            feat_dim: Expected size of each input.
            identity_init: If ``True``, models' weights initialised in a way when
                the model simply estimates L2 distance between the original embeddings.

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
            Distance between transformed inputs.

        """
        x1 = self.proj(x1)
        x2 = self.proj(x2)
        y = elementwise_dist(x1, x2, p=2)
        return y


class ResNetSiamese(IPairwiseModel):
    """
    The model takes two images as inputs, passes them through backbone and
    estimates the distance between them.

    """

    def __init__(self, pretrained: bool) -> None:
        """
        Args:
            pretrained: Set ``True`` if you want to use pretrained model.

        """
        super(ResNetSiamese, self).__init__()
        self.backbone = resnet18(pretrained=pretrained)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """

        Args:
            x1: The first input image.
            x2: The second input image.

        Returns:
            Distance between images.

        """
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        x = elementwise_dist(x1, x2, p=2)
        return x


class ImagesSiamese(IPairwiseModel, IFreezable):
    """
    This model concatenates two inputs and passes them through
    a given backbone and applyies a head after that.
    """

    def __init__(self, backbone: IExtractor) -> None:
        super(ImagesSiamese, self).__init__()
        self.extractor = backbone
        feat_dim = self.extractor.feat_dim

        # todo: parametrize
        self.head = nn.Sequential(
            *[
                nn.Linear(feat_dim, feat_dim // 2, bias=True),
                nn.Dropout(),
                nn.Sigmoid(),
                nn.Linear(feat_dim // 2, 1, bias=False),
            ]
        )

        self.train_backbone = True

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.concat([x1, x2], dim=2)

        with torch.set_grad_enabled(self.train_backbone):
            x = self.extractor(x)

        x = self.head(x)
        x = x.squeeze()

        return


__all__ = ["LinearSiamese", "ResNetSiamese"]
