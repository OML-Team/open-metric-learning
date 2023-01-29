from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn.modules.activation import Sigmoid
from torchvision.ops import MLP

from oml.interfaces.models import IExtractor, IFreezable, IPairwiseModel
from oml.models.utils import remove_prefix_from_state_dict
from oml.utils.io import download_checkpoint
from oml.utils.misc_torch import elementwise_dist


class LinearTrivialDistanceSiamese(IPairwiseModel):
    """
    This model is a useful tool mostly for development.

    """

    def __init__(self, feat_dim: int, identity_init: bool):
        """
        Args:
            feat_dim: Expected size of each input.
            identity_init: If ``True``, models' weights initialised in a way when
                the model simply estimates L2 distance between the original embeddings.

        """
        super(LinearTrivialDistanceSiamese, self).__init__()
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


class ConcatSiamese(IPairwiseModel, IFreezable):
    """
    This model concatenates two inputs and passes them through
    a given backbone and applyies a head after that.

    """

    pretrained_models: Dict[str, Any] = {}

    def __init__(
        self,
        extractor: IExtractor,
        mlp_hidden_dims: List[int],
        weights: Optional[Union[str, Path]] = None,
        strict_load: bool = True,
    ) -> None:
        """
        Args:
            extractor: Instance of ``IExtractor`` (e.g. ``ViTExtractor``)
            mlp_hidden_dims: Hidden dimensions of the head
            weights: Path to weights file or ``None`` for random initialization
            strict_load: Whether to use ``self.load_state_dict`` with strict argument

        """
        super(ConcatSiamese, self).__init__()
        self.extractor = extractor

        self.head = MLP(
            in_channels=self.extractor.feat_dim,
            hidden_channels=[*mlp_hidden_dims, 1],
            activation_layer=Sigmoid,
            dropout=0.5,
            inplace=None,
        )

        # turn off the last bias
        self.head[-2] = nn.Linear(self.head[-2].in_features, self.head[-2].out_features, bias=False)

        # turn off the last dropout
        self.head[-1] = nn.Identity()

        self.train_backbone = True

        if weights:
            if weights in self.pretrained_models:
                url_or_fid, hash_md5, fname = self.pretrained_models[weights]  # type: ignore
                weights = download_checkpoint(url_or_fid=url_or_fid, hash_md5=hash_md5, fname=fname)

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remove_prefix_from_state_dict(loaded, trial_key="extractor.")
            self.load_state_dict(loaded, strict=strict_load)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x = torch.concat([x1, x2], dim=2)

        with torch.set_grad_enabled(self.train_backbone):
            x = self.extractor(x)

        x = self.head(x)
        x = x.view(len(x))

        return x

    def freeze(self) -> None:
        self.train_backbone = False

    def unfreeze(self) -> None:
        self.train_backbone = True


class TrivialDistanceSiamese(IPairwiseModel):
    """
    This model is a useful tool mostly for development.

    """

    pretrained_models: Dict[str, Any] = {}

    def __init__(self, extractor: IExtractor) -> None:
        """
        Args:
            extractor: Instance of ``IExtractor`` (e.g. ``ViTExtractor``)

        """
        super(TrivialDistanceSiamese, self).__init__()
        self.extractor = extractor

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: The first input.
            x2: The second input.

        Returns:
            Distance between inputs.

        """
        x1 = self.extractor(x1)
        x2 = self.extractor(x2)
        return elementwise_dist(x1, x2, p=2)


__all__ = ["LinearTrivialDistanceSiamese", "ConcatSiamese", "TrivialDistanceSiamese"]
