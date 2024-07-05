from pathlib import Path
from typing import List, Optional, Union

import torch
from torchvision.ops import MLP

from oml.const import STORAGE_CKPTS
from oml.interfaces.models import IExtractor, IFreezable
from oml.models.utils import (
    remove_criterion_in_state_dict,
    remove_prefix_from_state_dict,
)
from oml.models.vit_dino.extractor import ViTExtractor
from oml.utils.io import download_checkpoint


class ExtractorWithMLP(IExtractor, IFreezable):
    """
    Class-wrapper for extractors which an additional MLP.

    """

    pretrained_models = {
        "vits16_224_mlp_384_inshop": {  # type: ignore
            "url": f"{STORAGE_CKPTS}/inshop/vits16_224_mlp_384_inshop.ckpt",
            "hash": "35244966",
            "fname": "vits16_224_mlp_384_inshop.ckpt",
            "init_args": {
                "extractor_creator": lambda: ViTExtractor(None, "vits16", False, use_multi_scale=False),
                "mlp_features": [384],
                "train_backbone": True,
            },
        }
    }

    def __init__(
        self,
        extractor: IExtractor,
        mlp_features: List[int],
        weights: Optional[Union[str, Path]] = None,
        train_backbone: bool = False,
    ):
        """
        Args:
            extractor: Instance of ``IExtractor`` (e.g. ``ViTExtractor``)
            mlp_features: Sizes of projection layers
            weights: Path to weights file or ``None`` for random initialization
            train_backbone: set ``False`` if you want to train only MLP head

        """
        IExtractor.__init__(self)

        self.extractor = extractor
        self.mlp_features = mlp_features
        self.train_backbone = train_backbone

        self.projection = MLP(self.extractor.feat_dim, self.mlp_features)

        if weights:
            if weights in self.pretrained_models:
                pretrained = self.pretrained_models[weights]  # type: ignore
                weights = download_checkpoint(
                    url_or_fid=pretrained["url"],  # type: ignore
                    hash_md5=pretrained["hash"],  # type: ignore
                    fname=pretrained["fname"],  # type: ignore
                )

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remove_criterion_in_state_dict(loaded)
            loaded = remove_prefix_from_state_dict(loaded, trial_key="extractor.")
            self.load_state_dict(loaded, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.train_backbone):
            features = self.extractor(x)

        return self.projection(features)

    @property
    def feat_dim(self) -> int:
        return self.mlp_features[-1]

    def freeze(self) -> None:
        self.train_backbone = False

    def unfreeze(self) -> None:
        self.train_backbone = True

    @classmethod
    def from_pretrained(cls, weights: str, **kwargs) -> "IExtractor":  # type: ignore
        # The current class takes another model as a constructor's argument, so, they need to be
        # in the `self.pretrained_models`. The problem is these models will be instantiated even if we simply
        # import something from the current module. To avoid it we added the logic of wrapping/unwrapping
        # constructors into lambda functions.

        ini = cls.pretrained_models[weights]["init_args"]
        ini["extractor"] = ini.pop("extractor_creator")()  # type: ignore

        return super().from_pretrained(weights, **kwargs)


__all__ = ["ExtractorWithMLP"]
