from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import torch

from oml.interfaces.models import IExtractor
from oml.models.audio.ecapa_tdnn.external.model import ECAPA_TDNN
from oml.models.utils import TStateDict
from oml.utils.io import download_checkpoint
from oml.utils.misc_torch import normalise


def ecapa_tdnn_taoruijie() -> ECAPA_TDNN:
    return ECAPA_TDNN(C=1024)


class ECAPATDNNExtractor(IExtractor):
    constructors = {
        "ecapa_tdnn_taoruijie": ecapa_tdnn_taoruijie,
    }
    pretrained_models = {
        "ecapa_tdnn_taoruijie": {
            "url": "https://github.com/TaoRuijie/ECAPA-TDNN/raw/refs/heads/main/exps/pretrain.model",
            "hash": "2bbc0ea09a41217eaff16b676ecb63d7",
            "fname": "ecapa_tdnn_taoruijie.pth",
            "init_args": {
                "arch": "ecapa_tdnn_taoruijie",
                "normalise_features": False,
            },
        }
    }

    def __init__(
        self,
        weights: Optional[Union[Path, str]],
        arch: str,
        normalise_features: bool = False,
    ):
        """
        Args:
            weights: Path to weights or special key for pretrained ones or ``None`` for random initialization.
             You can check available pretrained checkpoints in ``ECAPATDNNExtractor.pretrained_models``.
            arch: Model architecture, currently only supports ``ecapa_tdnn_taoruijie``.
            normalise_features: Set ``True`` to normalise output features.
        """
        super().__init__()

        assert arch in self.constructors, f"Unknown architecture: {arch}"

        self.normalise_features = normalise_features
        self.model = self.constructors[arch]()

        if weights is None:
            return

        if weights in self.pretrained_models:
            pretrained = self.pretrained_models[weights]  # type: ignore
            weights = download_checkpoint(
                pretrained["url"],  # type: ignore
                pretrained["hash"],  # type: ignore
                fname=pretrained["fname"],  # type: ignore
            )
            state_dict = self.prepare_state_dict(weights, filter_prefix="speaker_encoder.")
        else:
            state_dict = self.prepare_state_dict(weights, filter_prefix="model.")

        self.model.load_state_dict(state_dict, strict=True)

    def prepare_state_dict(self, weights_path: Union[Path, str], filter_prefix: Optional[str] = None) -> TStateDict:
        state_dict = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if filter_prefix is not None:
            state_dict = OrderedDict(
                (k[len(filter_prefix) :], v) for k, v in state_dict.items() if k.startswith(filter_prefix)
            )
        return state_dict

    @property
    def feat_dim(self) -> int:
        return self.model.fc6.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2 or (
            x.ndim == 3 and x.shape[1] == 1
        ), "The model expects input audio to have shape (batch_size, n_samples) or (batch_size, 1, n_samples)"

        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.model.forward(x, aug=False)
        if self.normalise_features:
            x = normalise(x)
        return x


__all__ = ["ECAPATDNNExtractor"]
