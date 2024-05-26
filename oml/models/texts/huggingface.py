from typing import Any

from torch import FloatTensor, Tensor

from oml.interfaces.models import IExtractor

THuggingFaceModel = Any


class HFWrapper(IExtractor):
    """
    This is a wrapper for models from HuggingFace (transformers) library.
    Please, double check `forward` method to learn how we extrac features.

    """

    def __init__(self, model: THuggingFaceModel, feat_dim: int):
        super().__init__()
        self.model = model
        self._feat_dim = feat_dim

    def forward(self, x: Tensor) -> FloatTensor:
        hf_output = self.model(x)
        embedding = hf_output["last_hidden_state"][:, 0, :]
        return embedding

    def feat_dim(self) -> int:
        return self._feat_dim


__all_ = ["HuggingFaceWrapper"]
