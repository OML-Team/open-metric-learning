from typing import Any

from torch import FloatTensor

from oml.interfaces.models import IExtractor

# We don't use actual types programmatically in order not to bring optional dependency here
THuggingFaceModel = Any
THuggingFaceBatchEncoding = Any


class HFWrapper(IExtractor):
    """
    This is a wrapper for models from HuggingFace (transformers) library.
    Please, double check `forward` method to learn how we extrac features.
    """

    def __init__(self, model: THuggingFaceModel, feat_dim: int):
        super().__init__()
        self.model = model
        self._feat_dim = feat_dim

    def forward(self, x: THuggingFaceBatchEncoding) -> FloatTensor:  # type: ignore
        hf_output = self.model(x["input_ids"], attention_mask=x["attention_mask"])
        embedding = hf_output["last_hidden_state"][:, 0, :]
        return embedding

    def feat_dim(self) -> int:
        return self._feat_dim


__all_ = ["HuggingFaceWrapper"]
