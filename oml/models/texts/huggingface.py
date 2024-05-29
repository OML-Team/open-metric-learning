import warnings
from typing import Any

from torch import FloatTensor

from oml.interfaces.models import IExtractor

THFModel = Any
TBatchEncoding = Any


def check_model_type(model):  # type: ignore
    try:
        import transformers as t

        if not isinstance(model, (t.RobertaModel, t.BertModel, t.DistilBertModel)):
            warnings.warn(
                f"Unexpected model type: {type(model)}. Make sure your model is "
                f"compatible with {HFWrapper.__name__}."
            )
    except ImportError:
        pass


class HFWrapper(IExtractor):
    """
    This is a wrapper for models from HuggingFace (transformers) library.
    Please, double check `forward` method to learn how we extrac features.
    """

    def __init__(self, model: THFModel, feat_dim: int):
        super().__init__()
        check_model_type(model)

        self.model = model
        self._feat_dim = feat_dim

    def forward(self, x: TBatchEncoding) -> FloatTensor:  # type: ignore
        hf_output = self.model(x["input_ids"], attention_mask=x["attention_mask"])
        embedding = hf_output["last_hidden_state"][:, 0, :]
        return embedding

    @property
    def feat_dim(self) -> int:
        return self._feat_dim


__all_ = ["HuggingFaceWrapper"]
