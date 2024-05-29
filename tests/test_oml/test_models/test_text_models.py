import pytest
import torch
from torch.utils.data import default_collate

from oml.datasets import TextLabeledDataset
from oml.models.texts import HFWrapper
from oml.utils import get_mock_texts_dataset


@pytest.mark.needs_optional_dependency
@pytest.mark.parametrize(
    "model_name, feat_dim", [("bert-base-uncased", 768), ("roberta-base", 768), ("distilbert-base-uncased", 768)]
)
def test_padding_doesnt_affect_outputs(model_name: str, feat_dim: int) -> None:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    df_train, _ = get_mock_texts_dataset()
    model_cfg = AutoConfig.from_pretrained(model_name)
    extractor = HFWrapper(AutoModel.from_config(model_cfg), feat_dim).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train1 = TextLabeledDataset(df_train, tokenizer=tokenizer, max_length=200)
    batch1 = default_collate([train1[0], train1[1]])
    outputs1 = extractor(batch1["input_tensors"])

    train2 = TextLabeledDataset(df_train, tokenizer=tokenizer, max_length=300)
    batch2 = default_collate([train2[0], train2[1]])
    outputs2 = extractor(batch2["input_tensors"])

    assert torch.allclose(outputs1, outputs2)
    assert outputs1.shape[-1] == outputs2.shape[-1] == feat_dim
