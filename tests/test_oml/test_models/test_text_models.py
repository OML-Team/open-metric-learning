import torch
from torch.utils.data import default_collate
from transformers import AutoModel, AutoTokenizer

from oml.datasets import TextLabeledDataset
from oml.models.texts import HFWrapper
from oml.utils import get_mock_texts_dataset


def test_padding_doesnt_affect_outputs() -> None:
    df_train, _ = get_mock_texts_dataset()
    extractor = HFWrapper(AutoModel.from_pretrained("bert-base-uncased"), 768).eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train1 = TextLabeledDataset(df_train, tokenizer=tokenizer, max_length=200)
    batch1 = default_collate([train1[0], train1[1]])
    outputs1 = extractor(batch1["input_tensors"])

    train2 = TextLabeledDataset(df_train, tokenizer=tokenizer, max_length=300)
    batch2 = default_collate([train2[0], train2[1]])
    outputs2 = extractor(batch2["input_tensors"])

    assert torch.allclose(outputs1, outputs2)
