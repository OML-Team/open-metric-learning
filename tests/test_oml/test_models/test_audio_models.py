import os
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from oml.datasets import AudioLabeledDataset
from oml.interfaces.models import IExtractor
from oml.models.audio.ecapa_tdnn.extractor import ECAPATDNNExtractor
from oml.utils import get_mock_audios_dataset


@pytest.mark.parametrize(
    "constructor,args",
    [
        (ECAPATDNNExtractor, {"normalise_features": False, "arch": "ecapa_tdnn_taoruijie"}),
    ],
)
def test_extractor(constructor: IExtractor, args: Dict[str, Any]) -> None:
    signal = torch.randn(1, 16000)

    extractor = constructor(weights=None, **args).eval()
    features1 = extractor.extract(signal)

    fname = "weights_tmp.pth"
    torch.save({"state_dict": extractor.state_dict()}, fname)

    extractor = ECAPATDNNExtractor(weights=fname, filter_state_dict_prefix="model.", **args).eval()
    features2 = extractor.extract(signal)
    Path(fname).unlink()

    assert features1.ndim == 2
    assert features1.shape[-1] == extractor.feat_dim
    assert torch.allclose(features1, features2)


@pytest.mark.long
@pytest.mark.skipif(os.getenv("DOWNLOAD_ZOO_IN_TESTS") != "yes", reason="It's a traffic consuming test.")
@pytest.mark.parametrize(
    "constructor,weights",
    [
        (ECAPATDNNExtractor, "ecapa_tdnn_taoruijie"),
    ],
)
def test_checkpoints_from_zoo(constructor: IExtractor, weights: str) -> None:

    df_train, _ = get_mock_audios_dataset(global_paths=True)
    dataset = AudioLabeledDataset(df_train)

    model = constructor.from_pretrained(weights).eval()
    emb1 = model.extract(dataset[0]["input_tensors"])
    emb2 = model.extract(dataset[1]["input_tensors"])
    emb3 = model.extract(dataset[3]["input_tensors"])

    sim12 = torch.nn.functional.cosine_similarity(emb1, emb2)
    sim13 = torch.nn.functional.cosine_similarity(emb1, emb3)

    assert dataset[0]["labels"] == dataset[1]["labels"]
    assert dataset[0]["labels"] != dataset[3]["labels"]

    assert sim12 > sim13
