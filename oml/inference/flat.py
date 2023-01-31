from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor, nn

from oml.const import PATHS_COLUMN, SPLIT_COLUMN
from oml.datasets.list_dataset import ListDataset
from oml.inference.abstract import _inference
from oml.interfaces.models import IExtractor
from oml.transforms.images.utils import TTransforms
from oml.utils.dataframe_format import check_retrieval_dataframe_format
from oml.utils.images.images import TImReader
from oml.utils.misc_torch import get_device


@torch.no_grad()
def inference_on_images(
    model: nn.Module,
    paths: List[Path],
    transform: TTransforms,
    num_workers: int,
    batch_size: int,
    verbose: bool = False,
    f_imread: Optional[TImReader] = None,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> Tensor:
    dataset = ListDataset(paths, bboxes=None, transform=transform, f_imread=f_imread, cache_size=0)
    device = get_device(model)

    def _apply(model_: nn.Module, batch_: Dict[str, Any]) -> Tensor:
        return model_(batch_[dataset.input_tensors_key].to(device))

    outputs = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
        use_fp16=use_fp16,
        accumulate_on_cpu=accumulate_on_cpu,
    )

    return outputs


def inference_on_dataframe(
    dataset_root: Union[Path, str],
    dataframe_name: str,
    extractor: IExtractor,
    transforms_extraction: TTransforms,
    output_cache_path: Optional[Union[str, Path]] = None,
    num_workers: int = 0,
    batch_size: int = 128,
    use_fp16: bool = False,
) -> Tuple[Tensor, Tensor, DataFrame, DataFrame]:
    df = pd.read_csv(Path(dataset_root) / dataframe_name)

    # it has now affect if paths are global already
    df[PATHS_COLUMN] = df[PATHS_COLUMN].apply(lambda x: Path(dataset_root) / x)

    check_retrieval_dataframe_format(df)

    if (output_cache_path is not None) and Path(output_cache_path).is_file():
        embeddings = torch.load(output_cache_path, map_location="cpu")
        print("Embeddings have been loaded from the disk.")
    else:
        embeddings = inference_on_images(
            model=extractor,
            paths=df[PATHS_COLUMN],
            transform=transforms_extraction,
            num_workers=num_workers,
            batch_size=batch_size,
            verbose=True,
            use_fp16=use_fp16,
            accumulate_on_cpu=True,
        )
        if output_cache_path is not None:
            torch.save(embeddings, output_cache_path)
            print("Embeddings have been saved to the disk.")

    train_mask = df[SPLIT_COLUMN] == "train"

    emb_train = embeddings[train_mask]
    emb_val = embeddings[~train_mask]

    df_train = df[train_mask]
    df_train.reset_index(inplace=True, drop=True)

    df_val = df[~train_mask]
    df_val.reset_index(inplace=True, drop=True)

    return emb_train, emb_val, df_train, df_val


__all__ = ["inference_on_images", "inference_on_dataframe"]
