import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

from oml.const import EMBEDDINGS_KEY, TCfg
from oml.datasets.images import (
    ImageLabeledDataset,
    ImageQueryGalleryLabeledDataset,
    get_retrieval_images_datasets,
)
from oml.inference import inference, inference_cached
from oml.interfaces.datasets import ILabeledDataset, IQueryGalleryLabeledDataset
from oml.registry.models import get_extractor_by_cfg
from oml.registry.transforms import get_transforms_by_cfg
from oml.utils.misc import flatten_dict


def get_image_datasets_by_cfg(**cfg: TCfg) -> Tuple[ILabeledDataset, IQueryGalleryLabeledDataset]:

    transforms_train = cfg.get("transforms_train", None)
    transforms_val = cfg.get("transforms_val", None)

    if transforms_train is not None:
        transforms_train = get_transforms_by_cfg(transforms_train)

    if transforms_val is not None:
        transforms_val = get_transforms_by_cfg(transforms_val)

    dataset_train, dataset_val = get_retrieval_images_datasets(
        dataset_root=Path(str(cfg["dataset_root"])),
        dataframe_name=str(cfg["dataframe_name"]),
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        cache_size=int(str(cfg.get("cache_size", 0))),
        verbose=bool(cfg.get("verbose", True)),
    )

    return dataset_train, dataset_val


def get_postprocessor_datasets_by_cfg(**cfg: TCfg) -> Tuple[ILabeledDataset, IQueryGalleryLabeledDataset]:

    transforms_extraction = get_transforms_by_cfg(cfg["transforms_extraction"])

    train_extraction, val_extraction = get_retrieval_images_datasets(
        dataset_root=Path(str(cfg["dataset_root"])),
        dataframe_name=str(cfg["dataframe_name"]),
        transforms_train=transforms_extraction,
        transforms_val=transforms_extraction,
    )

    args = {
        "model": get_extractor_by_cfg(cfg["feature_extractor"]).to(str(cfg["device"])),
        "num_workers": cfg["num_workers"],
        "batch_size": cfg["batch_size_inference"],
        "use_fp16": int(str(cfg.get("precision", 32))) == 16,
    }

    if cfg["embeddings_cache_dir"] is not None:
        hash_ = get_hash_of_extraction_stage_cfg(cfg)[:5]
        dir_ = Path(str(cfg["embeddings_cache_dir"]))
        emb_train = inference_cached(dataset=train_extraction, cache_path=str(dir_ / f"emb_train_{hash_}.pkl"), **args)
        emb_val = inference_cached(dataset=val_extraction, cache_path=str(dir_ / f"emb_val_{hash_}.pkl"), **args)
    else:
        emb_train = inference(dataset=train_extraction, **args)
        emb_val = inference(dataset=val_extraction, **args)

    train_dataset = ImageLabeledDataset(
        dataset_root=str(cfg["dataset_root"]),
        cache_size=int(str(cfg.get("cache_size", 0))),
        df=train_extraction.df,
        transform=get_transforms_by_cfg(cfg["transforms_train"]),
        extra_data={EMBEDDINGS_KEY: emb_train},
    )

    valid_dataset = ImageQueryGalleryLabeledDataset(
        dataset_root=str(cfg["dataset_root"]),
        cache_size=int(str(cfg.get("cache_size", 0))),
        df=val_extraction.df,
        transform=transforms_extraction,
        extra_data={EMBEDDINGS_KEY: emb_val},
    )

    return train_dataset, valid_dataset


# TODO: move to utils probably
def get_hash_of_extraction_stage_cfg(cfg: TCfg) -> str:
    def dict2str(dictionary: Dict[str, Any]) -> str:
        flatten_items = flatten_dict(dictionary).items()
        sorted(flatten_items, key=lambda x: x[0])
        return str(flatten_items)

    cfg_extraction_str = (
        dict2str(cfg["feature_extractor"])
        + dict2str(cfg["transforms_extraction"])
        + str(cfg["dataframe_name"])
        + str(cfg.get("precision", 32))
    )

    md5sum = hashlib.md5(cfg_extraction_str.encode("utf-8")).hexdigest()
    return md5sum


def get_image_datasets(
    dataset_name: str, **kwargs: Dict[str, Any]
) -> Tuple[ILabeledDataset, IQueryGalleryLabeledDataset]:
    return DATASET_BUILDER_REGISTRY[dataset_name](**kwargs)


DATASET_BUILDER_REGISTRY = {
    "oml_image_dataset": get_image_datasets_by_cfg,
    "oml_reranking_dataset": get_postprocessor_datasets_by_cfg,
}

__all__ = ["get_image_datasets", "DATASET_BUILDER_REGISTRY"]
