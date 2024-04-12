from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as albu

import oml.models.vit_unicom.external.vision_transformer as unicom  # type: ignore
from oml.transforms.images.albumentations import (
    get_augs_albu,
    get_normalisation_albu,
    get_normalisation_resize_albu,
    get_normalisation_resize_albu_clip,
)
from oml.transforms.images.torchvision import (
    get_augs_hypvit,
    get_augs_torch,
    get_normalisation_resize_hypvit,
    get_normalisation_resize_torch,
    get_normalisation_torch,
)
from oml.transforms.images.utils import TTransforms, get_im_reader_for_transforms
from oml.utils.images.images import TImReader
from oml.utils.misc import TCfg, dictconfig_to_dict

TRANSFORMS_ALBU = {
    "augs_albu": get_augs_albu,
    "norm_albu": get_normalisation_albu,
    "norm_resize_albu": get_normalisation_resize_albu,
    "norm_resize_albu_clip": get_normalisation_resize_albu_clip,
}

TRANSFORMS_TORCH = {
    "augs_torch": get_augs_torch,
    "norm_torch": get_normalisation_torch,
    "norm_resize_torch": get_normalisation_resize_torch,
    "augs_hypvit_torch": get_augs_hypvit,
    "norm_resize_hypvit_torch": get_normalisation_resize_hypvit,
    "unicom_transforms": unicom.transform,  # type: ignore
}

TRANSFORMS_REGISTRY = {**TRANSFORMS_ALBU, **TRANSFORMS_TORCH}


def get_transforms(name: str, **kwargs: Dict[str, Any]) -> TTransforms:
    augs = TRANSFORMS_REGISTRY[name](**kwargs)  # type: ignore
    return augs


def get_transforms_by_cfg(cfg: TCfg) -> TTransforms:
    cfg = dictconfig_to_dict(cfg)
    return get_transforms(name=cfg["name"], **cfg["args"])


TRANSFORMS_FOR_PRETRAINED = {
    "resnet50_moco_v2": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "resnet18_imagenet1k_v1": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "resnet34_imagenet1k_v1": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "resnet50_imagenet1k_v1": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "resnet101_imagenet1k_v1": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "resnet152_imagenet1k_v1": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "vitb8_dino": get_normalisation_resize_torch(im_size=224),
    "vitb16_dino": get_normalisation_resize_torch(im_size=224),
    "vits8_dino": get_normalisation_resize_torch(im_size=224),
    "vits16_dino": get_normalisation_resize_torch(im_size=224),
    "vitb14_dinov2": get_normalisation_resize_hypvit(im_size=224),
    "vitb14_reg_dinov2": get_normalisation_resize_hypvit(im_size=224),
    "vitl14_dinov2": get_normalisation_resize_hypvit(im_size=224),
    "vitl14_reg_dinov2": get_normalisation_resize_hypvit(im_size=224),
    "vits14_dinov2": get_normalisation_resize_hypvit(im_size=224),
    "vits14_reg_dinov2": get_normalisation_resize_hypvit(im_size=224),
    "sber_vitb32_224": get_normalisation_resize_albu_clip(im_size=224),
    "sber_vitb16_224": get_normalisation_resize_albu_clip(im_size=224),
    "sber_vitl14_224": get_normalisation_resize_albu_clip(im_size=224),
    "openai_vitb32_224": get_normalisation_resize_albu_clip(im_size=224),
    "openai_vitb16_224": get_normalisation_resize_albu_clip(im_size=224),
    "openai_vitl14_224": get_normalisation_resize_albu_clip(im_size=224),
    "vits16_inshop": get_normalisation_resize_hypvit(im_size=224, crop_size=224),
    "vits16_sop": get_normalisation_resize_hypvit(im_size=224, crop_size=224),
    "vits16_cars": get_normalisation_resize_albu(im_size=224),
    "vits16_cub": get_normalisation_resize_albu(im_size=224),
    "vits16_224_mlp_384_inshop": get_normalisation_resize_hypvit(im_size=256, crop_size=224),
    "vitb32_unicom": unicom.transform(224),  # type: ignore
    "vitb16_unicom": unicom.transform(224),  # type: ignore
    "vitl14_unicom": unicom.transform(224),  # type: ignore
    "vitl14_336px_unicom": unicom.transform(336),  # type: ignore
}


def get_transforms_for_pretrained(weights: str) -> Tuple[TTransforms, TImReader]:
    transforms = TRANSFORMS_FOR_PRETRAINED[weights]
    im_reader = get_im_reader_for_transforms(transforms)
    return transforms, im_reader


def save_transforms_as_files(cfg: TCfg) -> List[Tuple[str, str]]:
    """
    Function saves transforms as files in local filesystem and returns list of tuples
    (transform_cfg_key, path_to_transform_file) for each transform
    """
    keys_files = []

    for key, val in cfg.items():
        if "transforms" in key:
            try:
                transforms = get_transforms_by_cfg(cfg[key])
                if isinstance(transforms, albu.Compose):
                    transforms_file = str(Path(".hydra/") / f"{key}.yaml") if Path(".hydra").exists() else f"{key}.yaml"
                    albu.save(filepath=transforms_file, transform=transforms, data_format="yaml")
                    keys_files.append((key, transforms_file))
            except Exception:
                print(f"We are not able to interpret {key} as albumentations transforms and log them as a file.")
    return keys_files


__all__ = [
    "TRANSFORMS_TORCH",
    "TRANSFORMS_ALBU",
    "TRANSFORMS_REGISTRY",
    "get_transforms",
    "get_transforms_by_cfg",
    "get_transforms_for_pretrained",
    "save_transforms_as_files",
]
