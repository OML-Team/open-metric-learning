import torchvision.transforms as t
from torchvision.transforms import Compose, InterpolationMode, Normalize, ToTensor

from oml.const import MEAN, STD, TNormParam


def get_augs_torch(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> t.Compose:
    augs = t.Compose(
        [
            t.Resize(size=(im_size, im_size), antialias=True),
            t.RandomHorizontalFlip(),
            t.ColorJitter(),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ]
    )
    return augs


def get_augs_hypvit(im_size: int = 224, mean: TNormParam = MEAN, std: TNormParam = STD) -> t.Compose:
    augs = t.Compose(
        [
            t.RandomResizedCrop(im_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            t.RandomHorizontalFlip(),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ]
    )
    return augs


def get_normalisation_resize_hypvit(
    im_size: int = 224, crop_size: int = 224, mean: TNormParam = MEAN, std: TNormParam = STD
) -> t.Compose:
    transforms = t.Compose(
        [
            t.Resize(im_size, interpolation=InterpolationMode.BICUBIC),
            t.CenterCrop(crop_size),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ]
    )
    return transforms


def get_normalisation_torch(mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])


def get_normalisation_resize_torch(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([t.Resize(size=(im_size, im_size), antialias=True), ToTensor(), Normalize(mean=mean, std=std)])


__all__ = [
    "get_augs_torch",
    "get_normalisation_torch",
    "get_normalisation_resize_torch",
    "get_augs_hypvit",
    "get_normalisation_resize_hypvit",
]
