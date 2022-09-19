import torchvision.transforms as t
from torchvision.transforms import Compose, Normalize, ToTensor

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


def get_normalisation_torch(mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([ToTensor(), Normalize(mean=mean, std=std)])


def get_normalisation_resize_torch(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return Compose([t.Resize(size=(im_size, im_size), antialias=True), ToTensor(), Normalize(mean=mean, std=std)])


def get_clip_transforms(
    mean: TNormParam = (0.48145466, 0.4578275, 0.40821073),
    std: TNormParam = (0.26862954, 0.26130258, 0.27577711),
) -> Compose:
    return t.Compose(
        [
            t.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ]
    )


__all__ = ["get_augs_torch", "get_normalisation_torch", "get_normalisation_resize_torch", "get_clip_transforms"]
