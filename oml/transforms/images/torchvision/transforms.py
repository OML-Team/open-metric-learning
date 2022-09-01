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


def get_arcface_train_transforms(im_size: int = 224, mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return t.Compose(
        [
            t.Resize((256, 256), interpolation=t.InterpolationMode.LANCZOS),
            t.RandomResizedCrop(
                scale=(0.16, 1),
                ratio=(0.75, 1.33),
                size=im_size,
                interpolation=t.InterpolationMode.LANCZOS,
            ),
            t.RandomHorizontalFlip(),
            t.ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def get_arcface_test_transforms(im_size: int = 224, mean: TNormParam = MEAN, std: TNormParam = STD) -> Compose:
    return t.Compose(
        [
            t.Resize((256, 256), interpolation=t.InterpolationMode.LANCZOS),
            t.CenterCrop(im_size),
            t.ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


__all__ = ["get_augs_torch", "get_normalisation_torch", "get_normalisation_resize_torch"]
