import torchvision.transforms as t

from oml.const import MEAN, STD, TNormParam


def get_default_torch(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> t.Compose:
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


__all__ = ["get_default_torch"]
