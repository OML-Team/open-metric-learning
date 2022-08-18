from typing import Tuple

import torchvision.transforms as transforms


def get_arcface_transform() -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomResizedCrop(
                scale=(0.16, 1),
                ratio=(0.75, 1.33),
                size=224,
            ),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.CenterCrop(224),
            normalize,
        ]
    )

    return train_transform, test_transform


def get_arcface_transform_train_only() -> transforms.Compose:
    return get_arcface_transform()[0]


def get_arcface_transform_val_only() -> transforms.Compose:
    return get_arcface_transform()[1]


__all__ = ["get_arcface_transform", "get_arcface_transform_train_only"]
