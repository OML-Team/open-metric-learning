import albumentations as albu
import cv2

from oml.const import PAD_COLOR
from oml.transforms.images.albumentations.shared import TAugsList


def get_spatials_weak() -> TAugsList:
    spatial_augs = [
        albu.Perspective(scale=(0.06, 0.07), pad_mode=cv2.BORDER_CONSTANT, pad_val=PAD_COLOR, p=0.5),
        albu.Affine(
            scale=None, rotate=(-0.1, +0.1), translate_percent=(-0.05, 0.05), shear=(-7, 7), cval=PAD_COLOR, p=0.35
        ),
        albu.GridDistortion(distort_limit=0.2, border_mode=0, value=PAD_COLOR, p=0.15),
    ]
    return spatial_augs


def get_blurs_weak(blur_limit: int = 3) -> TAugsList:
    blur_augs = [
        albu.MotionBlur(blur_limit=blur_limit),
        albu.MedianBlur(blur_limit=blur_limit),
        albu.Blur(blur_limit=blur_limit),
        albu.GaussianBlur(blur_limit=(blur_limit, blur_limit)),
    ]
    return blur_augs


def get_colors_level_weak() -> TAugsList:
    color_augs = [
        albu.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        albu.CLAHE(clip_limit=3, p=0.2),
        albu.Sharpen(alpha=(0.2, 0.3), lightness=(0.2, 0.5), p=0.15),
        albu.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.15),
    ]
    return color_augs


def get_noises_weak() -> TAugsList:
    noise_augs = [
        albu.CoarseDropout(max_holes=3, max_height=10, max_width=10, fill_value=PAD_COLOR, p=0.3),
        albu.GaussNoise(p=0.7),
    ]
    return noise_augs


def get_noise_channels_weak() -> TAugsList:
    channels_noise_augs = [
        albu.ChannelDropout(p=0.1),
        albu.ToGray(p=0.7),
        albu.ChannelShuffle(p=0.2),
    ]
    return channels_noise_augs


def get_default_weak_albu() -> albu.Compose:
    augs = albu.Compose(
        [
            albu.HorizontalFlip(p=0.5),
            albu.OneOf(get_spatials_weak(), p=0.3),
            albu.OneOf(get_blurs_weak(), p=0.25),
            albu.OneOf(get_colors_level_weak(), p=0.4),
            albu.OneOf(get_noise_channels_weak(), p=0.1),
            albu.OneOf(get_noises_weak(), p=0.15),
        ]
    )
    return augs


__all__ = [
    "get_spatials_weak",
    "get_blurs_weak",
    "get_colors_level_weak",
    "get_noises_weak",
    "get_noise_channels_weak",
    "get_default_weak_albu",
]
