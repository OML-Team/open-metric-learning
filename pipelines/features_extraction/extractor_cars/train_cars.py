import albumentations as albu
import cv2
import hydra
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from packaging.version import Version

from oml.const import HYDRA_BEHAVIOUR
from oml.const import MEAN, PAD_COLOR, STD, TNormParam
from oml.lightning.pipelines.train import extractor_training_pipeline
from oml.registry import TRANSFORMS_REGISTRY
from oml.transforms.images.albumentations import TTransformsList

if Version(albu.__version__) > Version("1.3.1"):
    raise RuntimeError(f"Augmentations are designed for `albumentations` version 1.3.1, your version "
                       f"is {albu.__version__}. To run the training, downgrade the `albumentations` version to 1.3.1.")


def get_spatials() -> TTransformsList:
    spatial_augs = [
        albu.Perspective(scale=(0.06, 0.07), pad_mode=cv2.BORDER_CONSTANT, pad_val=PAD_COLOR),
        albu.Affine(
            scale=None,
            rotate=(-0.1, +0.1),
            translate_percent=(-0.05, 0.05),
            shear=(-7, 7),
            cval=PAD_COLOR,
        ),
        albu.GridDistortion(
            distort_limit=0.1,
            border_mode=0,
            value=PAD_COLOR,
        ),
    ]
    return spatial_augs


def get_blurs() -> TTransformsList:
    blur_augs = [
        albu.MotionBlur(),
        albu.MedianBlur(),
        albu.Blur(),
        albu.GaussianBlur(sigma_limit=(0.7, 2.0)),
    ]
    return blur_augs


def get_colors_level() -> TTransformsList:
    color_augs = [
        albu.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
        albu.HueSaturationValue(p=0.1),
        albu.CLAHE(p=0.1),
        albu.Sharpen(p=0.1),
        albu.Emboss(p=0.1),
        albu.RandomBrightnessContrast(p=0.1),
    ]
    return color_augs


def get_noises() -> TTransformsList:
    noise_augs = [
        albu.CoarseDropout(max_holes=3, max_height=20, max_width=20, fill_value=PAD_COLOR, p=0.3),
        albu.GaussNoise(p=0.7),
    ]
    return noise_augs


def get_noise_channels() -> TTransformsList:
    channels_noise_augs = [
        albu.ChannelDropout(p=0.1),
        albu.ToGray(p=0.8),
        albu.ChannelShuffle(p=0.1),
    ]
    return channels_noise_augs


def get_augs_albu(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    augs = albu.Compose(
        [
            albu.LongestMaxSize(max_size=im_size),
            albu.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=cv2.BORDER_CONSTANT, value=PAD_COLOR),
            albu.HorizontalFlip(p=0.5),
            albu.OneOf(get_spatials(), p=0.5),
            albu.OneOf(get_blurs(), p=0.5),
            albu.OneOf(get_colors_level(), p=0.8),
            albu.OneOf(get_noise_channels(), p=0.2),
            albu.OneOf(get_noises(), p=0.25),
            albu.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
    )
    return augs


TRANSFORMS_REGISTRY["augs_albu"] = get_augs_albu


@hydra.main(config_path=".", config_name="train_cars.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
