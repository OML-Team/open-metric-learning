import albumentations as albu
import cv2
import hydra
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

from oml.const import MEAN, PAD_COLOR, STD, TNormParam
from oml.lightning.entrypoints.train import pl_train
from oml.registry import TRANSFORMS_REGISTRY
from oml.transforms.images.albumentations.transforms import (
    get_blurs,
    get_colors_level,
    get_noise_channels,
    get_noises,
    get_spatials,
)


def augs_albu_with_crop(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    augs = albu.Compose(
        [
            albu.LongestMaxSize(max_size=im_size),
            albu.PadIfNeeded(min_height=im_size, min_width=im_size, border_mode=cv2.BORDER_CONSTANT, value=PAD_COLOR),
            albu.RandomResizedCrop(width=im_size, height=im_size, scale=(0.8, 1.0), ratio=(0.8, 1.1), p=1.0),
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


TRANSFORMS_REGISTRY["augs_albu_with_crop"] = augs_albu_with_crop


@hydra.main(config_path="configs", config_name="train_sop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    print("Training model on SOP dataset.")
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
