import albumentations as albu
import hydra
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

from oml.const import MEAN, STD, TNormParam
from oml.lightning.entrypoints.train import pl_train
from oml.registry.transforms import TRANSFORMS_REGISTRY
from oml.transforms.images.albumentations.transforms import (
    RandomSizedBBoxSafeCropPatched,
    get_blurs,
    get_colors_level,
    get_noise_channels,
    get_noises,
    get_spatials,
)


def get_augs(im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD) -> albu.Compose:
    augs = albu.Compose(
        [
            RandomSizedBBoxSafeCropPatched(im_size, erosion_rate=0.0),
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


TRANSFORMS_REGISTRY["inshop_augs_albu"] = get_augs


@hydra.main(config_path="configs", config_name="train_inshop.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
