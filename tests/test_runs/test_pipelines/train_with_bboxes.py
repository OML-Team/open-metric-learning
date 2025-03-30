import hydra
import torchvision.transforms as t
from omegaconf import DictConfig

from oml.const import HYDRA_BEHAVIOUR, MOCK_DATASET_PATH
from oml.lightning.pipelines.train import extractor_training_pipeline
from oml.registry.transforms import TRANSFORMS_REGISTRY
from oml.utils.download_mock_dataset import download_images_mock_dataset
from oml.utils.misc import dictconfig_to_dict


def get_custom_augs(im_size: int) -> t.Compose:
    return t.Compose(
        [
            t.Resize(size=(im_size, im_size)),
            t.RandomHorizontalFlip(),
            t.RandomGrayscale(),
            t.ToTensor(),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


TRANSFORMS_REGISTRY["custom_augmentations"] = get_custom_augs  # type: ignore


@hydra.main(config_path="configs", config_name="train_with_bboxes.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    download_images_mock_dataset(MOCK_DATASET_PATH)
    cfg["dataset_root"] = MOCK_DATASET_PATH
    extractor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
