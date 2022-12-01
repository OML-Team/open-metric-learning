import hydra
from omegaconf import DictConfig

from examples.inference import inference
from oml.const import MOCK_DATASET_PATH
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="inference_images_mock.yaml")
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    if not MOCK_DATASET_PATH.exists():
        download_mock_dataset(MOCK_DATASET_PATH)
    cfg["dataset_root"] = MOCK_DATASET_PATH
    inference(cfg)


if __name__ == "__main__":
    main_hydra()
