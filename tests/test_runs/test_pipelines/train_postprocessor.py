import hydra
from omegaconf import DictConfig

from oml.const import HYDRA_BEHAVIOUR, MOCK_DATASET_PATH
from oml.lightning.pipelines.train_postprocessor import postprocessor_training_pipeline
from oml.utils.download_mock_dataset import download_images_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="train_postprocessor.yaml", version_base=HYDRA_BEHAVIOUR)
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    download_images_mock_dataset(MOCK_DATASET_PATH)
    cfg["datasets"]["args"]["dataset_root"] = str(MOCK_DATASET_PATH)
    postprocessor_training_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
