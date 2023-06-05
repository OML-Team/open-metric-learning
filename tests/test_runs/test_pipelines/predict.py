import hydra
from omegaconf import DictConfig

from oml.const import MOCK_DATASET_PATH
from oml.lightning.pipelines.predict import extractor_prediction_pipeline
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.misc import dictconfig_to_dict


@hydra.main(config_path="configs", config_name="predict.yaml", version_base="1.1")
def main_hydra(cfg: DictConfig) -> None:
    cfg = dictconfig_to_dict(cfg)
    download_mock_dataset(MOCK_DATASET_PATH)
    cfg["data_dir"] = MOCK_DATASET_PATH
    extractor_prediction_pipeline(cfg)


if __name__ == "__main__":
    main_hydra()
